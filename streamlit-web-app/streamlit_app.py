import streamlit as st
from streamlit_drawable_canvas import st_canvas
from PIL import Image, ImageOps, ImageFilter
import torch, torch.nn as nn
import torchvision.transforms as T
import numpy as np
import altair as alt
import skimage.metrics as metrics

# — Page config & gradient styling —
st.set_page_config(
    page_title="Enhanced Image Inpainting using Transformer-GAN Hybrid Models",
    page_icon="🖌️",
    layout="wide",
    initial_sidebar_state="expanded"
)
st.markdown("""
<style>
/* Full-page gradient background */
body {
  background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
}

/* Hide header/footer */
header, footer { display: none !important; }

/* Title styling */
h1 {
  font-family: 'Segoe UI', sans-serif;
  font-size: 2.5rem;
  color: white;
  text-align: center;
  margin-bottom: 0.5rem;
}

/* Sidebar gradient */
section[data-testid="stSidebar"] {
  background: linear-gradient(90deg, #ff7e5f 0%, #feb47b 100%);
  border-right: 1px solid #d0d4da;
  padding: 1rem;
}

/* Gradient buttons */
.stButton>button {
  background: linear-gradient(90deg, #6a9ef7 0%, #a18cd1 100%);
  color: white;
  font-weight: bold;
  font-color: white;
  border: none;
  padding: 0.6rem 1.2rem;
  border-radius: 8px;
  transition: opacity 0.2s ease;
}
.stButton>button:hover {
  opacity: 0.85;
}

/* Slider rail (unfilled background) */
.css-1aumxhk > div > div:nth-child(1) {
  background: linear-gradient(90deg, #e0e0e0, #c0c0c0) !important;
  border-radius: 8px;
}

/* Slider track (filled portion) */
.css-1aumxhk > div > div:nth-child(2) {
  background: linear-gradient(90deg, #6a9ef7 0%, #a18cd1 100%) !important;
  border-radius: 8px;
}

/* Slider thumb (handle) */
.css-1aumxhk input[type="range"]::-webkit-slider-thumb {
  background: white;
  border: 2px solid #6a9ef7;
  box-shadow: 0 2px 6px rgba(0,0,0,0.2);
}
.css-1aumxhk input[type="range"]::-moz-range-thumb {
  background: white;
  border: 2px solid #6a9ef7;
  box-shadow: 0 2px 6px rgba(0,0,0,0.2);
}

/* Subheaders */
.stSubheader {
  font-size: 1.4rem;
  color: #444;
  margin-top: 1.2rem;
  margin-bottom: 0.6rem;
}

/* Image cards */
.stImage > div {
  box-shadow: 0 4px 20px rgba(0,0,0,0.1);
  border-radius: 8px;
}
</style>
""", unsafe_allow_html=True)

st.title("Enhanced Image Inpainting using Transformer-GAN Hybrid Models")

# — Sidebar controls —
with st.sidebar:
    st.header("Controls")
    uploaded    = st.file_uploader("Upload Image", type=["png","jpg","jpeg"])
    mask_mode   = st.radio("Mask Mode", ["Random","Freehand"])
    mask_size   = st.slider("Random Mask Size", 32, 128, 64, step=16)
    brush_w     = st.slider("Brush Width", 5, 50, 20)
    run_inpaint = st.button("Run Inpainting")

if not uploaded:
    st.info("Please upload an image to get started.")
    st.stop()

# — Display original —
orig = Image.open(uploaded).convert("RGB")
orig = ImageOps.fit(orig, (224,224))
st.subheader("Original Image")
st.image(orig, use_container_width=True)

# — Draw mask if requested —
canvas_data = None
if mask_mode == "Freehand":
    st.subheader("Draw Mask")
    canvas_data = st_canvas(
        fill_color="rgba(255,0,0,0.3)",
        stroke_width=brush_w,
        background_image=orig,
        height=224, width=224,
        drawing_mode="freedraw",
        key="canvas"
    )

# — Load V4 model —
@st.cache_resource
def load_v4(path="inpaint_netG_v4.pth"):
    class V4(nn.Module):
        def __init__(self):
            super().__init__()
            self.encoder = nn.Sequential(
                nn.Conv2d(3,64,4,2,1), nn.ReLU(True),
                nn.Conv2d(64,128,4,2,1), nn.BatchNorm2d(128), nn.ReLU(True),
                nn.Conv2d(128,256,4,2,1), nn.BatchNorm2d(256), nn.ReLU(True),
            )
            self.decoder = nn.Sequential(
                nn.ConvTranspose2d(256,128,4,2,1), nn.BatchNorm2d(128), nn.ReLU(True),
                nn.ConvTranspose2d(128,64,4,2,1),  nn.BatchNorm2d(64),  nn.ReLU(True),
                nn.ConvTranspose2d(64,3,4,2,1),                nn.Tanh(),
            )
        def forward(self, x):
            return self.decoder(self.encoder(x))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = V4().to(device)
    ckpt  = torch.load(path, map_location=device)
    model.load_state_dict(ckpt, strict=False)
    model.eval()
    return model, device

# — Run inpainting —
if run_inpaint:
    # 1) Build mask
    if mask_mode == "Random":
        mask = torch.zeros((1,1,224,224), dtype=torch.float32)
        t,l = np.random.randint(0,224-mask_size,2)
        mask[:,:,t:t+mask_size,l:l+mask_size] = 1.0
    else:
        if canvas_data and canvas_data.image_data is not None:
            alpha = canvas_data.image_data[:,:,3]
            arr   = (alpha>0).astype(np.float32)
            mask  = torch.from_numpy(arr).unsqueeze(0).unsqueeze(0)
        else:
            st.error("Draw a mask first!")
            st.stop()

    # 2) Prepare tensors
    toT    = T.ToTensor()
    orig_t = toT(np.array(orig)/255.0).unsqueeze(0).float()
    masked = orig_t * (1-mask)
    normed = (masked * 2 - 1).float()

    st.subheader("Masked Input")
    st.image(T.ToPILImage()(masked.squeeze(0)), use_container_width=True)

    # 3) Inference
    model, dev = load_v4()
    with torch.no_grad():
        out = model(normed.to(dev)).cpu().clamp(-1,1)
    out = ((out + 1)/2).float()

    # 4) Feather edges & blend
    mask_pil     = Image.fromarray((mask.squeeze().numpy()*255).astype(np.uint8))
    feather_mask = mask_pil.filter(ImageFilter.GaussianBlur(radius=5))
    alpha        = np.array(feather_mask)/255.0
    alpha_boost  = np.clip(alpha * 0.85, 0.0, 1.0)[:,:,None]

    patch_pil       = Image.fromarray((out.squeeze(0).permute(1,2,0).numpy()*255).astype(np.uint8))
    patch_blur_pil  = patch_pil.filter(ImageFilter.GaussianBlur(radius=3))
    patch_blur_np   = np.array(patch_blur_pil)/255.0

    orig_np  = np.array(orig)/255.0
    final_np = alpha_boost * patch_blur_np + (1 - alpha_boost) * orig_np

    st.subheader("Inpainted Output")
    st.image(final_np, use_container_width=True)

    # 5) Metrics
    l1   = np.mean(np.abs(final_np - orig_np))
    ssim = metrics.structural_similarity(
                orig_np, final_np,
                channel_axis=2, win_size=7, data_range=1.0
           )
    psnr = metrics.peak_signal_noise_ratio(orig_np, final_np, data_range=1.0)

    c1, c2, c3 = st.columns(3)
    c1.metric("L1 Loss", f"{l1:.4f}")
    c2.metric("SSIM",    f"{ssim:.4f}")
    c3.metric("PSNR",    f"{psnr:.2f} dB")

    # 6) Error histogram
    errs = np.abs(final_np - orig_np).flatten()
    hist = (
      alt.Chart(alt.Data(values=[{"error": float(v)} for v in errs]))
         .mark_bar()
         .encode(
            alt.X("error:Q", bin=alt.Bin(maxbins=50), title="Absolute Error"),
            alt.Y("count():Q", title="Frequency")
         )
         .properties(width=700, height=300)
    )
    st.subheader("Error Distribution")
    st.altair_chart(hist, use_container_width=True)
