import streamlit as st
from PIL import Image
import torch
from diffusers import StableDiffusionImg2ImgPipeline

# -------------------------------
# Display PyTorch info
# -------------------------------
st.title("AesthetIQ - AI Product Aesthetics Tester")
st.write("Check PyTorch & device info:")

st.write("Torch version:", torch.__version__)
device = "cuda" if torch.cuda.is_available() else "cpu"
st.write("Device:", device)
st.write("CUDA available:", torch.cuda.is_available())

# -------------------------------
# Load Stable Diffusion pipeline
# -------------------------------
@st.cache_resource(show_spinner=True)
def load_pipeline():
    pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5",
        torch_dtype=torch.float16 if device=="cuda" else torch.float32
    )
    pipe = pipe.to(device)
    # Disable NSFW filter
    pipe.safety_checker = lambda images, **kwargs: (images, [False]*len(images))
    return pipe

pipe = load_pipeline()

# -------------------------------
# Preprocess input image
# -------------------------------
def preprocess_image(img, size=(768, 768)):
    if img.mode != "RGB":
        img = img.convert("RGB")
    return img.resize(size)

# -------------------------------
# Generate AI variations
# -------------------------------
def generate_variations(input_image, prompt, num_variations=2):
    input_image = preprocess_image(input_image)
    results = []
    for _ in range(num_variations):
        output = pipe(
            prompt=prompt,
            image=input_image,
            strength=0.35,          # preserves original geometry
            guidanc
