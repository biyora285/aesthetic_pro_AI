# -------------------------------
# 1️⃣ Install dependencies (run once)
# -------------------------------
# !pip install streamlit diffusers transformers accelerate safetensors torch torchvision pillow opencv-python --quiet

# -------------------------------
# 2️⃣ Imports
# -------------------------------
import streamlit as st
import torch
from diffusers import StableDiffusionImg2ImgPipeline
from PIL import Image
import os

# -------------------------------
# 3️⃣ Set page config
# -------------------------------
st.set_page_config(
    page_title="AesthetIQ - AI Product Aesthetics Tester",
    layout="wide"
)
st.title("AesthetIQ - AI Product Aesthetics Tester")
st.markdown(
    "Upload your product image (watch, backpack, etc.), type a photorealistic style prompt, and generate multiple AI variations."
)

# -------------------------------
# 4️⃣ Upload Product Image
# -------------------------------
uploaded_file = st.file_uploader("Upload Product Image (JPG/PNG)", type=["jpg","png"])
prompt = st.text_area("Enter Aesthetic Prompt (e.g., 'Steampunk smartwatch with brass gears')", height=80)
num_variations = st.slider("Number of Variations", min_value=1, max_value=6, value=4)

# -------------------------------
# 5️⃣ Load Pipeline (API-free)
# -------------------------------
@st.cache_resource(show_spinner=True)
def load_pipe():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5",
        torch_dtype=torch.float16 if device=="cuda" else torch.float32
    )
    pipe = pipe.to(device)
    pipe.safety_checker = lambda images, **kwargs: (images, [False]*len(images))
    return pipe, device

pipe, device = load_pipe()

# -------------------------------
# 6️⃣ Function to preprocess image
# -------------------------------
def preprocess_image(img, size=(512,512)):
    if img.mode != "RGB":
        img = img.convert("RGB")
    img = img.resize(size)
    return img

# -------------------------------
# 7️⃣ Generate Button
# -------------------------------
if uploaded_file and prompt:
    if st.button("Generate Variations"):
        input_image = Image.open(uploaded_file)
        input_image = preprocess_image(input_image)
        
        st.info("Generating images... This may take a minute or two depending on GPU availability.")
        generated_images = []
        for i in range(num_variations):
            generated = pipe(
                prompt=prompt,
                image=input_image,
                strength=0.35,
                guidance_scale=7.5,
                num_inference_steps=30
            ).images[0]
            generated_images.append(generated)

        # Display images in columns
        cols = st.columns(num_variations)
        for idx, img in enumerate(generated_images):
            with cols[idx % num_variations]:
                st.image(img, caption=f"Variation {idx+1}", use_column_width=True)
                # Save button for each image
                img_path = f"variation_{idx+1}.png"
                img.save(img_path)
                st.download_button(label=f"Download Variation {idx+1}", data=open(img_path,"rb"), file_name=img_path)

