import streamlit as st
import torch
from diffusers import StableDiffusionImg2ImgPipeline
from PIL import Image

# -------------------------------
# Page setup
# -------------------------------
st.set_page_config(page_title="AesthetIQ - AI Product Aesthetics Tester", layout="wide")
st.title("AesthetIQ - AI Product Aesthetics Tester")
st.markdown(
    "Upload your product image (watch, backpack, etc.), type a photorealistic prompt, and generate multiple AI variations."
)

# -------------------------------
# Upload image
# -------------------------------
uploaded_file = st.file_uploader("Upload Product Image (JPG/PNG)", type=["jpg", "png"])
prompt = st.text_area("Enter Aesthetic Prompt (e.g., 'Steampunk smartwatch with brass gears')")
num_variations = st.slider("Number of Variations", 1, 6, 4)

# -------------------------------
# Load Stable Diffusion pipeline
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
# Preprocess image
# -------------------------------
def preprocess_image(img, size=(512,512)):
    if img.mode != "RGB":
        img = img.convert("RGB")
    return img.resize(size)

# -------------------------------
# Generate variations
# -------------------------------
if uploaded_file and prompt:
    if st.button("Generate Variations"):
        input_image = preprocess_image(Image.open(uploaded_file))
        st.info("Generating images... this may take a minute depending on GPU availability.")

        generated_images = []
        for i in range(num_variations):
            img = pipe(
                prompt=prompt,
                image=input_image,
                strength=0.35,
                guidance_scale=7.5,
                num_inference_steps=30
            ).images[0]
            generated_images.append(img)

        # Display images in columns
        cols = st.columns(num_variations)
        for idx, img in enumerate(generated_images):
            with cols[idx % num_variations]:
                st.image(img, caption=f"Variation {idx+1}", use_column_width=True)
                # Provide download button
                img_path = f"variation_{idx+1}.png"
                img.save(img_path)
                st.download_button(
                    label=f"Download Variation {idx+1}",
                    data=open(img_path, "rb"),
                    file_name=img_path
                )
