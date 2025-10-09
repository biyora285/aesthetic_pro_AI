import streamlit as st
from PIL import Image
import torch
from diffusers import StableDiffusionImg2ImgPipeline

# -------------------------------
# Load Stable Diffusion pipeline
# -------------------------------
device = "cuda" if torch.cuda.is_available() else "cpu"
pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    torch_dtype=torch.float16 if device == "cuda" else torch.float32
)
pipe = pipe.to(device)
# Disable NSFW filter for demo purposes
pipe.safety_checker = lambda images, **kwargs: (images, [False]*len(images))

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
            guidance_scale=7.5,
            num_inference_steps=50  # higher for clearer image
        ).images[0]
        results.append(output)
    return results

# -------------------------------
# Streamlit Interface
# -------------------------------
st.title("AesthetIQ - AI Product Aesthetics Tester")
st.write("Upload a product image, type a photorealistic prompt, and generate multiple AI variations.")

# Upload image
uploaded_image = st.file_uploader("Upload Product Image", type=["png", "jpg", "jpeg"])
prompt = st.text_input("Aesthetic Prompt", "Steampunk smartwatch with brass gears")
num_variations = st.slider("Number of Variations", min_value=1, max_value=4, value=2)

if uploaded_image and prompt:
    input_image = Image.open(uploaded_image)
    st.image(input_image, caption="Original Image", use_column_width=True)
    
    if st.button("Generate Variations"):
        with st.spinner("Generating AI variations..."):
            variations = generate_variations(input_image, prompt, num_variations)
        
        st.success("Variations Generated!")
        # Display results in columns
        cols = st.columns(len(variations))
        for col, img in zip(cols, variations):
            col.image(img, use_column_width=True)
