import streamlit as st
from diffusers import StableDiffusionPipeline
import torch
from PIL import Image
import numpy as np
from scipy.stats import entropy
import openai
import os

# -------------------------------
# Optional: Set OpenAI API Key
# -------------------------------
# Either set your API key in environment variables or paste here
# os.environ["OPENAI_API_KEY"] = "YOUR_OPENAI_API_KEY"
openai.api_key = os.getenv("73590d15-fde9-4be3-8c4b-8386f91efa52")

# -------------------------------
# Load Stable Diffusion Model
# -------------------------------
device = "cuda" if torch.cuda.is_available() else "cpu"
pipe = StableDiffusionPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    torch_dtype=torch.float16 if device=="cuda" else torch.float32
)
pipe = pipe.to(device)

# -------------------------------
# Aesthetic Scoring Function
# -------------------------------
def aesthetic_score(image):
    img = np.array(image.resize((256,256))) / 255.0
    brightness = np.mean(img)
    contrast = img.std()
    hist, _ = np.histogram(img, bins=256, range=(0,1))
    color_entropy = entropy(hist + 1e-7)
    edges = np.mean(np.abs(np.gradient(np.mean(img, axis=2))))
    score = (0.3*brightness + 0.3*contrast + 0.2*(1/color_entropy) + 0.2*edges)*100
    return round(score, 2)

# -------------------------------
# Function to Suggest Prompt Improvements
# -------------------------------
def improve_prompt(prompt):
    if not openai.api_key:
        return prompt  # skip improvement if no API key
    try:
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[{"role":"user", "content":
                       f"Suggest improvements to make this product prompt more visually appealing: {prompt}"}]
        )
        improved_prompt = response['choices'][0]['message']['content']
        return improved_prompt
    except:
        return prompt

# -------------------------------
# Streamlit Frontend
# -------------------------------
st.set_page_config(page_title="AI Aesthetics Tester", layout="centered")
st.title("🎨 AI-Generated Aesthetics Tester")
st.write("Generate product concept images and get interactive feedback to improve aesthetics.")

user_prompt = st.text_area("Enter Product Description:",
                           "A futuristic eco-friendly water bottle with smooth curves and metallic finish")

use_feedback = st.checkbox("Use AI prompt improvement")

if st.button("Generate & Test"):
    with st.spinner("Generating your product concept..."):
        # Improve prompt if enabled
        prompt_to_use = improve_prompt(user_prompt) if use_feedback else user_prompt
        if use_feedback:
            st.write("💡 AI Suggested Prompt:", prompt_to_use)
        
        # Generate image
        image = pipe(prompt_to_use).images[0]
        
        # Compute aesthetic score
        score = aesthetic_score(image)
        if score > 75:
            feedback = "✅ Visually appealing and well-balanced design."
        elif score >= 50:
            feedback = "⚙️ Decent design, but could improve contrast or color harmony."
        else:
            feedback = "❌ Needs refinement; lacks strong aesthetic appeal."
        
        st.image(image, caption="Generated Product Concept", use_column_width=True)
        st.subheader(f"✨ Aesthetic Score: {score}/100")
        st.write(feedback)

