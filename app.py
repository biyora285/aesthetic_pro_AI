import streamlit as st
from PIL import Image

# -------------------------------
# Function to analyze dominant color
# -------------------------------
def analyze_image(img):
    img = img.convert("RGB")
    colors = img.getcolors(maxcolors=1000000)
    most_common_color = max(colors, key=lambda x: x[0])[1]
    return most_common_color

# -------------------------------
# Generate aesthetic text variations
# -------------------------------
def generate_aesthetic_text(input_image, prompt, num_variations=3):
    if not prompt.strip():
        prompt = "A modern product"

    dominant_color = analyze_image(input_image)

    templates = [
        "Variation {i}: {prompt}, featuring a {color_desc} finish, ambient lighting, and a {style_desc} environment.",
        "Variation {i}: {prompt} with {color_desc} material, realistic reflections, and {style_desc} background.",
        "Variation {i}: {prompt} crafted with {color_desc} texture, dramatic lighting, and placed in {style_desc} scene."
    ]

    style_variations = [
        "steampunk-inspired workshop",
        "futuristic high-tech lab",
        "minimalist modern showroom",
        "vintage antique shop",
        "brutalist architectural setting",
        "lush outdoor nature scene"
    ]

    color_desc = f"dominant color RGB{dominant_color}"

    variations = []
    for i in range(num_variations):
        template = templates[i % len(templates)]
        style_desc = style_variations[i % len(style_variations)]
        text = template.format(i=i + 1, prompt=prompt, color_desc=color_desc, style_desc=style_desc)
        variations.append(text)

    return "\n\n".join(variations)

# -------------------------------
# Streamlit UI
# -------------------------------
st.set_page_config(page_title="AesthetIQ - AI Product Ae_
