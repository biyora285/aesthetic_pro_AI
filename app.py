import gradio as gr
import torch
from diffusers import StableDiffusionImg2ImgPipeline
from PIL import Image

# -------------------------------
# Load Stable Diffusion pipeline
# -------------------------------
device = "cuda" if torch.cuda.is_available() else "cpu"
pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    torch_dtype=torch.float16 if device=="cuda" else torch.float32
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
# Gradio Interface
# -------------------------------
demo = gr.Interface(
    fn=generate_variations,
    inputs=[
        gr.Image(type="pil", label="Upload Product Image"),
        gr.Textbox(label="Aesthetic Prompt", placeholder="Steampunk smartwatch with brass gears"),
        gr.Slider(minimum=1, maximum=4, step=1, value=2, label="Number of Variations")
    ],
    outputs=gr.Gallery(label="Generated Variations", show_label=True),
    title="AesthetIQ - AI Product Aesthetics Tester",
    description="Upload a product image, type a photorealistic prompt, and generate multiple AI variations."
)

if __name__ == "__main__":
    demo.launch()
