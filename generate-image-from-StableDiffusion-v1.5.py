from diffusers import StableDiffusionPipeline

# Load Stable Diffusion-v1.5
pipe = StableDiffusionPipeline.from_pretrained("stable-diffusion-v1-5/stable-diffusion-v1-5").to("mps")

# Prompts
prompts = ["A photo of a cat sitting on a table.", "A dog playing in the park.", ]

# Gnerate pictures
images = []
for prompt in prompts:
    image = pipe(prompt, guidance_scale=7.5).images[0]  # guidance_scale
    images.append(image)

# Save the generative pictures
for i, image in enumerate(images):
    image.save(f"generated_image_{i}.png")
    print(f"Image {i} saved as generated_image_{i}.png")
