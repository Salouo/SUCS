import torch
import matplotlib.pyplot as plt
import numpy as np
from transformers import CLIPProcessor, CLIPModel
from PIL import Image


def read_text_file(file_path):
    with open(file_path, "r") as file:
        return [line.strip() for line in file.readlines()]


def draw_comparison_bar_chart(sucs_dict, difficulty_levels):
    models = list(sucs_dict.keys())
    sucs_values = np.asarray(list(sucs_dict.values()))

    plt.figure(figsize=(10, 6))

    for i, model in enumerate(models):
        plt.plot(difficulty_levels, sucs_values[i], marker='o', label=model)

    plt.title('SUCS of Models Across Difficulty Levels', fontsize=16)
    plt.xlabel('Difficulty Levels', fontsize=14)
    plt.ylabel('SUCS', fontsize=14)
    
    plt.yticks(fontsize=12)
    plt.xticks(fontsize=12)

    plt.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.7)
    plt.legend(fontsize=12)

    plt.tight_layout()
    plt.show()


def main():
    # Load the pretrained CLIP(ViT-B/32)
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    model.to(device)

    # Name of generated models used in this report
    generative_model_name = ["DALL-E 3", "Stable Diffusion v1.4", "Stable Diffusion v1.5"]
    # difficulties: simple, medium, hard
    difficulties = ["simple", "medium", "hard"]

    # Load the text
    simple_texts = read_text_file("./data/texts/simple_texts.txt")
    medium_texts = read_text_file("./data/texts/medium_texts.txt")
    hard_texts = read_text_file("./data/texts/hard_texts.txt")
    texts = (simple_texts + medium_texts + hard_texts) * len(generative_model_name)

    # Number of image-text pairs for each difficulty
    num_simple = len(simple_texts)
    num_medium = len(medium_texts)
    num_hard = len(hard_texts)
    difficulties_num_dict = {"simple": num_simple,
                             "medium": num_medium,
                             "hard": num_hard}

    # Load pictures generated by generated models
    image_paths = []
    for name in generative_model_name:
        for diffucult in difficulties:
            for i in range(1, difficulties_num_dict[diffucult] + 1):
                image_path = "./" + "data" + "/" + "images" + "/" + name + "/" + diffucult + "/" + str(i) + ".png"
                image_paths.append(image_path)

    images = [Image.open(path).convert("RGB") for path in image_paths]

    # Convert the input to acceptable format
    inputs = processor(text=texts, images=images, return_tensors="pt", padding=True)
    inputs = {key: val.to(device) for key, val in inputs.items()}

    # Inference
    outputs = model(**inputs)
    image_features = outputs.image_embeds  # Image embedding
    text_features = outputs.text_embeds  # Text embedding

    # Normaliization
    image_features = image_features / image_features.norm(dim=1, keepdim=True)
    text_features = text_features / text_features.norm(dim=1, keepdim=True)

    # Compute the cosine similarity matrix and obtain the consine similarity of each text-image pair
    cosine_similarity = (image_features @ text_features.T).diag().cpu().detach().numpy()

    print("\n-----------------------------------------")
    sucs_dict = dict()
    for i, name in enumerate(generative_model_name):
        # Start index and end index of each model interval
        start = i * (num_simple + num_medium + num_hard)  # i*15
        end = (i + 1) * (num_simple + num_medium + num_hard)  # (i+1)*15

        print(f"Model name: {name}")
        print("\nScore for simple picture-text pairs:")
        sucs_simple = np.sqrt(cosine_similarity[start: start + num_simple].mean())
        print(f"{sucs_simple:.4f}")
        print("\nScore for medium picture-text pairs:")
        sucs_medium = np.sqrt(cosine_similarity[start + num_simple: start + num_simple + num_medium].mean())
        print(f"{sucs_medium:.4f}")
        print("\nScore for hard picture-text pairs:")
        sucs_hard = np.sqrt(cosine_similarity[start + num_simple + num_medium:
                                      start + num_simple + num_medium + num_hard].mean())
        print(f"{sucs_hard:.4f}")
        print("\nAverage Score:")
        sucs_average = (sucs_simple + sucs_medium + sucs_hard) / len(difficulties)
        print(f"{sucs_average:.4f}")
        print("-----------------------------------------")
        sucs_dict[name] = [sucs_simple, sucs_medium, sucs_hard]

    # draw the comparison bar chart
    draw_comparison_bar_chart(sucs_dict, difficulties)


if __name__ == '__main__':
    main()
