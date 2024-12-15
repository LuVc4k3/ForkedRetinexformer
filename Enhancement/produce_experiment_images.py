import os
import cv2
import yaml
import torch
import matplotlib.pyplot as plt
import numpy as np
from basicsr.models import create_model
from basicsr.utils.options import parse
from tqdm import tqdm


# Load configuration file
def load_config(config_path):
    with open(config_path, "r") as file:
        return yaml.safe_load(file)


# Load RetinexFormer model
def load_model(config):
    opt = parse(config["opt_path"], is_train=False)
    opt["dist"] = False
    model = create_model(opt).net_g
    checkpoint = torch.load(config["weights_path"])
    model.load_state_dict(checkpoint["params"])
    model = model.cuda().eval()
    return model


# Enhance image with a given model
def enhance_image(model, image):
    image_tensor = (
        torch.tensor(image / 255.0, dtype=torch.float32)
        .permute(2, 0, 1)
        .unsqueeze(0)
        .cuda()
    )
    with torch.no_grad():
        enhanced_tensor = model(image_tensor).clamp(0, 1)
    enhanced_image = (
        enhanced_tensor.squeeze(0).permute(1, 2, 0).cpu().numpy() * 255.0
    ).astype(np.uint8)
    return enhanced_image


# Plot results
def plot_results(images, labels, save_path):
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    for idx, (img, label) in enumerate(zip(images, labels)):
        ax = axes[idx // 4, idx % 4]
        ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        ax.set_title(label)
        ax.axis("off")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


# Main pipeline
def main_pipeline(config_file, lolv1_input, lolv1_target, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    config = load_config(config_file)

    # Load input and target images
    input_image = cv2.imread(lolv1_input)
    target_image = cv2.imread(lolv1_target)

    # Prepare results
    images = [input_image, target_image]
    labels = ["Original Image", "Target Image"]

    for experiment in tqdm(config["experiments"], desc="Processing Experiments"):
        model = load_model(experiment)
        enhanced_image = enhance_image(model, input_image)
        images.append(enhanced_image)
        labels.append(experiment["name"])

    # Plot and save results
    plot_results(images, labels, os.path.join(save_dir, "results_plot.png"))


if __name__ == "__main__":
    CONFIG_FILE = "Options/experiment_config.yml"
    LOLV1_INPUT = "data/LOLv2/Real_captured/Train/Low/00661.png"
    LOLV1_TARGET = "data/LOLv2/Real_captured/Train/Normal/00661.png"
    SAVE_DIR = "results/compilation_enhanced_images"
    main_pipeline(CONFIG_FILE, LOLV1_INPUT, LOLV1_TARGET, SAVE_DIR)
