import os
import cv2
import numpy as np
from tqdm import tqdm
from torchvision import transforms
from PIL import Image

# Input and output directories
images_dir = "images/train"  # Original images directory
masks_dir = "masks/train"    # Original masks directory
output_images_dir = "augmented_data/images/train"  # Augmented images output directory
output_masks_dir = "augmented_data/masks/train"    # Augmented masks output directory

# Create output directories if they do not exist
os.makedirs(output_images_dir, exist_ok=True)
os.makedirs(output_masks_dir, exist_ok=True)

# Augmentation transformations
augmentation_transforms = transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=0.5),
    transforms.RandomRotation(degrees=30),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
])

def augment_and_save(image_path, mask_path, output_image_path, output_mask_path, index):
    """Applies augmentations to an image and its corresponding mask and saves them."""
    image = Image.open(image_path).convert("RGB")
    mask = Image.open(mask_path).convert("L")

    # Apply augmentations
    augmented_image = augmentation_transforms(image)
    augmented_mask = augmentation_transforms(mask)

    # Save augmented images and masks
    augmented_image.save(f"{output_image_path}/augmented_{index}.png")
    augmented_mask.save(f"{output_mask_path}/augmented_{index}.png")

# Loop through the dataset
index = 0
for image_file, mask_file in tqdm(zip(sorted(os.listdir(images_dir)), sorted(os.listdir(masks_dir))),
                                  desc="Augmenting data", total=len(os.listdir(images_dir))):
    image_path = os.path.join(images_dir, image_file)
    mask_path = os.path.join(masks_dir, mask_file)

    if image_file.endswith(('.png', '.jpg', '.jpeg')) and mask_file.endswith(('.png', '.jpg', '.jpeg')):
        # Save original image and mask to the augmented dataset
        original_image = Image.open(image_path).convert("RGB")
        original_mask = Image.open(mask_path).convert("L")
        original_image.save(f"{output_images_dir}/original_{index}.png")
        original_mask.save(f"{output_masks_dir}/original_{index}.png")

        # Perform augmentations
        for i in range(5):  # Number of augmentations per image
            augment_and_save(image_path, mask_path, output_images_dir, output_masks_dir, f"{index}_{i}")

        index += 1

print("Augmentation complete! Augmented data saved.")
