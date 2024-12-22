import os
import cv2
from tqdm import tqdm

# Input and output paths
base_input_dir = r"D:\segmodels_pytorch\actualimg_masks"
base_output_dir = r"D:\segmodels_pytorch"
resize_dim = (512, 512)  # Target dimension for resizing

# Define subdirectories
subdirs = {
    "images": ["train", "valid"],
    "masks": ["train", "valid"]
}

# Function to create directories
for key, dirs in subdirs.items():
    for subdir in dirs:
        os.makedirs(os.path.join(base_output_dir, key, subdir), exist_ok=True)

def resize_and_save(input_dir, output_dir, interpolation):
    """
    Resizes files in the input directory and saves them to the output directory.
    :param input_dir: Path to the input directory.
    :param output_dir: Path to the output directory.
    :param interpolation: Interpolation method for resizing.
    """
    for file_name in tqdm(os.listdir(input_dir), desc=f"Resizing in {input_dir}"):
        if file_name.endswith(('.png', '.jpg', '.jpeg')):
            input_path = os.path.join(input_dir, file_name)
            output_path = os.path.join(output_dir, file_name)

            # Read image
            image = cv2.imread(input_path, cv2.IMREAD_GRAYSCALE)
            if image is None:
                print(f"Could not read {file_name}")
                continue

            # Resize image
            resized_image = cv2.resize(image, resize_dim, interpolation=interpolation)

            # Save resized image
            cv2.imwrite(output_path, resized_image)

# Process images and masks
for data_type, dirs in subdirs.items():
    for subdir in dirs:
        input_dir = os.path.join(base_input_dir, data_type, subdir)
        output_dir = os.path.join(base_output_dir, data_type, subdir)

        if data_type == "images":
            resize_and_save(input_dir, output_dir, interpolation=cv2.INTER_CUBIC)
        elif data_type == "masks":
            resize_and_save(input_dir, output_dir, interpolation=cv2.INTER_NEAREST)

print("Resizing complete! Images and masks have been processed.")
