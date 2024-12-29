import torch
from torchvision import transforms
from segmentation_models_pytorch import Unet
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

# Load the model
def load_model(model_path, encoder_name="resnet34", encoder_weights="imagenet", num_classes=1):
    """
    Load the pre-trained model.
    :param model_path: Path to the .pth model file.
    :param encoder_name: Encoder architecture name.
    :param encoder_weights: Pre-trained weights for the encoder.
    :param num_classes: Number of output classes.
    :return: Loaded model.
    """
    model = Unet(
        encoder_name=encoder_name,
        encoder_weights=encoder_weights,
        in_channels=3,
        classes=num_classes
    )
    model.load_state_dict(torch.load(model_path, map_location=torch.device("cuda" if torch.cuda.is_available() else "cpu")))
    model.eval()  # Set model to evaluation mode
    model = model.cuda() if torch.cuda.is_available() else model
    return model

# Predict function
def predict_image(model, image_path, transform, threshold=0.5):
    """
    Perform prediction on a single image.
    :param model: Loaded model.
    :param image_path: Path to the input image.
    :param transform: Transformation to apply to the image.
    :param threshold: Threshold for binary segmentation.
    :return: Predicted mask.
    """
    image = Image.open(image_path).convert("RGB")  # Load and convert image to RGB
    original_size = image.size  # Save original image size
    input_tensor = transform(image).unsqueeze(0)  # Apply transformations and add batch dimension
    input_tensor = input_tensor.cuda() if torch.cuda.is_available() else input_tensor

    with torch.no_grad():
        output = model(input_tensor)  # Forward pass
        output = torch.sigmoid(output)  # Apply sigmoid for binary segmentation
        output = (output > threshold).float()  # Apply threshold

    predicted_mask = output.squeeze().cpu().numpy()  # Remove batch dimension and convert to numpy
    predicted_mask_resized = Image.fromarray((predicted_mask * 255).astype(np.uint8)).resize(original_size, Image.NEAREST)  # Resize to original image size
    return np.array(predicted_mask_resized)

# Visualization function
def visualize_prediction(image_path, predicted_mask, actual_mask_path=None):
    """
    Visualize the original image, predicted mask, and optionally the actual mask.
    :param image_path: Path to the input image.
    :param predicted_mask: Predicted mask as a numpy array.
    :param actual_mask_path: Path to the actual mask (if available).
    """
    original_image = Image.open(image_path).convert("RGB")

    plt.figure(figsize=(18, 6))

    # Original image
    plt.subplot(1, 3, 1)
    plt.imshow(original_image)
    plt.title("Original Image")
    plt.axis("off")

    # Predicted mask formatted like actual mask
    plt.subplot(1, 3, 2)
    custom_cmap = ListedColormap(["purple", "yellow"])  # 0 -> purple, 1 -> yellow
    plt.imshow(predicted_mask, cmap=custom_cmap)
    plt.title("Predicted Mask (Resized to Original)")
    plt.axis("off")

    # Actual mask (if provided)
    if actual_mask_path:
        actual_mask = Image.open(actual_mask_path).convert("L")
        plt.subplot(1, 3, 3)
        plt.imshow(actual_mask, cmap=custom_cmap)
        plt.title("Actual Mask")
        plt.axis("off")

    plt.show()

# Main script
if __name__ == "__main__":
    # Model and data paths
    model_path = r"D:\segmodels_pytorch\checkpoints\Unet_resnet34_imagenet_20241220_225048\512_16_100\epochs\best_epoch.pth"  # Path to your trained .pth model file
    imgsvalid_dir = r"D:\segmodels_pytorch\actualimg_masks\images/train"
    masksvalid_dir = r"D:\segmodels_pytorch\actualimg_masks\masks/train"
    image_path = imgsvalid_dir + "/496.png"
    actual_mask_path = masksvalid_dir + "/496.png"

    # Transformations
    transform = transforms.Compose([
        transforms.Resize((512, 512)),  # Resize to match the training image size
        transforms.ToTensor()  # Convert to PyTorch tensor
    ])

    # Load model
    model = load_model(model_path)

    # Predict
    predicted_mask = predict_image(model, image_path, transform)

    # Visualize
    visualize_prediction(image_path, predicted_mask, actual_mask_path)
