import os
import json
import time
import sys
import torch
from datetime import datetime
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from segmentation_models_pytorch import Unet
from segmentation_models_pytorch.losses import DiceLoss, JaccardLoss, BCEWithLogitsLoss
from segmentation_models_pytorch.utils.metrics import IoU
from PIL import Image
import pynvml

# Initialize NVIDIA management library for GPU memory monitoring
if torch.cuda.is_available():
    pynvml.nvmlInit()

def get_gpu_memory_usage():
    """Returns the current GPU memory usage and utilization percentage."""
    if torch.cuda.is_available():
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
        utilization = pynvml.nvmlDeviceGetUtilizationRates(handle)
        return mem_info.used / 1e9, utilization.gpu  # Return memory in GB and GPU utilization
    return 0, 0

# Dataset Class
class DentalDataset(Dataset):
    def __init__(self, images_dir, masks_dir, transform=None):
        """
        Initializes the dataset class.
        :param images_dir: Directory containing the input images.
        :param masks_dir: Directory containing the corresponding masks.
        :param transform: Transformations to apply to images and masks.
        """
        self.images_dir = images_dir
        self.masks_dir = masks_dir
        self.image_filenames = sorted(os.listdir(images_dir))  # Sorted list of image filenames
        self.mask_filenames = sorted(os.listdir(masks_dir))    # Sorted list of mask filenames
        self.transform = transform

    def __len__(self):
        """
        Returns the total number of samples in the dataset.
        """
        return len(self.image_filenames)

    def __getitem__(self, idx):
        """
        Retrieves a single sample (image and mask) by index.
        :param idx: Index of the sample.
        """
        image_path = os.path.join(self.images_dir, self.image_filenames[idx])
        mask_path = os.path.join(self.masks_dir, self.mask_filenames[idx])

        image = Image.open(image_path).convert("RGB")  # Load the image and convert to RGB
        mask = Image.open(mask_path).convert("L")      # Load the mask and convert to grayscale

        if self.transform:
            image = self.transform(image)  # Apply transformations to the image
            mask = self.transform(mask)    # Apply transformations to the mask

        mask = (mask > 0).float()  # Convert mask to binary (0 or 1)
        return image, mask

# Load Config Function
def load_config(config_path):
    """
    Loads the configuration file.
    :param config_path: Path to the config file.
    :return: Dictionary containing the configuration parameters.
    """
    with open(config_path, "r") as f:
        return json.load(f)

# Metric Calculations
def calculate_metrics(predictions, targets):
    """
    Calculate various metrics for binary segmentation.
    :param predictions: Predicted tensor.
    :param targets: Ground truth tensor.
    :return: Dictionary of metrics.
    """
    predictions = (predictions > 0.5).float()  # Convert logits to binary predictions
    intersection = (predictions * targets).sum()  # True positives
    union = predictions.sum() + targets.sum() - intersection  # Union for IoU
    precision = intersection / (predictions.sum() + 1e-6)  # Precision calculation
    recall = intersection / (targets.sum() + 1e-6)  # Recall calculation
    f1_score = 2 * (precision * recall) / (precision + recall + 1e-6)  # F1-Score
    accuracy = (predictions == targets).float().mean()  # Accuracy calculation
    dice_score = (2 * intersection) / (predictions.sum() + targets.sum() + 1e-6)  # Dice coefficient
    iou = intersection / (union + 1e-6)  # IoU calculation

    return {
        "IoU": iou.item(),
        "Precision": precision.item(),
        "Recall": recall.item(),
        "F1-Score": f1_score.item(),
        "Accuracy": accuracy.item(),
        "Dice Score": dice_score.item()
    }

def get_checkpoint_dir(config):
    """
    Get the checkpoint directory based on model configuration.
    This creates a unique identifier without timestamp for matching purposes.
    """
    return os.path.join("checkpoints",
                       f"{config['model']['architecture']}_{config['model']['encoder_name']}_{config['model']['encoder_weights']}",
                       f"{config['image_size']}_{config['batch_size']}")

def get_run_dir(config, checkpoint_dir):
    """
    Get the run directory path. If use_timestamp is False, it will be the same as checkpoint_dir.
    Otherwise, it will append a timestamp to the checkpoint_dir.
    """
    if config.get('use_timestamp', True):  # Default to True if not specified
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return f"{checkpoint_dir}_{timestamp}"
    return checkpoint_dir

def save_checkpoint(model, optimizer, epoch, best_metrics, best_loss, total_time, checkpoint_path):
    """
    Save a checkpoint of the training state.
    """
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'best_metrics': best_metrics,
        'best_loss': best_loss,
        'total_training_time': total_time  # Save accumulated training time
    }
    torch.save(checkpoint, checkpoint_path)

def load_checkpoint(checkpoint_path, model, optimizer):
    """
    Load a checkpoint and return the training state.
    Note: Using weights_only=True for security (prevents arbitrary code execution during loading)
    """
    checkpoint = torch.load(checkpoint_path, weights_only=True, map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    return (checkpoint['epoch'], checkpoint['best_metrics'], 
            checkpoint['best_loss'], checkpoint.get('total_training_time', 0))  # Get total time, default to 0

def load_results(results_file):
    try:
        with open(results_file, 'r') as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return None

def save_results(results_file, data):
    with open(results_file, 'w') as f:
        json.dump(data, f, indent=2)

# Main Code
if __name__ == "__main__":
    # Load configuration file
    config_path = "config.json"
    config = load_config(config_path)

    # Generate checkpoint directory (without timestamp) for matching
    checkpoint_dir = get_checkpoint_dir(config)
    os.makedirs(checkpoint_dir, exist_ok=True)  # Create checkpoint directory if it doesn't exist
    
    # Generate run directory (with or without timestamp based on config)
    run_dir = get_run_dir(config, checkpoint_dir)
    
    epochs_dir = os.path.join(run_dir, "epochs")
    results_dir = os.path.join(run_dir, "results")
    os.makedirs(epochs_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)

    # Transforms
    transform = transforms.Compose([
        # transforms.Resize((config["image_size"], config["image_size"])),  # Resize images and masks
        transforms.ToTensor()  # Convert images and masks to PyTorch tensors
    ])

    # Datasets and DataLoaders
    train_dataset = DentalDataset(config["train_images_dir"], config["train_masks_dir"], transform=transform)
    valid_dataset = DentalDataset(config["valid_images_dir"], config["valid_masks_dir"], transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=config["batch_size"], shuffle=True)  # DataLoader for training
    valid_loader = DataLoader(valid_dataset, batch_size=config["batch_size"], shuffle=False) # DataLoader for validation

    # Print dataset information
    print(f"Total Images in Training Set: {len(train_dataset)}")
    print(f"Total Images in Validation Set: {len(valid_dataset)}")
    print(f"Batch Size: {config['batch_size']}")
    print(f"Image Size: {config['image_size']}x{config['image_size']}")
    print(f"Starting epochs: {config['num_epochs']}")
    if torch.cuda.is_available():
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
        print(f"Total GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    else:
        print("Using CPU")

    # Check for existing checkpoint and results
    checkpoint_file = os.path.join(checkpoint_dir, "latest_checkpoint.pth")
    results_file = os.path.join(checkpoint_dir, "results/results.json")
    start_epoch = 1
    total_training_time = 0

    # Create results directory if it doesn't exist
    os.makedirs(os.path.dirname(results_file), exist_ok=True)

    # Load or initialize results
    results = load_results(results_file)
    if results is None:
        results = {
            "dataset_info": {
                "train_size": len(train_dataset),
                "valid_size": len(valid_dataset),
                "batch_size": config['batch_size'],
                "image_size": config['image_size']
            },
            "hardware_info": {
                "device": "GPU: " + torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU",
                "gpu_memory": f"{torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB" if torch.cuda.is_available() else None
            },
            "training_info": {
                "architecture": config['model']['architecture'],
                "encoder": config['model']['encoder_name'],
                "pretrained_weights": config['model']['encoder_weights'],
                "image_size": config['image_size'],
                "batch_size": config['batch_size'],
                "initial_epochs": config['num_epochs'],
                "start_time": datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            },
            "epochs": [],
            "continuations": []
        }

    # Initialize model, criterion and optimizer
    model = Unet(
        encoder_name=config["model"]["encoder_name"],       # Encoder architecture (e.g., resnet34, mobilenet_v2, efficientnet-b0)
        encoder_weights=config["model"]["encoder_weights"], # Pretrained weights (e.g., imagenet, None)
        in_channels=3,                                      # Input channels (RGB images)
        classes=config["model"]["classes"]                  # Number of output classes (binary segmentation)
    )
    model = model.cuda() if torch.cuda.is_available() else model  # Move model to GPU if available

    # Loss Function Options:
    # 1. DiceLoss(mode='binary'/'multiclass'/'multilabel') - Good for imbalanced datasets
    # 2. BCEWithLogitsLoss() - Binary Cross Entropy, good for binary segmentation
    # 3. CrossEntropyLoss() - For multiclass segmentation
    # 4. JaccardLoss() - IoU-based loss, good for segmentation
    # 5. Combined losses: e.g., criterion = 0.5 * DiceLoss() + 0.5 * BCEWithLogitsLoss()
    criterion = DiceLoss(mode="binary")  # Initialize DiceLoss for binary segmentation

    # Optimizer Options:
    # 1. Adam: Good default choice, adapts learning rate per parameter
    #    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    # 2. SGD: Often used with learning rate scheduling
    #    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=1e-4)
    # 3. AdamW: Adam with better weight decay, good for large models
    #    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-2)
    # 4. RMSprop: Good for RNNs and some computer vision tasks
    #    optimizer = torch.optim.RMSprop(model.parameters(), lr=1e-3, weight_decay=1e-4)
    optimizer = torch.optim.Adam(model.parameters(), lr=config["learning_rate"])

    # Learning Rate Scheduler Options (can be added):
    # 1. ReduceLROnPlateau: Reduces LR when metric plateaus
    #    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=3)
    # 2. CosineAnnealingLR: Cycles LR between maximum and minimum values
    #    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)
    # 3. StepLR: Decays LR by gamma every step_size epochs
    #    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)

    # Now check for checkpoint
    if os.path.exists(checkpoint_file):
        print(f"Found existing checkpoint. Checking compatibility...")
        
        # Load the model and optimizer state
        start_epoch, best_metrics, best_loss, total_training_time = load_checkpoint(
            checkpoint_file, model, optimizer)
        print(f"Resuming from epoch {start_epoch}")
        print(f"Total training time so far: {total_training_time:.2f}s")
        
        # Add continuation marker
        results["continuations"].append({
            "epoch": start_epoch,
            "time": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            "total_epochs": config['num_epochs']
        })
        save_results(results_file, results)
    else:
        best_metrics = None
        best_loss = float('inf')

    best_epoch = -1

    # Training Loop
    session_start_time = time.time()
    for epoch in range(start_epoch, config["num_epochs"] + 1):
        model.train()  # Set model to training mode
        train_loss = 0
        train_metrics = {"IoU": 0, "Precision": 0, "Recall": 0, "F1-Score": 0, "Accuracy": 0, "Dice Score": 0}

        epoch_start_time = time.time()

        for batch_idx, (images, masks) in enumerate(train_loader):
            images, masks = images.cuda(), masks.cuda()  # Move data to GPU if available

            optimizer.zero_grad()  # Clear gradients
            outputs = model(images)  # Forward pass
            loss = criterion(outputs, masks)  # Compute loss
            loss.backward()  # Backward pass
            optimizer.step()  # Update model parameters

            train_loss += loss.item()  # Accumulate loss

            # Update metrics
            batch_metrics = calculate_metrics(outputs, masks)
            for key in train_metrics:
                train_metrics[key] += batch_metrics[key]

            # GPU memory usage
            gpu_memory_used, gpu_utilization = get_gpu_memory_usage()

            # Print progress (overwrite line)
            progress = (batch_idx + 1) / len(train_loader) * 100
            sys.stdout.write(f"\rEpoch {epoch}, Batch {batch_idx + 1}/{len(train_loader)} ({progress:.2f}%): Loss {loss.item():.4f} | Used GPU Memory: {gpu_memory_used:.2f} GB ({gpu_utilization:.2f}%)")
            sys.stdout.flush()

        print()  # Move to the next line after epoch progress

        # Average training metrics over the epoch
        train_loss /= len(train_loader)
        for key in train_metrics:
            train_metrics[key] /= len(train_loader)

        # Validation
        model.eval()  # Set model to evaluation mode
        valid_loss = 0
        valid_metrics = {"IoU": 0, "Precision": 0, "Recall": 0, "F1-Score": 0, "Accuracy": 0, "Dice Score": 0}

        with torch.no_grad():
            for images, masks in valid_loader:
                images, masks = images.cuda(), masks.cuda()  # Move data to GPU if available

                outputs = model(images)  # Forward pass
                loss = criterion(outputs, masks)  # Compute loss
                valid_loss += loss.item()  # Accumulate validation loss

                # Update validation metrics
                batch_metrics = calculate_metrics(outputs, masks)
                for key in valid_metrics:
                    valid_metrics[key] += batch_metrics[key]

            valid_loss /= len(valid_loader)
            for key in valid_metrics:
                valid_metrics[key] /= len(valid_loader)

        epoch_duration = time.time() - epoch_start_time

        # Print metrics after each epoch
        print(f"Epoch {epoch} Metrics:")
        print(f"  Train Loss: {train_loss:.4f}")
        print(f"  Valid Loss: {valid_loss:.4f}")
        for key in train_metrics:
            print(f"  Train {key}: {train_metrics[key]:.4f}")
        for key in valid_metrics:
            print(f"  Valid {key}: {valid_metrics[key]:.4f}")

        # Save epoch results
        epoch_data = {
            "epoch": epoch,
            "train_loss": train_loss,
            "valid_loss": valid_loss,
            "train_metrics": train_metrics,
            "valid_metrics": valid_metrics,
            "time": epoch_duration
        }
        results["epochs"].append(epoch_data)
        save_results(results_file, results)

        # Save checkpoint after each epoch
        epoch_total_time = total_training_time + (time.time() - session_start_time)
        save_checkpoint(model, optimizer, epoch + 1, best_metrics, best_loss, 
                       epoch_total_time, os.path.join(checkpoint_dir, "latest_checkpoint.pth"))
        
        # Save model state for this epoch
        torch.save(model.state_dict(), os.path.join(epochs_dir, f"epoch_{epoch}.pth"))
        
        if valid_loss < best_loss:
            best_loss = valid_loss
            best_epoch = epoch
            best_metrics = valid_metrics
            torch.save(model.state_dict(), os.path.join(epochs_dir, "best_epoch.pth"))

    session_time = time.time() - session_start_time
    total_training_time += session_time

    # Finalize Results
    with open(results_file, 'a') as f:
        results["training_info"]["end_time"] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        results["training_info"]["total_time"] = session_time
        results["training_info"]["total_epochs"] = config['num_epochs']
        results["training_info"]["best_epoch"] = best_epoch
        results["training_info"]["best_metrics"] = best_metrics
        if torch.cuda.is_available():
            total_memory_needed, _ = get_gpu_memory_usage()
            results["hardware_info"]["peak_gpu_memory"] = f"{total_memory_needed:.2f} GB"
        save_results(results_file, results)

    print(f"Training completed. Best epoch: {best_epoch}. Model saved at {run_dir}")
    print(f"Total training time across all sessions: {total_training_time:.2f}s")
