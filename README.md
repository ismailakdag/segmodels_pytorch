# PyTorch Segmentation Models

This repository contains a PyTorch implementation for image segmentation using the [segmentation_models.pytorch](https://github.com/qubvel/segmentation_models.pytorch) library. It provides a flexible and easy-to-use framework for training and evaluating various segmentation models.

## Features

- Support for various encoder architectures (ResNet, MobileNet, EfficientNet, etc.)
- Multiple loss functions (Dice, BCE, Jaccard, etc.)
- Training with customizable configurations
- Real-time GPU memory monitoring
- Checkpoint saving and loading
- Detailed metrics tracking
- Image augmentation support

## Installation

1. Clone this repository:
```bash
git clone <your-repo-url>
cd <repo-name>
```

2. Install the required packages:
```bash
pip install -r requirements.txt
```

## Project Structure

```
├── train.py           # Main training script
├── predict.py         # Inference script
├── config.json        # Configuration file
├── scripts/          # Utility scripts for data preprocessing
├── requirements.txt   # Python dependencies
└── .gitignore        # Git ignore file
```

## Configuration

Edit `config.json` to customize your training:

```json
{
    "train_images_dir": "path/to/train/images",
    "train_masks_dir": "path/to/train/masks",
    "valid_images_dir": "path/to/valid/images",
    "valid_masks_dir": "path/to/valid/masks",
    "model": {
        "architecture": "Unet",
        "encoder_name": "mobilenet_v2",
        "encoder_weights": "imagenet",
        "classes": 1
    },
    "batch_size": 8,
    "num_epochs": 100,
    "image_size": 512,
    "learning_rate": 0.001
}
```

## Usage

1. Prepare your dataset:
   - Organize your images and masks in separate directories
   - Update paths in `config.json`

2. Start training:
```bash
python train.py
```

3. For inference:
```bash
python predict.py --input path/to/image --model path/to/model
```

## Available Options

### Model Architectures
- Unet
- FPN
- DeepLabV3
- DeepLabV3+
- PSPNet

### Encoders
- ResNet (18, 34, 50, 101, 152)
- MobileNet (v2)
- EfficientNet (b0-b7)
- And more...

### Loss Functions
- DiceLoss
- BCEWithLogitsLoss
- CrossEntropyLoss
- JaccardLoss
- Combined losses

### Optimizers
- Adam
- SGD
- AdamW
- RMSprop

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- [segmentation_models.pytorch](https://github.com/qubvel/segmentation_models.pytorch) for the base implementation
- PyTorch team for the amazing deep learning framework
