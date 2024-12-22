# Segmentation Models PyTorch Framework

A comprehensive framework for image segmentation using PyTorch, featuring multiple state-of-the-art architectures and a user-friendly GUI interface.

## Technical Requirements

### Input Specifications
- **Image Dimensions**: Input dimensions must be multiples of 16 due to the architectural constraints of the encoder-decoder networks and feature pyramid networks.
- **Mask Format**: For binary segmentation tasks, masks should be binary matrices where:
  - Background pixels = 0
  - Foreground pixels = 1
  - Data type: uint8 or bool

### System Requirements
- CUDA-compatible GPU with minimum 4GB VRAM
- Python 3.8 or higher
- PyTorch 2.0 or higher

## Features

### Model Architectures
- **Available Architectures**:
  - U-Net
  - U-Net++
  - MAnet
  - Linknet
  - FPN (Feature Pyramid Network)
  - PSPNet (Pyramid Scene Parsing Network)
  - DeepLabV3
  - DeepLabV3+
  - PAN (Pyramid Attention Network)

- **Encoders**: Wide range of backbone networks including:
  - ResNet (18, 34, 50, 101, 152)
  - ResNeXt
  - SE-ResNet
  - DenseNet
  - EfficientNet (B0-B7)
  - And more...

### Graphical User Interfaces

#### 1. Training GUI
- Configure and monitor training sessions
- Features:
  - Model architecture and encoder selection
  - Training hyperparameter configuration
  - Real-time training progress visualization
  - GPU memory monitoring with safety limits
  - Checkpoint management system
  - Directory selection for training/validation data
  - Dynamic progress bars for epochs and batches

#### 2. Resize GUI
- Batch resize images and masks
- Features:
  - Independent width/height control
  - Automatic aspect ratio maintenance
  - Batch processing capability
  - Preview functionality
  - Dataset statistics display

#### 3. Augmentation GUI
- Configure and preview augmentations
- Features:
  - Real-time augmentation preview
  - Multiple augmentation techniques
  - Batch processing support
  - Parameter adjustment interface

## Installation

```bash
pip install -r requirements.txt
```

## Usage

### Starting the Training GUI
```bash
python scripts/train_gui.py
```

### Starting the Resize GUI
```bash
python scripts/resize_gui.py
```

### Starting the Augmentation GUI
```bash
python scripts/augmentation_gui.py
```

## Training Process

1. **Data Preparation**:
   - Organize images and masks in separate directories
   - Use Resize GUI to ensure consistent dimensions
   - Optionally apply augmentations using Augmentation GUI

2. **Configuration**:
   - Select model architecture and encoder
   - Set training parameters (batch size, learning rate, epochs)
   - Configure image dimensions (must be multiples of 16)
   - Set GPU memory safety limit

3. **Training**:
   - Start new training or continue from checkpoint
   - Monitor progress through GUI
   - View real-time metrics and GPU usage
   - Automatic checkpoint saving

4. **Checkpoints**:
   - Automatically saved in: `checkpoints/{architecture}_{encoder}_{weights}/{dimensions}_b{batch_size}/`
   - Continue training from latest checkpoint
   - Checkpoint naming format: `latest_checkpoint.pth`

## Project Structure

```
segmodels_pytorch/
├── checkpoints/          # Training checkpoints
├── scripts/             # GUI and utility scripts
├── data/                # Dataset directory
│   ├── images/         # Input images
│   └── masks/          # Segmentation masks
├── config.json         # Configuration file
├── requirements.txt    # Dependencies
└── README.md          # Documentation
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Segmentation Models PyTorch
- PyTorch
- PyQt5
