import sys
import os
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                           QHBoxLayout, QPushButton, QLabel, QFileDialog, 
                           QCheckBox, QProgressBar, QMessageBox, QTextEdit,
                           QGroupBox, QScrollArea)
from PyQt5.QtCore import Qt, QThread, pyqtSignal
import cv2
import numpy as np
from PIL import Image
import albumentations as A
from pathlib import Path
import random
from PyQt5.QtGui import QColor, QPalette, QFont, QPixmap, QImage
import io
import shutil

class AugmentationWorker(QThread):
    progress = pyqtSignal(int)
    status = pyqtSignal(str)
    finished = pyqtSignal()

    def __init__(self, input_path, output_path, options):
        super().__init__()
        self.input_path = input_path
        self.output_path = output_path
        self.options = options
        self.total_files = 0
        self.processed_files = 0
        
    def run(self):
        try:
            # Calculate total files and transformations
            splits = []
            if self.options['process_train']:
                splits.append('train')
            if self.options['process_valid']:
                splits.append('valid')

            # Count total files to process
            for split in splits:
                images_dir = os.path.join(self.input_path, 'images', split)
                if os.path.exists(images_dir):
                    files = [f for f in os.listdir(images_dir) 
                            if f.endswith(('.png', '.jpg', '.jpeg'))]
                    self.total_files += len(files)

            # Create transforms list
            transforms = []
            if self.options['horizontal_flip']:
                transforms.append(A.HorizontalFlip(p=1))
            if self.options['vertical_flip']:
                transforms.append(A.VerticalFlip(p=1))
            if self.options['rotate90']:
                transforms.append(A.RandomRotate90(p=1))
            if self.options['rotation']:
                transforms.append(A.Rotate(limit=30, p=1))
            if self.options['brightness_contrast']:
                transforms.append(A.RandomBrightnessContrast(p=1))
            if self.options['gaussian_noise']:
                transforms.append(A.GaussNoise(var_limit=(10.0, 50.0), p=1))
            if self.options['gaussian_blur']:
                transforms.append(A.GaussianBlur(blur_limit=(3, 7), p=1))
            if self.options['elastic']:
                transforms.append(A.ElasticTransform(alpha=120, sigma=120 * 0.05, alpha_affine=120 * 0.03, p=1))
            if self.options['grid_distortion']:
                transforms.append(A.GridDistortion(p=1))
            if self.options['clahe']:
                transforms.append(A.CLAHE(clip_limit=4.0, tile_grid_size=(8, 8), p=1))
            if self.options['random_gamma']:
                transforms.append(A.RandomGamma(gamma_limit=(80, 120), p=1))
            if self.options['cutout']:
                transforms.append(A.CoarseDropout(max_holes=8, max_height=32, max_width=32, p=1))

            # Process each split
            for split in splits:
                images_dir = os.path.join(self.input_path, 'images', split)
                masks_dir = os.path.join(self.input_path, 'masks', split)
                
                if not os.path.exists(images_dir) or not os.path.exists(masks_dir):
                    continue

                # Create output directories
                output_images_dir = os.path.join(self.output_path, 'images', split)
                output_masks_dir = os.path.join(self.output_path, 'masks', split)
                os.makedirs(output_images_dir, exist_ok=True)
                os.makedirs(output_masks_dir, exist_ok=True)

                # Get list of files
                files = [f for f in os.listdir(images_dir) 
                        if f.endswith(('.png', '.jpg', '.jpeg'))]

                # Copy original files first
                for filename in files:
                    img_path = os.path.join(images_dir, filename)
                    mask_path = os.path.join(masks_dir, filename)
                    
                    # Copy original files
                    shutil.copy2(img_path, os.path.join(output_images_dir, filename))
                    shutil.copy2(mask_path, os.path.join(output_masks_dir, filename))
                    
                    self.processed_files += 1
                    progress = int((self.processed_files / (self.total_files * (len(transforms) + 1))) * 100)
                    self.progress.emit(progress)
                    self.status.emit(f'Processing {split} set: {progress}%')

                # Apply each transform
                for i, transform in enumerate(transforms):
                    for filename in files:
                        img_path = os.path.join(images_dir, filename)
                        mask_path = os.path.join(masks_dir, filename)
                        
                        # Read images
                        image = cv2.imread(img_path)
                        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
                        
                        # Apply transform
                        transformed = transform(image=image, mask=mask)
                        aug_image = cv2.cvtColor(transformed['image'], cv2.COLOR_RGB2BGR)
                        aug_mask = transformed['mask']
                        
                        # Save augmented images
                        base_name = os.path.splitext(filename)[0]
                        ext = os.path.splitext(filename)[1]
                        aug_filename = f"{base_name}_aug_{i+1}{ext}"
                        
                        cv2.imwrite(os.path.join(output_images_dir, aug_filename), aug_image)
                        cv2.imwrite(os.path.join(output_masks_dir, aug_filename), aug_mask)
                        
                        self.processed_files += 1
                        progress = int((self.processed_files / (self.total_files * (len(transforms) + 1))) * 100)
                        self.progress.emit(progress)
                        self.status.emit(f'Processing {split} set: {progress}%')

            self.status.emit('Augmentation complete!')
            self.progress.emit(100)
            self.finished.emit()
            
        except Exception as e:
            self.status.emit(f'Error: {str(e)}')
            self.finished.emit()

class PreviewWindow(QWidget):
    def __init__(self, image, mask, augmented_image, augmented_mask):
        super().__init__()
        self.setWindowTitle("Augmentation Preview")
        self.setMinimumSize(800, 400)
        
        layout = QHBoxLayout()
        
        # Original images
        original_group = QGroupBox("Original")
        original_layout = QVBoxLayout()
        original_group.setLayout(original_layout)
        
        image_label = QLabel()
        image_label.setPixmap(self.array_to_pixmap(image))
        mask_label = QLabel()
        mask_label.setPixmap(self.array_to_pixmap(mask))
        
        original_layout.addWidget(QLabel("Image:"))
        original_layout.addWidget(image_label)
        original_layout.addWidget(QLabel("Mask:"))
        original_layout.addWidget(mask_label)
        
        # Augmented images
        augmented_group = QGroupBox("Augmented")
        augmented_layout = QVBoxLayout()
        augmented_group.setLayout(augmented_layout)
        
        aug_image_label = QLabel()
        aug_image_label.setPixmap(self.array_to_pixmap(augmented_image))
        aug_mask_label = QLabel()
        aug_mask_label.setPixmap(self.array_to_pixmap(augmented_mask))
        
        augmented_layout.addWidget(QLabel("Image:"))
        augmented_layout.addWidget(aug_image_label)
        augmented_layout.addWidget(QLabel("Mask:"))
        augmented_layout.addWidget(aug_mask_label)
        
        layout.addWidget(original_group)
        layout.addWidget(augmented_group)
        self.setLayout(layout)
    
    def array_to_pixmap(self, img):
        if len(img.shape) == 2:  # Mask
            img = cv2.cvtColor(img.astype(np.uint8) * 255, cv2.COLOR_GRAY2RGB)
        h, w = img.shape[:2]
        scale = min(300/h, 300/w)
        new_size = (int(w*scale), int(h*scale))
        img = cv2.resize(img, new_size)
        height, width, channel = img.shape
        bytes_per_line = 3 * width
        qimg = QPixmap.fromImage(QImage(img.data, width, height, bytes_per_line, QImage.Format_RGB888))
        return qimg

class AugmentationGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.worker = None
        self.initUI()
        
    def initUI(self):
        self.setWindowTitle('Medical Image Augmentation Tool')
        self.setStyleSheet("""
            QMainWindow {
                background-color: #f0f0f0;
            }
            QGroupBox {
                background-color: white;
                border-radius: 5px;
                margin-top: 10px;
                font-weight: bold;
                padding: 10px;
            }
            QGroupBox::title {
                color: #2c3e50;
                subcontrol-origin: margin;
                left: 10px;
            }
            QPushButton {
                background-color: #3498db;
                color: white;
                border-radius: 3px;
                padding: 5px;
                min-width: 80px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #2980b9;
            }
            QProgressBar {
                border: 2px solid #bdc3c7;
                border-radius: 5px;
                text-align: center;
                height: 20px;
            }
            QProgressBar::chunk {
                background-color: #2ecc71;
            }
            QCheckBox {
                spacing: 5px;
                padding: 2px;
            }
            QCheckBox::indicator {
                width: 15px;
                height: 15px;
            }
            QLabel {
                color: #2c3e50;
            }
        """)
        
        # Create central widget and main layout
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        layout = QVBoxLayout(central_widget)
        
        # Directory selection
        dir_group = QGroupBox("Directory Selection")
        dir_layout = QVBoxLayout()
        
        # Input directory
        input_layout = QHBoxLayout()
        self.input_label = QLabel("Input Directory:")
        self.input_path = QLabel("Not selected")
        input_btn = QPushButton("Browse")
        input_btn.clicked.connect(lambda: self.select_directory('input'))
        input_layout.addWidget(self.input_label)
        input_layout.addWidget(self.input_path)
        input_layout.addWidget(input_btn)
        
        # Output directory
        output_layout = QHBoxLayout()
        self.output_label = QLabel("Output Directory:")
        self.output_path = QLabel("Not selected")
        output_btn = QPushButton("Browse")
        output_btn.clicked.connect(lambda: self.select_directory('output'))
        output_layout.addWidget(self.output_label)
        output_layout.addWidget(self.output_path)
        output_layout.addWidget(output_btn)
        
        dir_layout.addLayout(input_layout)
        dir_layout.addLayout(output_layout)
        dir_group.setLayout(dir_layout)
        
        # Split selection
        split_group = QGroupBox("Dataset Splits")
        split_layout = QHBoxLayout()
        self.process_train = QCheckBox('Training Set')
        self.process_valid = QCheckBox('Validation Set')
        self.process_train.setChecked(True)
        self.process_valid.setChecked(True)
        split_layout.addWidget(self.process_train)
        split_layout.addWidget(self.process_valid)
        split_group.setLayout(split_layout)
        
        # Dataset information
        info_group = QGroupBox("Dataset Information")
        info_layout = QVBoxLayout()
        self.info_text = QTextEdit()
        self.info_text.setReadOnly(True)
        self.info_text.setMinimumHeight(150)
        self.info_text.setStyleSheet("""
            QTextEdit {
                background-color: #f8f9fa;
                border: 1px solid #dee2e6;
                border-radius: 4px;
                padding: 8px;
                font-family: monospace;
            }
        """)
        info_layout.addWidget(self.info_text)
        info_group.setLayout(info_layout)
        
        # Augmentation options
        aug_group = QGroupBox("Augmentation Options")
        aug_layout = QVBoxLayout()
        
        # Basic transformations
        basic_group = QGroupBox("Basic Transformations")
        basic_layout = QVBoxLayout()
        self.horizontal_flip = QCheckBox('Horizontal Flip')
        self.vertical_flip = QCheckBox('Vertical Flip')
        self.rotate90 = QCheckBox('Rotate 90°')
        self.rotation = QCheckBox('Random Rotation (±30°)')
        basic_layout.addWidget(self.horizontal_flip)
        basic_layout.addWidget(self.vertical_flip)
        basic_layout.addWidget(self.rotate90)
        basic_layout.addWidget(self.rotation)
        basic_group.setLayout(basic_layout)
        
        # Intensity transformations
        intensity_group = QGroupBox("Intensity Transformations")
        intensity_layout = QVBoxLayout()
        self.brightness_contrast = QCheckBox('Brightness/Contrast')
        self.gaussian_noise = QCheckBox('Gaussian Noise')
        self.gaussian_blur = QCheckBox('Gaussian Blur')
        self.clahe = QCheckBox('CLAHE (Contrast Limited AHE)')
        self.random_gamma = QCheckBox('Random Gamma')
        intensity_layout.addWidget(self.brightness_contrast)
        intensity_layout.addWidget(self.gaussian_noise)
        intensity_layout.addWidget(self.gaussian_blur)
        intensity_layout.addWidget(self.clahe)
        intensity_layout.addWidget(self.random_gamma)
        intensity_group.setLayout(intensity_layout)
        
        # Spatial transformations
        spatial_group = QGroupBox("Spatial Transformations")
        spatial_layout = QVBoxLayout()
        self.elastic = QCheckBox('Elastic Transform')
        self.grid_distortion = QCheckBox('Grid Distortion')
        self.cutout = QCheckBox('Cutout (Random Holes)')
        spatial_layout.addWidget(self.elastic)
        spatial_layout.addWidget(self.grid_distortion)
        spatial_layout.addWidget(self.cutout)
        spatial_group.setLayout(spatial_layout)
        
        aug_layout.addWidget(basic_group)
        aug_layout.addWidget(intensity_group)
        aug_layout.addWidget(spatial_group)
        aug_group.setLayout(aug_layout)
        
        # Create a scroll area for augmentation options
        scroll = QScrollArea()
        scroll.setWidget(aug_group)
        scroll.setWidgetResizable(True)
        
        # Preview and Start buttons
        button_layout = QHBoxLayout()
        preview_btn = QPushButton("Preview")
        preview_btn.clicked.connect(self.preview_augmentation)
        self.start_btn = QPushButton("Start Augmentation")  # Store as instance variable
        self.start_btn.clicked.connect(self.start_augmentation)
        button_layout.addWidget(preview_btn)
        button_layout.addWidget(self.start_btn)
        
        # Progress bar and status
        self.progress_bar = QProgressBar()
        self.status_label = QLabel("Ready")
        self.status_label.setAlignment(Qt.AlignCenter)
        
        # Add all components to main layout
        layout.addWidget(dir_group)
        layout.addWidget(split_group)
        layout.addWidget(info_group)
        layout.addWidget(scroll)
        layout.addLayout(button_layout)
        layout.addWidget(self.progress_bar)
        layout.addWidget(self.status_label)
        
        # Store paths
        self.paths = {'input': '', 'output': ''}
        
        # Connect checkbox signals
        for checkbox in [self.horizontal_flip, self.vertical_flip, self.rotate90,
                        self.rotation, self.brightness_contrast, self.gaussian_noise,
                        self.gaussian_blur, self.elastic, self.grid_distortion,
                        self.clahe, self.random_gamma, self.cutout]:
            checkbox.stateChanged.connect(self.update_augmentation_info)
        
        self.process_train.stateChanged.connect(self.update_augmentation_info)
        self.process_valid.stateChanged.connect(self.update_augmentation_info)
        
        self.setGeometry(100, 100, 800, 600)
        self.show()
        
    def select_directory(self, dir_type):
        directory = QFileDialog.getExistingDirectory(self, f'Select {dir_type} Directory')
        if directory:
            self.paths[dir_type] = directory
            if dir_type == 'input':
                self.input_path.setText(directory)
                self.update_data_info()
            else:
                self.output_path.setText(directory)
    
    def update_data_info(self):
        if not self.paths['input']:
            return
            
        info_text = "Dataset Information:\n\n"
        
        total_images = {'train': 0, 'valid': 0}
        total_masks = {'train': 0, 'valid': 0}
        image_sizes = set()
        
        # Collect information
        for data_type in ['images', 'masks']:
            for split in ['train', 'valid']:
                dir_path = os.path.join(self.paths['input'], data_type, split)
                if os.path.exists(dir_path):
                    files = [f for f in os.listdir(dir_path) 
                            if f.endswith(('.png', '.jpg', '.jpeg'))]
                    
                    if data_type == 'images':
                        total_images[split] = len(files)
                        # Get image sizes from first image
                        if files:
                            img_path = os.path.join(dir_path, files[0])
                            img = cv2.imread(img_path)
                            if img is not None:
                                image_sizes.add(f"{img.shape[1]}x{img.shape[0]}")
                    else:
                        total_masks[split] = len(files)
        
        # Format information
        if self.process_train.isChecked():
            info_text += f"Training Set:\n"
            info_text += f"  - Original Images: {total_images['train']}\n"
            info_text += f"  - Original Masks: {total_masks['train']}\n\n"
        
        if self.process_valid.isChecked():
            info_text += f"Validation Set:\n"
            info_text += f"  - Original Images: {total_images['valid']}\n"
            info_text += f"  - Original Masks: {total_masks['valid']}\n\n"
        
        info_text += f"Image Sizes Found: {', '.join(image_sizes)}\n\n"
        
        # Verify data integrity
        if self.process_train.isChecked() and total_images['train'] != total_masks['train']:
            info_text += "⚠️ WARNING: Number of training images and masks don't match!\n"
        if self.process_valid.isChecked() and total_images['valid'] != total_masks['valid']:
            info_text += "⚠️ WARNING: Number of validation images and masks don't match!\n\n"
        
        self.info_text.setText(info_text)
    
    def update_augmentation_info(self):
        """Update the information display with dataset info and augmentation preview"""
        if not self.paths['input']:
            self.info_text.setText("Please select input directory")
            return
            
        info_text = "Dataset Information:\n\n"
        
        total_images = {'train': 0, 'valid': 0}
        total_masks = {'train': 0, 'valid': 0}
        image_sizes = set()
        
        # Collect information
        for data_type in ['images', 'masks']:
            for split in ['train', 'valid']:
                dir_path = os.path.join(self.paths['input'], data_type, split)
                if os.path.exists(dir_path):
                    files = [f for f in os.listdir(dir_path) 
                            if f.endswith(('.png', '.jpg', '.jpeg'))]
                    
                    if data_type == 'images':
                        total_images[split] = len(files)
                        # Get image sizes from first image
                        if files:
                            img_path = os.path.join(dir_path, files[0])
                            img = cv2.imread(img_path)
                            if img is not None:
                                image_sizes.add(f"{img.shape[1]}x{img.shape[0]}")
                    else:
                        total_masks[split] = len(files)
        
        # Format information
        if self.process_train.isChecked():
            info_text += f"Training Set:\n"
            info_text += f"  - Original Images: {total_images['train']}\n"
            info_text += f"  - Original Masks: {total_masks['train']}\n\n"
        
        if self.process_valid.isChecked():
            info_text += f"Validation Set:\n"
            info_text += f"  - Original Images: {total_images['valid']}\n"
            info_text += f"  - Original Masks: {total_masks['valid']}\n\n"
        
        info_text += f"Image Sizes Found: {', '.join(image_sizes)}\n\n"
        
        # Verify data integrity
        if self.process_train.isChecked() and total_images['train'] != total_masks['train']:
            info_text += "⚠️ WARNING: Number of training images and masks don't match!\n"
        if self.process_valid.isChecked() and total_images['valid'] != total_masks['valid']:
            info_text += "⚠️ WARNING: Number of validation images and masks don't match!\n\n"
        
        # Count selected augmentations
        selected_augs = sum([
            self.horizontal_flip.isChecked(),
            self.vertical_flip.isChecked(),
            self.rotate90.isChecked(),
            self.rotation.isChecked(),
            self.brightness_contrast.isChecked(),
            self.gaussian_noise.isChecked(),
            self.gaussian_blur.isChecked(),
            self.elastic.isChecked(),
            self.grid_distortion.isChecked(),
            self.clahe.isChecked(),
            self.random_gamma.isChecked(),
            self.cutout.isChecked()
        ])
        
        info_text += "Augmentation Summary:\n"
        if selected_augs == 0:
            info_text += "No augmentations selected\n"
        else:
            # Calculate new totals
            multiplier = selected_augs + 1  # +1 for original
            new_train_total = total_images['train'] * multiplier if self.process_train.isChecked() else 0
            new_valid_total = total_images['valid'] * multiplier if self.process_valid.isChecked() else 0
            
            info_text += f"Selected augmentations: {selected_augs}\n\n"
            info_text += f"After augmentation:\n"
            if self.process_train.isChecked():
                info_text += f"  Training set: {total_images['train']} → {new_train_total} images\n"
            if self.process_valid.isChecked():
                info_text += f"  Validation set: {total_images['valid']} → {new_valid_total} images\n"
            info_text += f"  Total new images: {new_train_total + new_valid_total}\n"
        
        self.info_text.setText(info_text)
        
    def start_augmentation(self):
        if not all(self.paths.values()):
            QMessageBox.warning(self, 'Warning', 'Please select all directories first!')
            return
            
        if not any([self.horizontal_flip.isChecked(), self.vertical_flip.isChecked(),
                   self.rotate90.isChecked(), self.rotation.isChecked(),
                   self.brightness_contrast.isChecked(), self.gaussian_noise.isChecked(),
                   self.gaussian_blur.isChecked(), self.elastic.isChecked(),
                   self.grid_distortion.isChecked(), self.clahe.isChecked(),
                   self.random_gamma.isChecked(), self.cutout.isChecked()]):
            QMessageBox.warning(self, 'Warning', 'Please select at least one augmentation option!')
            return
            
        if not any([self.process_train.isChecked(), self.process_valid.isChecked()]):
            QMessageBox.warning(self, 'Warning', 'Please select at least one dataset split!')
            return

        # Disable start button
        self.start_btn.setEnabled(False)
        
        # Reset progress
        self.progress_bar.setValue(0)
        
        # Create worker thread
        self.worker = AugmentationWorker(
            self.paths['input'],
            self.paths['output'],
            {
                'horizontal_flip': self.horizontal_flip.isChecked(),
                'vertical_flip': self.vertical_flip.isChecked(),
                'rotate90': self.rotate90.isChecked(),
                'rotation': self.rotation.isChecked(),
                'brightness_contrast': self.brightness_contrast.isChecked(),
                'gaussian_noise': self.gaussian_noise.isChecked(),
                'gaussian_blur': self.gaussian_blur.isChecked(),
                'elastic': self.elastic.isChecked(),
                'grid_distortion': self.grid_distortion.isChecked(),
                'clahe': self.clahe.isChecked(),
                'random_gamma': self.random_gamma.isChecked(),
                'cutout': self.cutout.isChecked(),
                'process_train': self.process_train.isChecked(),
                'process_valid': self.process_valid.isChecked()
            }
        )
        
        # Connect signals
        self.worker.progress.connect(self.progress_bar.setValue)
        self.worker.status.connect(self.status_label.setText)
        self.worker.finished.connect(self.augmentation_finished)
        
        # Start processing
        self.worker.start()
    
    def augmentation_finished(self):
        self.start_btn.setEnabled(True)
        QMessageBox.information(self, 'Success', 'Augmentation process completed successfully!')
        
    def preview_augmentation(self):
        if not self.paths['input']:
            QMessageBox.warning(self, 'Warning', 'Please select input directory first!')
            return
            
        # Get a random image from the dataset
        splits = []
        if self.process_train.isChecked():
            splits.append('train')
        if self.process_valid.isChecked():
            splits.append('valid')
            
        if not splits:
            QMessageBox.warning(self, 'Warning', 'Please select at least one dataset split!')
            return
            
        # Get transforms
        transforms = []
        if self.horizontal_flip.isChecked():
            transforms.append(A.HorizontalFlip(p=1))
        if self.vertical_flip.isChecked():
            transforms.append(A.VerticalFlip(p=1))
        if self.rotate90.isChecked():
            transforms.append(A.RandomRotate90(p=1))
        if self.rotation.isChecked():
            transforms.append(A.Rotate(limit=30, p=1))
        if self.brightness_contrast.isChecked():
            transforms.append(A.RandomBrightnessContrast(p=1))
        if self.gaussian_noise.isChecked():
            transforms.append(A.GaussNoise(var_limit=(10.0, 50.0), p=1))
        if self.gaussian_blur.isChecked():
            transforms.append(A.GaussianBlur(blur_limit=(3, 7), p=1))
        if self.elastic.isChecked():
            transforms.append(A.ElasticTransform(alpha=120, sigma=120 * 0.05, alpha_affine=120 * 0.03, p=1))
        if self.grid_distortion.isChecked():
            transforms.append(A.GridDistortion(p=1))
        if self.clahe.isChecked():
            transforms.append(A.CLAHE(clip_limit=4.0, tile_grid_size=(8, 8), p=1))
        if self.random_gamma.isChecked():
            transforms.append(A.RandomGamma(gamma_limit=(80, 120), p=1))
        if self.cutout.isChecked():
            transforms.append(A.CoarseDropout(max_holes=8, max_height=32, max_width=32, p=1))
            
        if not transforms:
            QMessageBox.warning(self, 'Warning', 'Please select at least one augmentation option!')
            return
            
        # Get a random image
        split = random.choice(splits)
        images_dir = os.path.join(self.paths['input'], 'images', split)
        masks_dir = os.path.join(self.paths['input'], 'masks', split)
        
        image_files = [f for f in os.listdir(images_dir) 
                      if f.endswith(('.png', '.jpg', '.jpeg'))]
        if not image_files:
            QMessageBox.warning(self, 'Warning', f'No images found in {split} split!')
            return
            
        image_file = random.choice(image_files)
        image_path = os.path.join(images_dir, image_file)
        mask_path = os.path.join(masks_dir, image_file)
        
        # Load and process images
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        
        # Apply transforms
        transform = A.Compose(transforms)
        augmented = transform(image=image, mask=mask)
        aug_image = augmented['image']
        aug_mask = augmented['mask']
        
        # Show preview window
        self.preview_window = PreviewWindow(image, mask, aug_image, aug_mask)
        self.preview_window.show()
        
if __name__ == '__main__':
    app = QApplication(sys.argv)
    gui = AugmentationGUI()
    gui.show()
    sys.exit(app.exec_())
