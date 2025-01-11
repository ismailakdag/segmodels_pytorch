import sys
import os
import json
import re
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                           QHBoxLayout, QPushButton, QLabel, QFileDialog, 
                           QSpinBox, QDoubleSpinBox, QComboBox, QProgressBar, 
                           QMessageBox, QGroupBox, QTextEdit, QScrollArea, QGridLayout)
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QUrl
from PyQt5.QtGui import QFont, QDesktopServices
import torch
from segmentation_models_pytorch import (Unet, UnetPlusPlus, MAnet, Linknet, 
                                       FPN, PSPNet, DeepLabV3, DeepLabV3Plus, PAN)
import subprocess
import glob

class TrainWorker(QThread):
    progress = pyqtSignal(str)  # For text output
    epoch_progress = pyqtSignal(int)  # For current epoch
    batch_progress = pyqtSignal(int)  # For current batch
    gpu_memory = pyqtSignal(float)  # For GPU memory usage
    finished = pyqtSignal()

    def __init__(self, config_path, use_checkpoint=False, max_gpu_memory=10.0):
        super().__init__()
        self.config_path = config_path
        self.process = None
        self.should_stop = False
        self.use_checkpoint = use_checkpoint
        self.max_gpu_memory = max_gpu_memory
        
        # Load config to get total epochs and batch size
        with open(config_path, 'r') as f:
            self.config = json.load(f)
        
        # Calculate total batches per epoch
        train_dir = self.config['train_images_dir']
        self.total_images = len([f for f in os.listdir(train_dir) if f.endswith(('.png', '.jpg', '.jpeg'))])
        self.total_batches = self.total_images // self.config['batch_size']
        if self.total_images % self.config['batch_size'] != 0:
            self.total_batches += 1

    def stop(self):
        self.should_stop = True
        if self.process:
            self.process.terminate()

    def run(self):
        try:
            cmd = ['python', 'train.py']
            if self.use_checkpoint:
                cmd.append('--resume')
                
            self.process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                universal_newlines=True,
                bufsize=1
            )

            current_epoch = 0
            current_batch = 0
            last_progress_line = ""
            
            # Read output in real-time
            for line in self.process.stdout:
                if self.should_stop:
                    self.process.terminate()
                    self.progress.emit("Training stopped by user")
                    break

                line = line.strip()
                
                # Check GPU memory usage
                memory_match = re.search(r'Used GPU Memory: (\d+\.\d+) GB', line)
                if memory_match:
                    gpu_memory = float(memory_match.group(1))
                    self.gpu_memory.emit(gpu_memory)
                    if gpu_memory > self.max_gpu_memory:
                        self.progress.emit(f"\n⚠️WARNING: GPU memory usage ({gpu_memory:.2f} GB) exceeded limit ({self.max_gpu_memory:.2f} GB). Stopping training...")
                        self.stop()
                        break
                
                # Parse the line for progress information
                epoch_match = re.search(r'Epoch (\d+)', line)
                if epoch_match:
                    current_epoch = int(epoch_match.group(1))
                    epoch_progress = int((current_epoch / self.config['num_epochs']) * 100)
                    self.epoch_progress.emit(epoch_progress)
                
                batch_match = re.search(r'Batch (\d+)/\d+', line)
                if batch_match:
                    current_batch = int(batch_match.group(1))
                    batch_progress = int((current_batch / self.total_batches) * 100)
                    self.batch_progress.emit(batch_progress)
                    
                    # Update the last progress line instead of adding new ones
                    if re.search(r'Epoch \d+, Batch \d+/\d+', line):
                        last_progress_line = line
                        self.progress.emit("PROGRESS_UPDATE:" + line)
                    else:
                        self.progress.emit(line)
                else:
                    self.progress.emit(line)

            self.process.wait()
            self.finished.emit()

        except Exception as e:
            self.progress.emit(f'Error: {str(e)}')
            self.finished.emit()

class TrainGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.worker = None
        self.config = self.load_default_config()
        
        # Available architectures and encoders
        self.available_architectures = {
            'Unet': Unet,
            'UnetPlusPlus': UnetPlusPlus,
            'MAnet': MAnet,
            'Linknet': Linknet,
            'FPN': FPN,
            'PSPNet': PSPNet,
            'DeepLabV3': DeepLabV3,
            'DeepLabV3Plus': DeepLabV3Plus,
            'PAN': PAN
        }
        
        self.available_encoders = [
            'resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152',
            'resnext50_32x4d', 'resnext101_32x8d', 'timm-resnext50_32x4d',
            'se_resnet50', 'se_resnet101', 'se_resnet152',
            'densenet121', 'densenet169', 'densenet201',
            'dpn68', 'dpn92', 'dpn98', 'dpn107', 'dpn131',
            'vgg11', 'vgg13', 'vgg16', 'vgg19',
            'efficientnet-b0', 'efficientnet-b1', 'efficientnet-b2', 'efficientnet-b3',
            'efficientnet-b4', 'efficientnet-b5', 'efficientnet-b6', 'efficientnet-b7'
        ]
        
        # Setup custom fonts
        self.setup_fonts()
        self.initUI()
        self.check_for_checkpoint()

    def setup_fonts(self):
        # Add custom fonts
        self.title_font = QFont("Segoe UI", 10, QFont.Bold)
        self.text_font = QFont("Consolas", 10)
        self.output_font = QFont("Cascadia Code", 10)  # Modern monospace font
        
    def check_for_checkpoint(self):
        # Get the checkpoint directory path based on current config
        checkpoint_dir = os.path.join(
            'checkpoints',
            f"{self.config['model']['architecture']}_{self.config['model']['encoder_name']}_{self.config['model']['encoder_weights']}",
            f"{self.config['image_size']['width']}x{self.config['image_size']['height']}_b{self.config['batch_size']}"
        )
        checkpoint_path = os.path.join(checkpoint_dir, 'latest_checkpoint.pth')
        
        if os.path.exists(checkpoint_path):
            self.continue_btn.setEnabled(True)
            self.continue_btn.setStyleSheet("""
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
            """)
            self.continue_btn.setToolTip(f"Continue from: {checkpoint_path}")
        else:
            self.continue_btn.setEnabled(False)
            self.continue_btn.setStyleSheet("""
                QPushButton {
                    background-color: #bdc3c7;
                    color: #7f8c8d;
                    border-radius: 3px;
                    padding: 5px;
                    min-width: 80px;
                    font-weight: bold;
                }
            """)
            self.continue_btn.setToolTip("No checkpoint found for current configuration")

    def load_default_config(self):
        try:
            with open('config.json', 'r') as f:
                return json.load(f)
        except:
            return {
                "train_images_dir": "",
                "train_masks_dir": "",
                "valid_images_dir": "",
                "valid_masks_dir": "",
                "batch_size": 16,
                "learning_rate": 0.0001,
                "num_epochs": 100,
                "image_size": {
                    "width": 512,
                    "height": 512
                },
                "model": {
                    "architecture": "Unet",
                    "encoder_name": "resnet34",
                    "encoder_weights": "imagenet",
                    "classes": 1
                },
                "use_timestamp": False
            }

    def initUI(self):
        self.setWindowTitle('Training Configuration GUI')
        self.setFont(self.text_font)
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
            QComboBox, QSpinBox, QDoubleSpinBox {
                padding: 5px;
                border: 1px solid #bdc3c7;
                border-radius: 3px;
                background: white;
            }
            QLabel {
                color: #2c3e50;
            }
            QTextEdit {
                background-color: #1e1e1e;
                color: #d4d4d4;
                border: 1px solid #dee2e6;
                border-radius: 4px;
                padding: 8px;
            }
        """)

        # Create central widget and scroll area
        central_widget = QWidget()
        scroll = QScrollArea()
        scroll.setWidget(central_widget)
        scroll.setWidgetResizable(True)
        self.setCentralWidget(scroll)
        
        main_layout = QVBoxLayout(central_widget)

        # Directory Selection
        dir_group = QGroupBox("Directory Selection")
        dir_layout = QVBoxLayout()

        # Training Images
        train_img_layout = QHBoxLayout()
        self.train_img_path = QLabel(self.config["train_images_dir"])
        self.setFont(self.text_font)
        train_img_btn = QPushButton("Browse")
        train_img_btn.clicked.connect(lambda: self.select_directory('train_images_dir'))
        train_img_layout.addWidget(QLabel("Training Images:"))
        train_img_layout.addWidget(self.train_img_path)
        train_img_layout.addWidget(train_img_btn)

        # Training Masks
        train_mask_layout = QHBoxLayout()
        self.train_mask_path = QLabel(self.config["train_masks_dir"])
        train_mask_btn = QPushButton("Browse")
        train_mask_btn.clicked.connect(lambda: self.select_directory('train_masks_dir'))
        train_mask_layout.addWidget(QLabel("Training Masks:"))
        train_mask_layout.addWidget(self.train_mask_path)
        train_mask_layout.addWidget(train_mask_btn)

        # Validation Images
        valid_img_layout = QHBoxLayout()
        self.valid_img_path = QLabel(self.config["valid_images_dir"])
        valid_img_btn = QPushButton("Browse")
        valid_img_btn.clicked.connect(lambda: self.select_directory('valid_images_dir'))
        valid_img_layout.addWidget(QLabel("Validation Images:"))
        valid_img_layout.addWidget(self.valid_img_path)
        valid_img_layout.addWidget(valid_img_btn)

        # Validation Masks
        valid_mask_layout = QHBoxLayout()
        self.valid_mask_path = QLabel(self.config["valid_masks_dir"])
        valid_mask_btn = QPushButton("Browse")
        valid_mask_btn.clicked.connect(lambda: self.select_directory('valid_masks_dir'))
        valid_mask_layout.addWidget(QLabel("Validation Masks:"))
        valid_mask_layout.addWidget(self.valid_mask_path)
        valid_mask_layout.addWidget(valid_mask_btn)

        dir_layout.addLayout(train_img_layout)
        dir_layout.addLayout(train_mask_layout)
        dir_layout.addLayout(valid_img_layout)
        dir_layout.addLayout(valid_mask_layout)
        dir_group.setLayout(dir_layout)

        # Training Parameters
        params_group = QGroupBox("Training Parameters")
        params_layout = QVBoxLayout()

        # Image Size
        size_layout = QHBoxLayout()
        self.width_input = QSpinBox()
        self.width_input.setRange(1, 2048)
        self.width_input.setValue(self.config["image_size"]["width"])
        self.height_input = QSpinBox()
        self.height_input.setRange(1, 2048)
        self.height_input.setValue(self.config["image_size"]["height"])
        
        size_layout.addWidget(QLabel("Width:"))
        size_layout.addWidget(self.width_input)
        size_layout.addSpacing(20)
        size_layout.addWidget(QLabel("Height:"))
        size_layout.addWidget(self.height_input)
        size_layout.addStretch()

        # Batch Size
        batch_layout = QHBoxLayout()
        self.batch_size = QSpinBox()
        self.batch_size.setRange(1, 128)
        self.batch_size.setValue(self.config["batch_size"])
        batch_layout.addWidget(QLabel("Batch Size:"))
        batch_layout.addWidget(self.batch_size)
        batch_layout.addStretch()

        # Learning Rate
        lr_layout = QHBoxLayout()
        self.learning_rate = QDoubleSpinBox()
        self.learning_rate.setRange(0.000001, 1.0)
        self.learning_rate.setDecimals(6)
        self.learning_rate.setValue(self.config["learning_rate"])
        lr_layout.addWidget(QLabel("Learning Rate:"))
        lr_layout.addWidget(self.learning_rate)
        lr_layout.addStretch()

        # Number of Epochs
        epoch_layout = QHBoxLayout()
        self.num_epochs = QSpinBox()
        self.num_epochs.setRange(1, 1000)
        self.num_epochs.setValue(self.config["num_epochs"])
        epoch_layout.addWidget(QLabel("Number of Epochs:"))
        epoch_layout.addWidget(self.num_epochs)
        epoch_layout.addStretch()

        params_layout.addLayout(size_layout)
        params_layout.addLayout(batch_layout)
        params_layout.addLayout(lr_layout)
        params_layout.addLayout(epoch_layout)
        params_group.setLayout(params_layout)

        # Model Parameters
        model_group = QGroupBox("Model Parameters")
        model_group.setFont(self.title_font)
        model_layout = QGridLayout()

        # Architecture selection
        arch_label = QLabel("Architecture:")
        arch_label.setFont(self.text_font)
        self.arch_combo = QComboBox()
        self.arch_combo.setFont(self.text_font)
        architectures = {
            'Unet': {'name': 'U-Net', 'paper': 'https://arxiv.org/abs/1505.04597', 'docs': 'https://segmentation-models.readthedocs.io/en/latest/models.html#unet'},
            'UnetPlusPlus': {'name': 'U-Net++', 'paper': 'https://arxiv.org/abs/1807.10165', 'docs': 'https://segmentation-models.readthedocs.io/en/latest/models.html#unetplusplus'},
            'MAnet': {'name': 'MA-Net', 'paper': 'https://arxiv.org/abs/1902.05016', 'docs': 'https://segmentation-models.readthedocs.io/en/latest/models.html#manet'},
            'Linknet': {'name': 'LinkNet', 'paper': 'https://arxiv.org/abs/1707.03718', 'docs': 'https://segmentation-models.readthedocs.io/en/latest/models.html#linknet'},
            'FPN': {'name': 'FPN', 'paper': 'https://arxiv.org/abs/1612.03144', 'docs': 'https://segmentation-models.readthedocs.io/en/latest/models.html#fpn'},
            'PSPNet': {'name': 'PSPNet', 'paper': 'https://arxiv.org/abs/1612.01105', 'docs': 'https://segmentation-models.readthedocs.io/en/latest/models.html#pspnet'},
            'DeepLabV3': {'name': 'DeepLabV3', 'paper': 'https://arxiv.org/abs/1706.05587', 'docs': 'https://segmentation-models.readthedocs.io/en/latest/models.html#deeplabv3'},
            'DeepLabV3Plus': {'name': 'DeepLabV3+', 'paper': 'https://arxiv.org/abs/1802.02611', 'docs': 'https://segmentation-models.readthedocs.io/en/latest/models.html#deeplabv3plus'},
            'PAN': {'name': 'PAN', 'paper': 'https://arxiv.org/abs/1805.10180', 'docs': 'https://segmentation-models.readthedocs.io/en/latest/models.html#pan'},
        }
        for arch_key in architectures:
            arch_info = architectures[arch_key]
            self.arch_combo.addItem(arch_info['name'], arch_key)
            
        # Add info buttons for architecture
        arch_paper_btn = QPushButton("📄 Paper")
        arch_paper_btn.setFont(self.text_font)
        arch_docs_btn = QPushButton("📚 Docs")
        arch_docs_btn.setFont(self.text_font)
        
        # Connect buttons to open URLs
        arch_paper_btn.clicked.connect(lambda: self.open_url(architectures[self.arch_combo.currentData()]['paper']))
        arch_docs_btn.clicked.connect(lambda: self.open_url(architectures[self.arch_combo.currentData()]['docs']))
        
        model_layout.addWidget(arch_label, 0, 0)
        model_layout.addWidget(self.arch_combo, 0, 1)
        model_layout.addWidget(arch_paper_btn, 0, 2)
        model_layout.addWidget(arch_docs_btn, 0, 3)
        
        # Encoder Family selection
        encoder_family_label = QLabel("Encoder Family:")
        encoder_family_label.setFont(self.text_font)
        self.encoder_family_combo = QComboBox()
        self.encoder_family_combo.setFont(self.text_font)
        self.encoder_family_combo.addItems(['ResNet', 'ResNeXt', 'DenseNet', 'Inception', 'VGG', 'EfficientNet'])
        
        model_layout.addWidget(encoder_family_label, 1, 0)
        model_layout.addWidget(self.encoder_family_combo, 1, 1, 1, 3)
        
        # Encoder selection
        encoder_label = QLabel("Encoder:")
        encoder_label.setFont(self.text_font)
        self.encoder_combo = QComboBox()
        self.encoder_combo.setFont(self.text_font)
        
        # Encoder info label
        self.encoder_info_label = QLabel()
        self.encoder_info_label.setFont(self.text_font)
        
        model_layout.addWidget(encoder_label, 2, 0)
        model_layout.addWidget(self.encoder_combo, 2, 1, 1, 2)
        model_layout.addWidget(self.encoder_info_label, 2, 3)
        
        # Weights selection
        weights_label = QLabel("Pretrained Weights:")
        weights_label.setFont(self.text_font)
        self.weights_combo = QComboBox()
        self.weights_combo.setFont(self.text_font)
        
        model_layout.addWidget(weights_label, 3, 0)
        model_layout.addWidget(self.weights_combo, 3, 1, 1, 3)
        
        # Connect signals
        self.encoder_family_combo.currentTextChanged.connect(self.update_encoders)
        self.encoder_combo.currentTextChanged.connect(self.update_weights)
        
        model_group.setLayout(model_layout)

        # Progress Bars
        progress_group = QGroupBox("Training Progress")
        progress_layout = QVBoxLayout()
        
        # Epoch progress
        epoch_layout = QHBoxLayout()
        epoch_layout.addWidget(QLabel("Epoch Progress:"))
        self.epoch_progress = QProgressBar()
        epoch_layout.addWidget(self.epoch_progress)
        
        # Batch progress
        batch_layout = QHBoxLayout()
        batch_layout.addWidget(QLabel("Batch Progress:"))
        self.batch_progress = QProgressBar()
        batch_layout.addWidget(self.batch_progress)
        
        progress_layout.addLayout(epoch_layout)
        progress_layout.addLayout(batch_layout)
        progress_group.setLayout(progress_layout)

        # Training Output
        output_group = QGroupBox("Training Output")
        output_layout = QVBoxLayout()
        self.output_text = QTextEdit()
        self.output_text.setFont(self.output_font)
        self.output_text.setReadOnly(True)
        self.output_text.setMinimumHeight(200)
        output_layout.addWidget(self.output_text)
        output_group.setLayout(output_layout)

        # Control Buttons
        button_layout = QHBoxLayout()
        
        # Save Config Button
        self.save_config_btn = QPushButton("Save Config")
        self.save_config_btn.clicked.connect(self.save_config)
        self.save_config_btn.setStyleSheet("""
            QPushButton {
                background-color: #2ecc71;
                color: white;
                border-radius: 3px;
                padding: 5px;
                min-width: 80px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #27ae60;
            }
        """)
        
        # Start Button
        self.start_btn = QPushButton("Start Training")
        self.start_btn.clicked.connect(lambda: self.start_training(False))
        self.start_btn.setStyleSheet("""
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
        """)
        
        # Continue Button
        self.continue_btn = QPushButton("Continue Training")
        self.continue_btn.clicked.connect(lambda: self.start_training(True))
        # Style will be set by check_for_checkpoint()
        
        # Stop Button
        self.stop_btn = QPushButton("Stop Training")
        self.stop_btn.clicked.connect(self.stop_training)
        self.stop_btn.setEnabled(False)
        self.stop_btn.setStyleSheet("""
            QPushButton {
                background-color: #e74c3c;
                color: white;
                border-radius: 3px;
                padding: 5px;
                min-width: 80px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #c0392b;
            }
            QPushButton:disabled {
                background-color: #bdc3c7;
                color: #7f8c8d;
            }
        """)
        
        button_layout.addWidget(self.save_config_btn)
        button_layout.addWidget(self.start_btn)
        button_layout.addWidget(self.continue_btn)
        button_layout.addWidget(self.stop_btn)

        # Add all components to main layout
        main_layout.addWidget(dir_group)
        main_layout.addWidget(params_group)
        main_layout.addWidget(model_group)
        main_layout.addWidget(progress_group)
        main_layout.addWidget(output_group)
        main_layout.addLayout(button_layout)

        self.setGeometry(100, 100, 800, 900)
        self.show()

    def select_directory(self, dir_type):
        directory = QFileDialog.getExistingDirectory(self, f'Select {dir_type.replace("_", " ").title()}')
        if directory:
            self.config[dir_type] = directory
            if dir_type == 'train_images_dir':
                self.train_img_path.setText(directory)
            elif dir_type == 'train_masks_dir':
                self.train_mask_path.setText(directory)
            elif dir_type == 'valid_images_dir':
                self.valid_img_path.setText(directory)
            elif dir_type == 'valid_masks_dir':
                self.valid_mask_path.setText(directory)

    def update_config(self):
        # Store old config values to check if they changed
        old_arch = self.config["model"]["architecture"]
        old_encoder = self.config["model"]["encoder_name"]
        old_weights = self.config["model"]["encoder_weights"]
        old_width = self.config["image_size"]["width"]
        old_height = self.config["image_size"]["height"]
        old_batch = self.config["batch_size"]
        
        # Update config with new values
        self.config["model"]["architecture"] = self.arch_combo.currentData()
        self.config["model"]["encoder_name"] = self.encoder_combo.currentText()
        self.config["model"]["encoder_weights"] = self.weights_combo.currentText()
        self.config["batch_size"] = self.batch_size.value()
        self.config["learning_rate"] = self.learning_rate.value()
        self.config["num_epochs"] = self.num_epochs.value()
        self.config["image_size"]["width"] = self.width_input.value()
        self.config["image_size"]["height"] = self.height_input.value()
        
        # Check if any checkpoint-related config changed
        if (old_arch != self.config["model"]["architecture"] or
            old_encoder != self.config["model"]["encoder_name"] or
            old_weights != self.config["model"]["encoder_weights"] or
            old_width != self.config["image_size"]["width"] or
            old_height != self.config["image_size"]["height"] or
            old_batch != self.config["batch_size"]):
            self.check_for_checkpoint()

    def save_config(self):
        self.update_config()
        try:
            with open('config.json', 'w') as f:
                json.dump(self.config, f, indent=4)
            QMessageBox.information(self, 'Success', 'Configuration saved successfully!')
        except Exception as e:
            QMessageBox.warning(self, 'Error', f'Failed to save configuration: {str(e)}')

    def start_training(self, use_checkpoint=False):
        # First save the configuration
        self.save_config()

        # Reset progress bars
        self.epoch_progress.setValue(0)
        self.batch_progress.setValue(0)

        # Update button states
        self.start_btn.setEnabled(False)
        self.continue_btn.setEnabled(False)
        self.save_config_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        
        self.output_text.clear()

        # Create and start the worker thread
        self.worker = TrainWorker('config.json', use_checkpoint, 10.0)
        self.worker.progress.connect(self.update_output)
        self.worker.epoch_progress.connect(self.epoch_progress.setValue)
        self.worker.batch_progress.connect(self.batch_progress.setValue)
        self.worker.finished.connect(self.training_finished)
        self.worker.start()

    def stop_training(self):
        if self.worker:
            reply = QMessageBox.question(self, 'Confirm Stop', 
                                       'Are you sure you want to stop training? Progress will be saved in the latest checkpoint.',
                                       QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
            if reply == QMessageBox.Yes:
                self.worker.stop()

    def update_output(self, text):
        if text.startswith("PROGRESS_UPDATE:"):
            # Update the last line if it's a progress update
            text = text[15:]  # Remove the PROGRESS_UPDATE: prefix
            current_text = self.output_text.toPlainText()
            lines = current_text.split('\n')
            if lines and re.search(r'Epoch \d+, Batch \d+/\d+', lines[-1]):
                lines[-1] = text
                self.output_text.setText('\n'.join(lines))
            else:
                self.output_text.append(text)
        else:
            self.output_text.append(text)
        
        # Scroll to the bottom
        scrollbar = self.output_text.verticalScrollBar()
        scrollbar.setValue(scrollbar.maximum())

    def training_finished(self):
        self.start_btn.setEnabled(True)
        self.save_config_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.check_for_checkpoint()  # Refresh checkpoint status
        QMessageBox.information(self, 'Complete', 'Training process has finished!')

    def open_url(self, url):
        """Open URL in default browser."""
        QDesktopServices.openUrl(QUrl(url))

    def update_encoders(self, family):
        """Update available encoders when family changes."""
        self.encoder_combo.clear()
        if family == 'ResNet':
            self.encoder_combo.addItems(['resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152'])
        elif family == 'ResNeXt':
            self.encoder_combo.addItems(['resnext50_32x4d', 'resnext101_32x8d', 'timm-resnext50_32x4d'])
        elif family == 'DenseNet':
            self.encoder_combo.addItems(['densenet121', 'densenet169', 'densenet201'])
        elif family == 'Inception':
            self.encoder_combo.addItems(['inceptionv4'])
        elif family == 'VGG':
            self.encoder_combo.addItems(['vgg11', 'vgg13', 'vgg16', 'vgg19'])
        elif family == 'EfficientNet':
            self.encoder_combo.addItems(['efficientnet-b0', 'efficientnet-b1', 'efficientnet-b2', 'efficientnet-b3', 'efficientnet-b4', 'efficientnet-b5', 'efficientnet-b6', 'efficientnet-b7'])
        self.update_weights()

    def update_weights(self):
        """Update available weights when encoder changes."""
        self.weights_combo.clear()
        self.weights_combo.addItems(['imagenet', 'None'])

if __name__ == '__main__':
    app = QApplication(sys.argv)
    gui = TrainGUI()
    sys.exit(app.exec_())
