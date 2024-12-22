import sys
import os
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                           QHBoxLayout, QPushButton, QLabel, QFileDialog, 
                           QSpinBox, QComboBox, QProgressBar, QMessageBox,
                           QGroupBox, QTextEdit)
from PyQt5.QtCore import Qt, QThread, pyqtSignal
import cv2
import shutil

class ResizeWorker(QThread):
    progress = pyqtSignal(int)
    status = pyqtSignal(str)
    finished = pyqtSignal()

    def __init__(self, input_path, output_path, width, height):
        super().__init__()
        self.input_path = input_path
        self.output_path = output_path
        self.width = width
        self.height = height
        self.total_files = 0
        self.processed_files = 0

    def run(self):
        try:
            # Count total files
            for split in ['train', 'valid']:
                for data_type in ['images', 'masks']:
                    dir_path = os.path.join(self.input_path, data_type, split)
                    if os.path.exists(dir_path):
                        self.total_files += len([f for f in os.listdir(dir_path) 
                                               if f.endswith(('.png', '.jpg', '.jpeg'))])

            # Process files
            for split in ['train', 'valid']:
                for data_type in ['images', 'masks']:
                    input_dir = os.path.join(self.input_path, data_type, split)
                    output_dir = os.path.join(self.output_path, data_type, split)
                    
                    if not os.path.exists(input_dir):
                        continue
                        
                    os.makedirs(output_dir, exist_ok=True)
                    
                    files = [f for f in os.listdir(input_dir) 
                            if f.endswith(('.png', '.jpg', '.jpeg'))]
                    
                    interpolation = cv2.INTER_CUBIC if data_type == "images" else cv2.INTER_NEAREST
                    
                    for filename in files:
                        input_path = os.path.join(input_dir, filename)
                        output_path = os.path.join(output_dir, filename)
                        
                        img = cv2.imread(input_path)
                        if img is not None:
                            resized = cv2.resize(img, (self.width, self.height), 
                                               interpolation=interpolation)
                            cv2.imwrite(output_path, resized)
                        
                        self.processed_files += 1
                        progress = int((self.processed_files / self.total_files) * 100)
                        self.progress.emit(progress)
                        self.status.emit(f'Processing {split}/{data_type}: {progress}%')

            self.status.emit('Resizing complete!')
            self.progress.emit(100)
            self.finished.emit()
            
        except Exception as e:
            self.status.emit(f'Error: {str(e)}')
            self.finished.emit()

class ResizeGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.worker = None
        self.initUI()
        
    def initUI(self):
        self.setWindowTitle('Medical Image Resize Tool')
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
            QComboBox, QSpinBox {
                padding: 5px;
                border: 1px solid #bdc3c7;
                border-radius: 3px;
                background: white;
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
        
        # Dataset information
        info_group = QGroupBox("Dataset Information")
        info_layout = QVBoxLayout()
        self.info_text = QTextEdit()
        self.info_text.setReadOnly(True)
        self.info_text.setMinimumHeight(100)
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
        
        # Resize options
        options_group = QGroupBox("Resize Options")
        options_layout = QVBoxLayout()
        
        # Width and Height inputs
        size_layout = QHBoxLayout()
        
        # Width
        width_layout = QHBoxLayout()
        width_layout.addWidget(QLabel("Width:"))
        self.width_input = QSpinBox()
        self.width_input.setRange(1, 2048)
        self.width_input.setValue(512)
        self.width_input.valueChanged.connect(self.update_info)
        width_layout.addWidget(self.width_input)
        size_layout.addLayout(width_layout)
        
        # Add some spacing between width and height
        size_layout.addSpacing(20)
        
        # Height
        height_layout = QHBoxLayout()
        height_layout.addWidget(QLabel("Height:"))
        self.height_input = QSpinBox()
        self.height_input.setRange(1, 2048)
        self.height_input.setValue(512)
        self.height_input.valueChanged.connect(self.update_info)
        height_layout.addWidget(self.height_input)
        size_layout.addLayout(height_layout)
        
        # Add stretch to push controls to the left
        size_layout.addStretch()
        
        options_layout.addLayout(size_layout)
        options_group.setLayout(options_layout)
        
        # Start button
        self.start_btn = QPushButton("Start Resizing")
        self.start_btn.clicked.connect(self.start_resize)
        
        # Progress bar and status
        self.progress_bar = QProgressBar()
        self.status_label = QLabel("Ready")
        self.status_label.setAlignment(Qt.AlignCenter)
        
        # Add all components to main layout
        layout.addWidget(dir_group)
        layout.addWidget(info_group)
        layout.addWidget(options_group)
        layout.addWidget(self.start_btn)
        layout.addWidget(self.progress_bar)
        layout.addWidget(self.status_label)
        
        self.setGeometry(100, 100, 600, 500)
        self.show()
        
        # Store paths
        self.paths = {'input': '', 'output': ''}
        
    def select_directory(self, dir_type):
        directory = QFileDialog.getExistingDirectory(self, f'Select {dir_type} Directory')
        if directory:
            self.paths[dir_type] = directory
            if dir_type == 'input':
                self.input_path.setText(directory)
                self.update_info()
            else:
                self.output_path.setText(directory)
    
    def update_info(self):
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
                        if files:
                            img_path = os.path.join(dir_path, files[0])
                            img = cv2.imread(img_path)
                            if img is not None:
                                image_sizes.add(f"{img.shape[1]}x{img.shape[0]}")
                    else:
                        total_masks[split] = len(files)
        
        # Format information
        info_text += f"Training Set:\n"
        info_text += f"  - Images: {total_images['train']}\n"
        info_text += f"  - Masks: {total_masks['train']}\n\n"
        
        info_text += f"Validation Set:\n"
        info_text += f"  - Images: {total_images['valid']}\n"
        info_text += f"  - Masks: {total_masks['valid']}\n\n"
        
        info_text += f"Current Image Sizes: {', '.join(image_sizes)}\n"
        info_text += f"Target Size: {self.width_input.value()}x{self.height_input.value()}\n"
        
        # Verify data integrity
        if total_images['train'] != total_masks['train']:
            info_text += "\n⚠️ WARNING: Number of training images and masks don't match!\n"
        if total_images['valid'] != total_masks['valid']:
            info_text += "⚠️ WARNING: Number of validation images and masks don't match!\n"
        
        self.info_text.setText(info_text)
    
    def start_resize(self):
        if not all(self.paths.values()):
            QMessageBox.warning(self, 'Warning', 'Please select all directories first!')
            return

        # Disable start button
        self.start_btn.setEnabled(False)
        
        # Reset progress
        self.progress_bar.setValue(0)
        
        # Create worker thread
        self.worker = ResizeWorker(
            self.paths['input'],
            self.paths['output'],
            self.width_input.value(),
            self.height_input.value()
        )
        
        # Connect signals
        self.worker.progress.connect(self.progress_bar.setValue)
        self.worker.status.connect(self.status_label.setText)
        self.worker.finished.connect(self.resize_finished)
        
        # Start processing
        self.worker.start()
    
    def resize_finished(self):
        self.start_btn.setEnabled(True)
        QMessageBox.information(self, 'Success', 'Resize process completed successfully!')

if __name__ == '__main__':
    app = QApplication(sys.argv)
    gui = ResizeGUI()
    sys.exit(app.exec_())
