import os
import sys
import numpy as np
from PIL import Image, ImageDraw
import cv2
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QGridLayout,
                           QPushButton, QLabel, QFileDialog, QProgressBar, QTabWidget,
                           QColorDialog, QSpinBox, QComboBox, QCheckBox, QGroupBox, QListWidget,
                           QMessageBox)
from PyQt5.QtCore import Qt, QThread, pyqtSignal
from PyQt5.QtGui import QImage, QPixmap
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.patches import Rectangle
from matplotlib.colors import ListedColormap

class SegmentationProcessor:
    @staticmethod
    def create_binary_mask(points, image_width, image_height):
        # Create an empty black image
        mask = Image.new('L', (image_width, image_height), 0)
        draw = ImageDraw.Draw(mask)
        
        # Convert normalized coordinates to actual coordinates
        actual_points = []
        for i in range(1, len(points), 2):
            x = float(points[i]) * image_width
            y = float(points[i + 1]) * image_height
            actual_points.append((x, y))
        
        # Draw the polygon
        draw.polygon(actual_points, fill=1)
        return np.array(mask)

    @staticmethod
    def get_bounding_box(points, image_width, image_height):
        coords = np.array([(float(points[i]) * image_width, float(points[i + 1]) * image_height) 
                          for i in range(1, len(points), 2)])
        x_min, y_min = coords.min(axis=0)
        x_max, y_max = coords.max(axis=0)
        
        # Normalize coordinates back
        return [x_min/image_width, y_min/image_height, x_max/image_width, y_max/image_height]

class SegmentationToolsGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('Segmentation Tools')
        self.file_extensions = ('.png', '.jpg', '.jpeg', '.bmp')
        
        # Set application style
        self.setStyleSheet("""
            QMainWindow {
                background-color: #f5f6fa;
            }
            QTabWidget::pane {
                border: 2px solid #dcdde1;
                border-radius: 5px;
                background-color: #f5f6fa;
            }
            QTabBar::tab {
                background-color: #dcdde1;
                color: #2f3640;
                padding: 8px 15px;
                margin-right: 2px;
                border-top-left-radius: 4px;
                border-top-right-radius: 4px;
            }
            QTabBar::tab:selected {
                background-color: #4a90e2;
                color: white;
            }
            QTabBar::tab:hover:!selected {
                background-color: #74b9ff;
            }
            QGroupBox {
                background-color: white;
                border: 2px solid #dcdde1;
                border-radius: 5px;
                margin-top: 1ex;
                padding: 10px;
            }
            QGroupBox::title {
                color: #2f3640;
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 3px;
            }
            QPushButton {
                background-color: #4a90e2;
                color: white;
                border: none;
                padding: 8px;
                border-radius: 4px;
                min-width: 80px;
            }
            QPushButton:hover {
                background-color: #357abd;
            }
            QPushButton:disabled {
                background-color: #bdc3c7;
            }
            QLabel {
                color: #2f3640;
            }
            QComboBox {
                border: 1px solid #dcdde1;
                border-radius: 4px;
                padding: 5px;
                background-color: white;
            }
            QComboBox:hover {
                border-color: #4a90e2;
            }
            QListWidget {
                background-color: white;
                border: 1px solid #dcdde1;
                border-radius: 4px;
            }
            QListWidget::item:selected {
                background-color: #4a90e2;
                color: white;
            }
            QProgressBar {
                border: 1px solid #dcdde1;
                border-radius: 4px;
                text-align: center;
            }
            QProgressBar::chunk {
                background-color: #4a90e2;
            }
            QCheckBox {
                color: #2f3640;
            }
            QSpinBox {
                border: 1px solid #dcdde1;
                border-radius: 4px;
                padding: 5px;
            }
        """)
        
        # Create main widget and layout
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        layout = QVBoxLayout(main_widget)
        
        # Create tab widget
        self.tab_widget = QTabWidget()
        
        # Create tabs
        converter_tab = self.create_converter_tab()
        viewer_tab = self.create_viewer_tab()
        resizer_tab = self.create_resizer_tab()
        combined_viewer_tab = self.create_combined_viewer_tab()
        
        # Add icons to tabs
        self.tab_widget.addTab(converter_tab, '🔄 Converter')
        self.tab_widget.addTab(viewer_tab, '👁 Viewer')
        self.tab_widget.addTab(resizer_tab, '📐 Resizer')
        self.tab_widget.addTab(combined_viewer_tab, '📈 Combined View')
        
        layout.addWidget(self.tab_widget)

    def create_converter_tab(self):
        tab = QWidget()
        layout = QVBoxLayout(tab)
        
        # Directory selection group
        dir_group = QGroupBox("📁 Directory Selection")
        dir_layout = QVBoxLayout(dir_group)
        
        # Input directory
        input_layout = QHBoxLayout()
        self.input_txt_btn = QPushButton('📄 Select TXT Directory')
        self.txt_dir_label = QLabel('TXT Directory: Not selected')
        self.txt_dir_label.setWordWrap(True)  # Allow text wrapping
        input_layout.addWidget(self.input_txt_btn)
        input_layout.addWidget(self.txt_dir_label, 1)
        
        # Image directory
        img_layout = QHBoxLayout()
        self.input_img_btn = QPushButton('🖼 Select Image Directory')
        self.img_dir_label = QLabel('Image Directory: Not selected')
        self.img_dir_label.setWordWrap(True)  # Allow text wrapping
        img_layout.addWidget(self.input_img_btn)
        img_layout.addWidget(self.img_dir_label, 1)
        
        # Output directory
        output_layout = QHBoxLayout()
        self.output_dir_btn = QPushButton('💾 Select Output Directory')
        self.output_dir_label = QLabel('Output Directory: Not selected')
        self.output_dir_label.setWordWrap(True)  # Allow text wrapping
        output_layout.addWidget(self.output_dir_btn)
        output_layout.addWidget(self.output_dir_label, 1)
        
        dir_layout.addLayout(input_layout)
        dir_layout.addLayout(img_layout)
        dir_layout.addLayout(output_layout)
        
        # Options group
        options_group = QGroupBox("⚙ Conversion Options")
        options_layout = QVBoxLayout(options_group)
        
        self.create_mask_cb = QCheckBox('Create Binary Masks')
        self.create_bbox_cb = QCheckBox('Create Bounding Boxes')
        
        options_layout.addWidget(self.create_mask_cb)
        options_layout.addWidget(self.create_bbox_cb)
        
        # Progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        
        # Convert button
        self.convert_btn = QPushButton('▶ Start Conversion')
        self.convert_btn.setEnabled(True)  # Initially disabled
        
        # Layout setup
        layout.addWidget(dir_group)
        layout.addWidget(options_group)
        layout.addWidget(self.progress_bar)
        layout.addWidget(self.convert_btn)
        layout.addStretch()
        
        # Connect signals
        self.input_txt_btn.clicked.connect(lambda: self.select_directory('txt'))
        self.input_img_btn.clicked.connect(lambda: self.select_directory('img'))
        self.output_dir_btn.clicked.connect(lambda: self.select_directory('output'))
        self.convert_btn.clicked.connect(self.start_conversion)
        
        return tab

    def create_viewer_tab(self):
        tab = QWidget()
        layout = QHBoxLayout(tab)

        # Left panel for file browser and controls
        left_panel = QWidget()
        left_panel.setStyleSheet("""
            QGroupBox {
                background-color: #f0f0f0;
                border: 2px solid #c0c0c0;
                border-radius: 5px;
                margin-top: 1ex;
            }
            QGroupBox::title {
                color: #404040;
            }
            QPushButton {
                background-color: #4a90e2;
                color: white;
                border: none;
                padding: 5px;
                border-radius: 3px;
            }
            QPushButton:hover {
                background-color: #357abd;
            }
            QLabel {
                color: #404040;
            }
        """)
        left_layout = QVBoxLayout(left_panel)
        
        # Directory selection
        dir_group = QGroupBox("📁 Directory Selection")
        dir_layout = QVBoxLayout(dir_group)
        
        # Images directory
        self.view_img_dir_btn = QPushButton('🖼 Select Images Directory')
        self.img_dir_label = QLabel('Images Directory: Not selected')
        self.img_info_label = QLabel('Total Images: 0')
        
        # Mask directory
        self.view_mask_dir_btn = QPushButton('🖼 Select Mask Directory')
        self.mask_dir_label = QLabel('Mask Directory: Not selected')
        self.mask_info_label = QLabel('Total Masks: 0')
        
        # Segmentation directory
        self.view_seg_dir_btn = QPushButton('📄 Select Segmentation TXT Directory')
        self.seg_dir_label = QLabel('Segmentation Directory: Not selected')
        self.seg_info_label = QLabel('Total Segmentation Files: 0')
        
        # Detection directory
        self.view_det_dir_btn = QPushButton('📄 Select Detection TXT Directory')
        self.det_dir_label = QLabel('Detection Directory: Not selected')
        self.det_info_label = QLabel('Total Detection Files: 0')
        
        self.file_list = QListWidget()
        self.file_list.setSelectionMode(QListWidget.SingleSelection)
        
        dir_layout.addWidget(self.view_img_dir_btn)
        dir_layout.addWidget(self.img_dir_label)
        dir_layout.addWidget(self.img_info_label)
        dir_layout.addWidget(self.view_mask_dir_btn)
        dir_layout.addWidget(self.mask_dir_label)
        dir_layout.addWidget(self.mask_info_label)
        dir_layout.addWidget(self.view_seg_dir_btn)
        dir_layout.addWidget(self.seg_dir_label)
        dir_layout.addWidget(self.seg_info_label)
        dir_layout.addWidget(self.view_det_dir_btn)
        dir_layout.addWidget(self.det_dir_label)
        dir_layout.addWidget(self.det_info_label)
        dir_layout.addWidget(QLabel('Files:'))
        dir_layout.addWidget(self.file_list)
        
        left_layout.addWidget(dir_group)

        # Visualization options
        options_group = QGroupBox("🔍 Display Options")
        options_layout = QVBoxLayout(options_group)
        
        self.view_type_combo = QComboBox()
        self.view_type_combo.addItems([
            'Image + Mask',
            'Image + Segmentation',
            'Image + Detection',
            'Image + Seg + Detection'
        ])
        
        options_layout.addWidget(QLabel('View Type:'))
        options_layout.addWidget(self.view_type_combo)
        
        # Zoom and pan controls
        controls_layout = QHBoxLayout()
        
        zoom_group = QHBoxLayout()
        self.zoom_in_btn = QPushButton('🔍+')
        self.zoom_out_btn = QPushButton('🔍-')
        self.reset_view_btn = QPushButton('↺ Reset')
        zoom_group.addWidget(self.zoom_in_btn)
        zoom_group.addWidget(self.zoom_out_btn)
        zoom_group.addWidget(self.reset_view_btn)
        
        pan_group = QHBoxLayout()
        self.pan_btn = QPushButton('✋ Pan')
        self.pan_btn.setCheckable(True)
        pan_group.addWidget(self.pan_btn)
        
        controls_layout.addLayout(zoom_group)
        controls_layout.addLayout(pan_group)
        options_layout.addLayout(controls_layout)
        
        left_layout.addWidget(options_group)
        left_layout.addStretch()
        
        # Right panel for visualization
        right_panel = QWidget()
        right_panel.setStyleSheet("""
            QGroupBox {
                background-color: #f0f0f0;
                border: 2px solid #c0c0c0;
                border-radius: 5px;
                margin-top: 1ex;
            }
            QGroupBox::title {
                color: #404040;
            }
        """)
        right_layout = QVBoxLayout(right_panel)
        
        # Create matplotlib figures
        self.figure_image = Figure()
        self.canvas_image = FigureCanvas(self.figure_image)
        
        self.figure_mask = Figure()
        self.canvas_mask = FigureCanvas(self.figure_mask)
        
        # Enable mouse wheel zooming and panning
        self.canvas_image.mpl_connect('scroll_event', self.on_scroll)
        self.canvas_mask.mpl_connect('scroll_event', self.on_scroll)
        self.canvas_image.mpl_connect('button_press_event', self.on_mouse_press)
        self.canvas_mask.mpl_connect('button_press_event', self.on_mouse_press)
        self.canvas_image.mpl_connect('button_release_event', self.on_mouse_release)
        self.canvas_mask.mpl_connect('button_release_event', self.on_mouse_release)
        self.canvas_image.mpl_connect('motion_notify_event', self.on_mouse_move)
        self.canvas_mask.mpl_connect('motion_notify_event', self.on_mouse_move)
        
        self._pan_start = None
        
        right_layout.addWidget(self.canvas_image)
        right_layout.addWidget(self.canvas_mask)
        
        # Add panels to main layout
        layout.addWidget(left_panel, 1)
        layout.addWidget(right_panel, 3)

        # Connect signals
        self.view_img_dir_btn.clicked.connect(lambda: self.select_view_directory('images'))
        self.view_mask_dir_btn.clicked.connect(lambda: self.select_view_directory('masks'))
        self.view_seg_dir_btn.clicked.connect(lambda: self.select_view_directory('segmentation'))
        self.view_det_dir_btn.clicked.connect(lambda: self.select_view_directory('detection'))
        self.file_list.currentItemChanged.connect(self.on_file_selected)
        self.view_type_combo.currentTextChanged.connect(self.update_visualization)
        self.zoom_in_btn.clicked.connect(lambda: self.zoom_view(1.2))
        self.zoom_out_btn.clicked.connect(lambda: self.zoom_view(0.8))
        self.reset_view_btn.clicked.connect(self.reset_view)

        return tab

    def create_resizer_tab(self):
        tab = QWidget()
        layout = QVBoxLayout(tab)
        
        # Folder type selection
        folder_type_group = QGroupBox("🗂 Folder Type")
        folder_type_layout = QVBoxLayout(folder_type_group)
        
        self.folder_type_label = QLabel('Select the type of folder to process:')
        self.folder_type_combo = QComboBox()
        self.folder_type_combo.addItems(['Specific Folder', 'Images Folder', 'Masks Folder'])
        
        folder_type_layout.addWidget(self.folder_type_label)
        folder_type_layout.addWidget(self.folder_type_combo)
        
        # Directory selection
        dir_group = QGroupBox("📁 Directory Selection")
        dir_layout = QVBoxLayout(dir_group)
        
        # Input directory
        input_layout = QHBoxLayout()
        self.resize_input_btn = QPushButton('📁 Select Input Directory')
        self.resize_input_label = QLabel('Input Directory: Not selected')
        input_layout.addWidget(self.resize_input_btn)
        input_layout.addWidget(self.resize_input_label, 1)
        
        # Output directory
        output_layout = QHBoxLayout()
        self.resize_output_btn = QPushButton('💾 Select Output Directory')
        self.resize_output_label = QLabel('Output Directory: Not selected')
        output_layout.addWidget(self.resize_output_btn)
        output_layout.addWidget(self.resize_output_label, 1)
        
        dir_layout.addLayout(input_layout)
        dir_layout.addLayout(output_layout)
        
        # Size settings
        size_group = QGroupBox("📏 Size Settings")
        size_layout = QHBoxLayout(size_group)
        
        self.size_label = QLabel('Target Size:')
        self.size_spinbox = QSpinBox()
        self.size_spinbox.setRange(32, 4096)
        self.size_spinbox.setValue(1024)
        self.size_spinbox.setSingleStep(32)
        
        size_layout.addWidget(self.size_label)
        size_layout.addWidget(self.size_spinbox)
        size_layout.addStretch()
        
        # Status
        self.resize_status = QLabel('Ready')
        self.resize_status.setStyleSheet('color: #2ecc71;')  # Green color for status
        
        # Progress bar
        self.resize_progress = QProgressBar()
        self.resize_progress.setVisible(False)
        
        # Resize button
        self.resize_btn = QPushButton('🔄 Start Resizing')
        self.resize_btn.setEnabled(False)
        
        # Layout setup
        layout.addWidget(folder_type_group)
        layout.addWidget(dir_group)
        layout.addWidget(size_group)
        layout.addWidget(self.resize_status)
        layout.addWidget(self.resize_progress)
        layout.addWidget(self.resize_btn)
        layout.addStretch()
        
        # Connect signals
        self.folder_type_combo.currentTextChanged.connect(self.update_folder_type)
        self.resize_input_btn.clicked.connect(lambda: self.select_resize_directory('input'))
        self.resize_output_btn.clicked.connect(lambda: self.select_resize_directory('output'))
        self.resize_btn.clicked.connect(self.start_resizing)
        
        return tab

    def create_combined_viewer_tab(self):
        tab = QWidget()
        layout = QVBoxLayout()
        
        # Directory selection group
        dir_group = QGroupBox("Directory Selection")
        dir_layout = QGridLayout()
        
        # Image directory
        self.combined_img_dir_label = QLabel("Image Directory: Not selected")
        self.combined_img_dir_label.setWordWrap(True)
        select_img_btn = QPushButton("Select Image Directory")
        select_img_btn.clicked.connect(lambda: self.select_view_directory('images', 'combined'))
        
        # Segmentation directory
        self.combined_seg_dir_label = QLabel("Segmentation Directory: Not selected")
        self.combined_seg_dir_label.setWordWrap(True)
        select_seg_btn = QPushButton("Select Segmentation Directory")
        select_seg_btn.clicked.connect(lambda: self.select_view_directory('segmentation', 'combined'))
        
        # Detection directory
        self.combined_det_dir_label = QLabel("Detection Directory: Not selected")
        self.combined_det_dir_label.setWordWrap(True)
        select_det_btn = QPushButton("Select Detection Directory")
        select_det_btn.clicked.connect(lambda: self.select_view_directory('detection', 'combined'))

        # Mask directory
        self.combined_mask_dir_label = QLabel("Mask Directory: Not selected")
        self.combined_mask_dir_label.setWordWrap(True)
        select_mask_btn = QPushButton("Select Mask Directory")
        select_mask_btn.clicked.connect(lambda: self.select_view_directory('masks', 'combined'))
        
        # Add to layout
        dir_layout.addWidget(self.combined_img_dir_label, 0, 0)
        dir_layout.addWidget(select_img_btn, 0, 1)
        dir_layout.addWidget(self.combined_seg_dir_label, 1, 0)
        dir_layout.addWidget(select_seg_btn, 1, 1)
        dir_layout.addWidget(self.combined_det_dir_label, 2, 0)
        dir_layout.addWidget(select_det_btn, 2, 1)
        dir_layout.addWidget(self.combined_mask_dir_label, 3, 0)
        dir_layout.addWidget(select_mask_btn, 3, 1)
        
        dir_group.setLayout(dir_layout)
        
        # Mask visualization options
        mask_options_group = QGroupBox("Mask Visualization Options")
        mask_options_layout = QHBoxLayout()
        
        self.use_custom_colors_cb = QCheckBox("Use Custom Colors")
        self.use_custom_colors_cb.setChecked(True)
        self.use_custom_colors_cb.stateChanged.connect(self.show_combined_view)
        
        mask_options_layout.addWidget(self.use_custom_colors_cb)
        mask_options_group.setLayout(mask_options_layout)
        
        # Image display area
        display_layout = QHBoxLayout()
        
        # Left side - Original image with overlays
        left_display = QWidget()
        left_layout = QVBoxLayout()
        
        self.combined_image_figure = Figure(figsize=(10, 10))
        self.combined_image_canvas = FigureCanvas(self.combined_image_figure)
        self.combined_image_toolbar = NavigationToolbar(self.combined_image_canvas, self)
        
        # Enable mouse wheel zoom
        self.combined_image_canvas.mpl_connect('scroll_event', self.on_mouse_wheel_zoom)
        
        left_layout.addWidget(self.combined_image_toolbar)
        left_layout.addWidget(self.combined_image_canvas)
        left_display.setLayout(left_layout)
        
        # Right side - Mask image
        right_display = QWidget()
        right_layout = QVBoxLayout()
        
        self.combined_mask_figure = Figure(figsize=(10, 10))
        self.combined_mask_canvas = FigureCanvas(self.combined_mask_figure)
        self.combined_mask_toolbar = NavigationToolbar(self.combined_mask_canvas, self)
        
        # Enable mouse wheel zoom
        self.combined_mask_canvas.mpl_connect('scroll_event', self.on_mouse_wheel_zoom)
        
        right_layout.addWidget(self.combined_mask_toolbar)
        right_layout.addWidget(self.combined_mask_canvas)
        right_display.setLayout(right_layout)
        
        display_layout.addWidget(left_display)
        display_layout.addWidget(right_display)
        
        # File list
        self.combined_file_list = QListWidget()
        self.combined_file_list.itemSelectionChanged.connect(self.show_combined_view)
        
        # Add all to main layout
        layout.addWidget(dir_group)
        layout.addWidget(mask_options_group)
        layout.addLayout(display_layout)
        layout.addWidget(self.combined_file_list)
        
        tab.setLayout(layout)
        return tab

    def update_folder_type(self, folder_type):
        if folder_type == 'Images Folder':
            self.resize_input_btn.setText('🖼 Select Images Directory')
            self.file_extensions = ('.png', '.jpg', '.jpeg', '.bmp')
        elif folder_type == 'Masks Folder':
            self.resize_input_btn.setText('🖼 Select Masks Directory')
            self.file_extensions = ('.png',)  # Only PNG for masks
        else:  # Specific Folder
            self.resize_input_btn.setText('📁 Select Input Directory')
            self.file_extensions = ('.png', '.jpg', '.jpeg', '.bmp')

    def select_directory(self, dir_type):
        try:
            print(f"Attempting to select directory for: {dir_type}")
            dialog = QFileDialog()
            dialog.setFileMode(QFileDialog.FileMode.Directory)
            dialog.setOption(QFileDialog.Option.ShowDirsOnly, True)
            
            if dialog.exec():
                selected_dir = dialog.selectedFiles()[0]
                print(f"Selected directory: {selected_dir}")
                
                # Validate that a directory was actually selected
                if not selected_dir or not os.path.isdir(selected_dir):
                    QMessageBox.warning(self, 'Error', 'Please select a valid directory')
                    return
                
                if dir_type == 'txt':
                    self.txt_dir = selected_dir
                    self.txt_dir_label.setText(f'TXT Directory: {selected_dir}')
                    print(f"TXT directory set: {self.txt_dir}")
                    self.txt_dir_label.update()
                    
                elif dir_type == 'img':
                    # Verify that the directory contains valid image files
                    image_files = []
                    for file in os.listdir(selected_dir):
                        if any(file.lower().endswith(ext) for ext in self.file_extensions):
                            image_files.append(file)
                    
                    print(f"Found image files: {image_files}")
                    if not image_files:
                        QMessageBox.warning(self, 'Error', f'No valid image files found in selected directory.\nSupported formats: {", ".join(self.file_extensions)}')
                        return
                    
                    self.img_dir = selected_dir
                    self.img_dir_label.setText(f'Image Directory: {selected_dir}')
                    print(f"Image directory set: {self.img_dir}")
                    self.img_dir_label.update()
                    
                elif dir_type == 'output':
                    # Ensure the output directory is writable
                    if not os.access(selected_dir, os.W_OK):
                        QMessageBox.warning(self, 'Error', 'Selected directory is not writable')
                        return
                    
                    self.output_dir = selected_dir
                    self.output_dir_label.setText(f'Output Directory: {selected_dir}')
                    print(f"Output directory set: {self.output_dir}")
                    self.output_dir_label.update()
                
                # Force update of the entire directory group
                self.update()
                QApplication.processEvents()
                
                # Enable convert button if all directories are selected
                if all(hasattr(self, attr) for attr in ['txt_dir', 'img_dir', 'output_dir']):
                    self.convert_btn.setEnabled(True)
                        
        except Exception as e:
            QMessageBox.critical(self, 'Error', f'Error selecting directory: {str(e)}')
            print(f"Error selecting directory: {str(e)}")

    def select_view_directory(self, dir_type, view_type='normal'):
        try:
            dialog = QFileDialog()
            dialog.setFileMode(QFileDialog.FileMode.Directory)
            dialog.setOption(QFileDialog.Option.ShowDirsOnly, True)
            
            if dialog.exec():
                dir_path = dialog.selectedFiles()[0]
                
                if view_type == 'combined':
                    if dir_type == 'images':
                        self.combined_img_dir = dir_path
                        self.combined_img_dir_label.setText(f'Image Directory: {dir_path}')
                        # Update file list
                        self.combined_file_list.clear()
                        files = [f for f in os.listdir(dir_path) if any(f.lower().endswith(ext) for ext in self.file_extensions)]
                        self.combined_file_list.addItems(sorted(files))
                    elif dir_type == 'segmentation':
                        self.combined_seg_dir = dir_path
                        self.combined_seg_dir_label.setText(f'Segmentation Directory: {dir_path}')
                    elif dir_type == 'detection':
                        self.combined_det_dir = dir_path
                        self.combined_det_dir_label.setText(f'Detection Directory: {dir_path}')
                    elif dir_type == 'masks':
                        self.combined_mask_dir = dir_path
                        self.combined_mask_dir_label.setText(f'Mask Directory: {dir_path}')
                else:
                    # Original viewer logic...
                    if dir_type == 'images':
                        # Verify that the directory contains valid image files
                        image_files = []
                        for file in os.listdir(dir_path):
                            if any(file.lower().endswith(ext) for ext in self.file_extensions):
                                image_files.append(file)
                        
                        print(f"Found image files: {image_files}")
                        if not image_files:
                            QMessageBox.warning(self, 'Error', f'No valid image files found in selected directory.\nSupported formats: {", ".join(self.file_extensions)}')
                            return
                        
                        self.img_dir = dir_path
                        self.img_dir_label.setText(f'Images Directory: {dir_path}')
                        self.update_file_list()
                        
                    elif dir_type == 'masks':
                        # Verify that the directory contains PNG files for masks
                        mask_files = []
                        for file in os.listdir(dir_path):
                            if file.lower().endswith('.png'):
                                mask_files.append(file)
                        
                        print(f"Found mask files: {mask_files}")
                        if not mask_files:
                            QMessageBox.warning(self, 'Error', 'No PNG mask files found in selected directory')
                            return
                        
                        self.mask_dir = dir_path
                        self.mask_dir_label.setText(f'Mask Directory: {dir_path}')
                        self.update_directory_info('mask')
                        
                    elif dir_type == 'segmentation':
                        # Verify that the directory contains TXT files
                        seg_files = []
                        for file in os.listdir(dir_path):
                            if file.lower().endswith('.txt'):
                                seg_files.append(file)
                        
                        print(f"Found segmentation files: {seg_files}")
                        if not seg_files:
                            QMessageBox.warning(self, 'Error', 'No TXT segmentation files found in selected directory')
                            return
                        
                        self.seg_dir = dir_path
                        self.seg_dir_label.setText(f'Segmentation Directory: {dir_path}')
                        self.update_directory_info('segmentation')
                        
                    elif dir_type == 'detection':
                        # Verify that the directory contains TXT files
                        det_files = []
                        for file in os.listdir(dir_path):
                            if file.lower().endswith('.txt'):
                                det_files.append(file)
                        
                        print(f"Found detection files: {det_files}")
                        if not det_files:
                            QMessageBox.warning(self, 'Error', 'No TXT detection files found in selected directory')
                            return
                        
                        self.det_dir = dir_path
                        self.det_dir_label.setText(f'Detection Directory: {dir_path}')
                        self.update_directory_info('detection')
                    
        except Exception as e:
            QMessageBox.critical(self, 'Error', f'Error selecting directory: {str(e)}')

    def update_directory_info(self, dir_type):
        try:
            if dir_type == 'mask' and hasattr(self, 'mask_dir'):
                mask_files = []
                for file in os.listdir(self.mask_dir):
                    if file.lower().endswith('.png'):
                        mask_files.append(file)
                self.mask_info_label.setText(f'Total Masks: {len(mask_files)}')
                
            elif dir_type == 'segmentation' and hasattr(self, 'seg_dir'):
                seg_files = []
                for file in os.listdir(self.seg_dir):
                    if file.lower().endswith('.txt'):
                        seg_files.append(file)
                self.seg_info_label.setText(f'Total Segmentation Files: {len(seg_files)}')
                
            elif dir_type == 'detection' and hasattr(self, 'det_dir'):
                det_files = []
                for file in os.listdir(self.det_dir):
                    if file.lower().endswith('.txt'):
                        det_files.append(file)
                self.det_info_label.setText(f'Total Detection Files: {len(det_files)}')
        except Exception as e:
            print(f"Error updating directory info: {str(e)}")

    def update_file_list(self):
        self.file_list.clear()
        if hasattr(self, 'img_dir'):
            try:
                files = []
                for file in os.listdir(self.img_dir):
                    if any(file.lower().endswith(ext) for ext in self.file_extensions):
                        files.append(file)
                files.sort()  # Sort files alphabetically
                self.file_list.addItems(files)
                self.img_info_label.setText(f'Total Images: {len(files)}')
                
                # Update all directory information
                self.update_directory_info('mask')
                self.update_directory_info('segmentation')
                self.update_directory_info('detection')
                
            except Exception as e:
                print(f"Error updating file list: {str(e)}")

    def on_file_selected(self, current, previous):
        if current is None:
            return
        
        filename = current.text()
        base_name = os.path.splitext(filename)[0]
        
        # Update current image
        self.current_image = os.path.join(self.img_dir, filename)
        
        # Find corresponding mask/label file
        self.current_label = None
        if hasattr(self, 'mask_dir'):
            if self.view_type_combo.currentText() == 'Image + Mask':
                mask_path = os.path.join(self.mask_dir, base_name + '.png')
                if os.path.exists(mask_path):
                    self.current_label = mask_path
            else:  # Segmentation Points
                label_path = os.path.join(self.mask_dir, base_name + '.txt')
                if os.path.exists(label_path):
                    self.current_label = label_path
        
        self.update_visualization()

    def update_visualization(self):
        if not hasattr(self, 'current_image') or not self.current_image:
            return

        # Clear previous plots
        self.figure_image.clear()
        self.figure_mask.clear()

        # Display main image
        img = cv2.imread(self.current_image)
        if img is None:
            return
            
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        height, width = img.shape[:2]
        
        # Display original image
        ax_img = self.figure_image.add_subplot(111)
        ax_img.imshow(img)
        ax_img.axis('off')
        
        view_type = self.view_type_combo.currentText()
        
        if view_type == 'Image + Mask':
            if not hasattr(self, 'mask_dir'):
                QMessageBox.warning(self, 'Warning', 'Please select mask directory first')
                return
                
            # Show binary mask
            mask_path = os.path.join(self.mask_dir, os.path.splitext(os.path.basename(self.current_image))[0] + '.png')
            if os.path.exists(mask_path):
                mask = cv2.imread(mask_path)
                if mask is not None:
                    # Create colored visualization
                    colored_mask = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)
                    colored_mask[mask[:,:,0] == 0] = [128, 0, 128]  # Purple for background
                    colored_mask[mask[:,:,0] == 1] = [255, 255, 0]  # Yellow for segmentation
                    
                    ax_mask = self.figure_mask.add_subplot(111)
                    ax_mask.imshow(colored_mask)
                    ax_mask.axis('off')
                    self.canvas_mask.setVisible(True)
                    self.canvas_mask.draw()
        
        elif view_type == 'Image + Segmentation':
            if not hasattr(self, 'seg_dir'):
                QMessageBox.warning(self, 'Warning', 'Please select segmentation directory first')
                return
                
            # Draw segmentation points
            txt_path = os.path.join(self.seg_dir, os.path.splitext(os.path.basename(self.current_image))[0] + '.txt')
            if os.path.exists(txt_path):
                with open(txt_path, 'r') as f:
                    for line in f:
                        values = line.strip().split()
                        if len(values) > 4:  # Need at least class and 2 points
                            points = [(float(values[i]) * width, float(values[i+1]) * height) 
                                    for i in range(1, len(values), 2)]
                            points = np.array(points, dtype=np.int32)
                            # Draw thinner line in yellow color
                            ax_img.plot(points[:, 0], points[:, 1], color='yellow', linewidth=1, alpha=0.8)
            
            self.canvas_mask.setVisible(False)
        
        elif view_type == 'Image + Detection':
            if not hasattr(self, 'det_dir'):
                QMessageBox.warning(self, 'Warning', 'Please select detection directory first')
                return
                
            # Show only bounding boxes
            txt_path = os.path.join(self.det_dir, os.path.splitext(os.path.basename(self.current_image))[0] + '.txt')
            if os.path.exists(txt_path):
                with open(txt_path, 'r') as f:
                    for line in f:
                        values = line.strip().split()
                        if len(values) > 4:  # class + at least 2 points
                            class_id = values[0]
                            x_center = float(values[1]) * width
                            y_center = float(values[2]) * height
                            w = float(values[3]) * width
                            h = float(values[4]) * height
                            x1 = int(x_center - w/2)
                            y1 = int(y_center - h/2)
                            x2 = int(x_center + w/2)
                            y2 = int(y_center + h/2)
                            rect = Rectangle((x1, y1), x2-x1, y2-y1,
                                              fill=False, color='g', linewidth=1, alpha=0.8)
                            ax_img.add_patch(rect)
            
            self.canvas_mask.setVisible(False)
        
        elif view_type == 'Image + Seg + Detection':
            if not hasattr(self, 'seg_dir'):
                QMessageBox.warning(self, 'Warning', 'Please select segmentation directory first')
                return
                
            # Draw both segmentation and detection from segmentation file
            txt_path = os.path.join(self.seg_dir, os.path.splitext(os.path.basename(self.current_image))[0] + '.txt')
            if os.path.exists(txt_path):
                with open(txt_path, 'r') as f:
                    for line in f:
                        values = line.strip().split()
                        if len(values) > 4:  # Need at least class and 2 points
                            points = [(float(values[i]) * width, float(values[i+1]) * height) 
                                    for i in range(1, len(values), 2)]
                            points = np.array(points, dtype=np.int32)
                            
                            # Draw thinner segmentation line in yellow
                            ax_img.plot(points[:, 0], points[:, 1], color='yellow', linewidth=1, alpha=0.8)
                            
                            # Draw thinner detection box in green
                            x_min, y_min = points.min(axis=0)
                            x_max, y_max = points.max(axis=0)
                            rect = Rectangle((x_min, y_min), x_max-x_min, y_max-y_min,
                                              fill=False, color='g', linewidth=1, alpha=0.8)
                            ax_img.add_patch(rect)
            
            self.canvas_mask.setVisible(False)
        
        self.canvas_image.draw()

    def load_detection_file(self, det_path):
        try:
            with open(det_path, 'r') as f:
                lines = f.readlines()
            
            # Parse YOLO format: class x_center y_center width height
            boxes = []
            for line in lines:
                values = line.strip().split()
                if len(values) == 5:  # class + 4 box coordinates
                    class_id = values[0]
                    x_center = float(values[1])
                    y_center = float(values[2])
                    width = float(values[3])
                    height = float(values[4])
                    boxes.append([class_id, x_center, y_center, width, height])
            
            return boxes
            
        except Exception as e:
            print(f"Error loading detection file: {str(e)}")
            return []

    def load_segmentation_file(self, seg_path):
        try:
            with open(seg_path, 'r') as f:
                lines = f.readlines()
            
            # Parse segmentation points
            all_points = []
            for line in lines:
                values = line.strip().split()
                if len(values) > 4:  # class + at least 2 points
                    # Skip class id and get points
                    points = []
                    for i in range(1, len(values), 2):
                        x = float(values[i])
                        y = float(values[i+1])
                        # Ensure coordinates are within bounds
                        x = max(0, min(1, x))
                        y = max(0, min(1, y))
                        points.append((x, y))
                    all_points.append(points)
            
            return all_points
            
        except Exception as e:
            print(f"Error loading segmentation file: {str(e)}")
            return []

    def update_image_display(self, image_path, seg_path=None, det_path=None, mask_path=None):
        try:
            # Load and display the image
            self.current_image = cv2.imread(image_path)
            if self.current_image is None:
                return
            
            # Convert BGR to RGB
            self.current_image = cv2.cvtColor(self.current_image, cv2.COLOR_BGR2RGB)
            
            # Load segmentation points if available
            if seg_path and os.path.exists(seg_path):
                self.current_seg_points = self.load_segmentation_file(seg_path)
            else:
                self.current_seg_points = None
            
            # Load detection boxes if available
            if det_path and os.path.exists(det_path):
                self.current_det_boxes = self.load_detection_file(det_path)
            else:
                self.current_det_boxes = None
            
            # Load mask if available
            if mask_path and os.path.exists(mask_path):
                self.current_mask = cv2.imread(mask_path)
                if self.current_mask is not None:
                    self.current_mask = cv2.cvtColor(self.current_mask, cv2.COLOR_BGR2RGB)
            else:
                self.current_mask = None
            
            # Update the display
            self.update_display()
            
        except Exception as e:
            print(f"Error updating image display: {str(e)}")

    def update_display(self):
        try:
            if not hasattr(self, 'current_image') or self.current_image is None:
                return
            
            # Clear the current axes
            self.ax_img.clear()
            
            # Display the image
            self.ax_img.imshow(self.current_image, cmap='gray')
            
            # Get image dimensions
            height, width = self.current_image.shape[:2]
            
            # If segmentation points are available and should be shown
            if hasattr(self, 'current_seg_points') and self.current_seg_points and 'Image + Seg' in self.view_type_combo.currentText():
                for points in self.current_seg_points:
                    # Convert normalized coordinates to pixel coordinates
                    points_px = [(float(x) * width, float(y) * height) 
                               for x, y in points]
                    points_px = np.array(points_px, dtype=np.int32)
                    
                    # Draw segmentation line in yellow
                    self.ax_img.plot(points_px[:, 0], points_px[:, 1], 
                                   color='yellow', linewidth=1, alpha=0.8)
                    
                    # Calculate and draw bounding box if needed
                    if 'Detection' in self.view_type_combo.currentText():
                        x_min, y_min = points_px.min(axis=0)
                        x_max, y_max = points_px.max(axis=0)
                        rect = Rectangle((x_min, y_min), x_max-x_min, y_max-y_min,
                                      fill=False, color='g', linewidth=1, alpha=0.8)
                        self.ax_img.add_patch(rect)
            
            # If detection boxes are available and should be shown
            if hasattr(self, 'current_det_boxes') and self.current_det_boxes and 'Detection' in self.view_type_combo.currentText():
                for box in self.current_det_boxes:
                    # Get normalized coordinates
                    x_center, y_center, box_width, box_height = map(float, box[1:])
                    
                    # Convert to pixel coordinates
                    x_min = int((x_center - box_width/2) * width)
                    y_min = int((y_center - box_height/2) * height)
                    box_width_px = int(box_width * width)
                    box_height_px = int(box_height * height)
                    
                    # Create rectangle patch
                    rect = Rectangle(
                        (x_min, y_min), 
                        box_width_px, 
                        box_height_px,
                        linewidth=1, 
                        edgecolor='g', 
                        facecolor='none', 
                        alpha=0.8
                    )
                    self.ax_img.add_patch(rect)
            
            # If mask is available and should be shown
            if hasattr(self, 'current_mask') and self.current_mask is not None and 'Mask' in self.view_type_combo.currentText():
                mask_overlay = np.zeros((*self.current_mask.shape[:2], 4))
                mask_overlay[self.current_mask > 0] = [1, 1, 0, 0.3]  # Yellow with 0.3 alpha
                self.ax_img.imshow(mask_overlay)
            
            # Remove axes for cleaner look
            self.ax_img.set_xticks([])
            self.ax_img.set_yticks([])
            
            # Update the canvas
            self.canvas.draw()
            
        except Exception as e:
            print(f"Error updating display: {str(e)}")

    def select_color(self, color_type):
        color = QColorDialog.getColor()
        if color.isValid():
            if color_type == 'bg':
                self.bg_color = (color.red(), color.green(), color.blue())
            else:
                self.fg_color = (color.red(), color.green(), color.blue())
            self.update_visualization()

    def select_resize_directory(self, dir_type):
        dir_path = QFileDialog.getExistingDirectory(self, f"Select {dir_type.title()} Directory")
        if dir_path:
            if dir_type == 'input':
                self.resize_input_label.setText(f'Input Directory: {dir_path}')
                self.resize_input_dir = dir_path
            else:
                self.resize_output_label.setText(f'Output Directory: {dir_path}')
                self.resize_output_dir = dir_path
            
            # Enable resize button if both directories are selected
            if hasattr(self, 'resize_input_dir') and hasattr(self, 'resize_output_dir'):
                self.resize_btn.setEnabled(True)

    def display_image(self, image_data, mask_data=None):
        if image_data is None:
            return
            
        self.figure_image.clear()
        ax_img = self.figure_image.add_subplot(111)
        ax_img.imshow(image_data)
        ax_img.axis('off')
        self.canvas_image.draw()
        
        if mask_data is not None:
            self.figure_mask.clear()
            ax_mask = self.figure_mask.add_subplot(111)
            ax_mask.imshow(mask_data)
            ax_mask.axis('off')
            self.canvas_mask.draw()

    def start_conversion(self):
        if not hasattr(self, 'txt_dir') or not hasattr(self, 'img_dir') or not hasattr(self, 'output_dir'):
            QMessageBox.warning(self, 'Warning', 'Please select all required directories first.')
            return

        if not self.create_mask_cb.isChecked() and not self.create_bbox_cb.isChecked():
            QMessageBox.warning(self, 'Warning', 'Please select at least one conversion option.')
            return

        try:
            txt_files = [f for f in os.listdir(self.txt_dir) if f.endswith('.txt')]
            total_files = len(txt_files)
            
            self.progress_bar.setVisible(True)
            self.progress_bar.setValue(0)
            
            for i, txt_file in enumerate(txt_files):
                base_name = os.path.splitext(txt_file)[0]
                txt_path = os.path.join(self.txt_dir, txt_file)
                
                # Find corresponding image
                img_path = None
                for ext in ['.jpg', '.jpeg', '.png', '.bmp']:
                    temp_path = os.path.join(self.img_dir, base_name + ext)
                    if os.path.exists(temp_path):
                        img_path = temp_path
                        break
                
                if img_path is None:
                    continue
                
                # Read image dimensions
                img = cv2.imread(img_path)
                if img is None:
                    continue
                    
                height, width = img.shape[:2]
                
                # Read segmentation points
                with open(txt_path, 'r') as f:
                    lines = f.readlines()
                
                if self.create_mask_cb.isChecked():
                    # Create mask (black background, white segmentation)
                    mask = np.zeros((height, width, 3), dtype=np.uint8)
                    
                    # Process each segmentation separately
                    for line in lines:
                        values = line.strip().split()
                        if len(values) > 4:
                            # Create separate mask for each segmentation
                            seg_mask = np.zeros((height, width), dtype=np.uint8)
                            
                            # Convert normalized coordinates to pixel coordinates
                            points = [(float(values[i]) * width, float(values[i+1]) * height) 
                                     for i in range(1, len(values), 2)]
                            points = np.array(points, dtype=np.int32)
                            
                            # Fill the polygon for this segmentation
                            cv2.fillPoly(seg_mask, [points], 1)
                            
                            # Add to the main mask
                            mask[seg_mask > 0] = [1, 1, 1]
                    
                    mask_path = os.path.join(self.output_dir, base_name + '.png')
                    cv2.imwrite(mask_path, mask)
                
                if self.create_bbox_cb.isChecked():
                    bbox_path = os.path.join(self.output_dir, base_name + '.txt')
                    with open(bbox_path, 'w') as f:
                        for line in lines:
                            values = line.strip().split()
                            if len(values) > 4:
                                class_id = values[0]
                                
                                # Convert to pixel coordinates first
                                points = [(float(values[i]) * width, float(values[i+1]) * height) 
                                         for i in range(1, len(values), 2)]
                                points = np.array(points, dtype=np.int32)
                                
                                # Get min/max in pixel coordinates
                                x_min, y_min = points.min(axis=0)
                                x_max, y_max = points.max(axis=0)
                                
                                # Convert back to normalized coordinates
                                x_center = (x_min + x_max) / (2 * width)
                                y_center = (y_min + y_max) / (2 * height)
                                bbox_width = (x_max - x_min) / width
                                bbox_height = (y_max - y_min) / height
                                
                                # Ensure values are within [0, 1]
                                x_center = max(0, min(1, x_center))
                                y_center = max(0, min(1, y_center))
                                bbox_width = max(0, min(1, bbox_width))
                                bbox_height = max(0, min(1, bbox_height))
                                
                                # Write YOLO format
                                f.write(f"{class_id} {x_center:.6f} {y_center:.6f} {bbox_width:.6f} {bbox_height:.6f}\n")
                
                progress = int((i + 1) / total_files * 100)
                self.progress_bar.setValue(progress)
            
            QMessageBox.information(self, 'Success', 'Conversion completed successfully!')
            
        except Exception as e:
            QMessageBox.critical(self, 'Error', f'An error occurred: {str(e)}')
        
        finally:
            self.progress_bar.setVisible(False)

    def start_resizing(self):
        target_size = self.size_spinbox.value()
        folder_type = self.folder_type_combo.currentText()
        self.resize_worker = ResizeWorker(
            self.resize_input_dir, 
            self.resize_output_dir, 
            target_size,
            folder_type,
            self.file_extensions
        )
        self.resize_worker.progress.connect(self.update_resize_progress)
        self.resize_worker.status.connect(self.update_resize_status)
        self.resize_worker.info.connect(self.update_resize_info)
        self.resize_worker.finished.connect(self.resize_finished)
        
        self.resize_progress.setVisible(True)
        self.resize_progress.setValue(0)
        self.resize_btn.setEnabled(False)
        self.resize_input_btn.setEnabled(False)
        self.resize_output_btn.setEnabled(False)
        
        self.resize_worker.start()

    def update_resize_progress(self, value):
        self.resize_progress.setValue(value)

    def update_resize_status(self, message):
        self.resize_status.setText(message)

    def update_resize_info(self, message):
        self.info_label.setText(message)

    def resize_finished(self):
        self.resize_btn.setEnabled(True)
        self.resize_input_btn.setEnabled(True)
        self.resize_output_btn.setEnabled(True)
        self.resize_status.setText('Resizing completed!')

    def on_scroll(self, event):
        """Handle mouse wheel scrolling for zoom"""
        if event.inaxes:
            # Get the current axis
            ax = event.inaxes
            # Get the current x and y limits
            cur_xlim = ax.get_xlim()
            cur_ylim = ax.get_ylim()
            
            # Get the cursor position
            xdata = event.xdata
            ydata = event.ydata
            
            # Scaling factors
            base_scale = 1.1
            if event.button == 'up':
                scale_factor = 1/base_scale
            else:
                scale_factor = base_scale
            
            # Calculate new limits
            new_width = (cur_xlim[1] - cur_xlim[0]) * scale_factor
            new_height = (cur_ylim[1] - cur_ylim[0]) * scale_factor
            
            # Set new limits while maintaining the cursor position
            rel_x = (cur_xlim[1] - xdata)/(cur_xlim[1] - cur_xlim[0])
            rel_y = (cur_ylim[1] - ydata)/(cur_ylim[1] - cur_ylim[0])
            
            ax.set_xlim([xdata - new_width * (1-rel_x), xdata + new_width * rel_x])
            ax.set_ylim([ydata - new_height * (1-rel_y), ydata + new_height * rel_y])
            
            # Redraw
            if event.inaxes in self.figure_image.axes:
                self.canvas_image.draw()
            else:
                self.canvas_mask.draw()

    def on_mouse_press(self, event):
        if self.pan_btn.isChecked() and event.button == 1:  # Left click
            self._pan_start = (event.xdata, event.ydata)

    def on_mouse_release(self, event):
        self._pan_start = None

    def on_mouse_move(self, event):
        if self._pan_start is not None and event.inaxes:
            dx = event.xdata - self._pan_start[0]
            dy = event.ydata - self._pan_start[1]
            
            ax = event.inaxes
            ax.set_xlim(ax.get_xlim() - dx)
            ax.set_ylim(ax.get_ylim() - dy)
            
            if event.inaxes in self.figure_image.axes:
                self.canvas_image.draw()
            else:
                self.canvas_mask.draw()

    def zoom_view(self, factor):
        """Zoom in or out on both canvases"""
        for fig in [self.figure_image, self.figure_mask]:
            if fig.axes:
                ax = fig.axes[0]
                # Get the current limits
                xlim = ax.get_xlim()
                ylim = ax.get_ylim()
                
                # Calculate center
                xcenter = (xlim[1] + xlim[0]) / 2
                ycenter = (ylim[1] + ylim[0]) / 2
                
                # Calculate new limits
                new_width = (xlim[1] - xlim[0]) / factor
                new_height = (ylim[1] - ylim[0]) / factor
                
                # Set new limits
                ax.set_xlim([xcenter - new_width/2, xcenter + new_width/2])
                ax.set_ylim([ycenter - new_height/2, ycenter + new_height/2])
        
        # Redraw both canvases
        self.canvas_image.draw()
        self.canvas_mask.draw()

    def reset_view(self):
        """Reset the view to original size"""
        for fig in [self.figure_image, self.figure_mask]:
            if fig.axes:
                ax = fig.axes[0]
                ax.set_xlim(0, ax.get_images()[0].get_array().shape[1])
                ax.set_ylim(ax.get_images()[0].get_array().shape[0], 0)
        
        # Redraw both canvases
        self.canvas_image.draw()
        self.canvas_mask.draw()

    def show_combined_view(self):
        try:
            current_item = self.combined_file_list.currentItem()
            if not current_item:
                return
                
            file_path = os.path.join(self.combined_img_dir, current_item.text())
            
            # Clear previous plots
            self.combined_image_figure.clear()
            self.combined_mask_figure.clear()
            
            # Create subplots
            ax_img = self.combined_image_figure.add_subplot(111)
            ax_mask = self.combined_mask_figure.add_subplot(111)
            
            # Load and display the original image
            image = cv2.imread(file_path)
            if image is None:
                raise Exception("Failed to load image")
            
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            height, width = image.shape[:2]
            
            # Display original image
            ax_img.imshow(image)
            ax_img.axis('off')
            
            # Add segmentation lines if available
            if hasattr(self, 'combined_seg_dir'):
                seg_file = os.path.join(self.combined_seg_dir, os.path.splitext(os.path.basename(file_path))[0] + '.txt')
                if os.path.exists(seg_file):
                    with open(seg_file, 'r') as f:
                        for line in f:
                            values = line.strip().split()
                            if len(values) > 4:
                                points = [(float(values[i]) * width, float(values[i+1]) * height) 
                                        for i in range(1, len(values), 2)]
                                points = np.array(points)
                                ax_img.plot(points[:, 0], points[:, 1], 'g-', linewidth=2)
            
            # Add detection boxes if available
            if hasattr(self, 'combined_det_dir'):
                det_file = os.path.join(self.combined_det_dir, os.path.splitext(os.path.basename(file_path))[0] + '.txt')
                if os.path.exists(det_file):
                    with open(det_file, 'r') as f:
                        for line in f:
                            values = line.strip().split()
                            if len(values) == 5:
                                x_center = float(values[1]) * width
                                y_center = float(values[2]) * height
                                w = float(values[3]) * width
                                h = float(values[4]) * height
                                
                                x1 = x_center - w/2
                                y1 = y_center - h/2
                                
                                rect = Rectangle((x1, y1), w, h,
                                              fill=False, color='b', linewidth=2)
                                ax_img.add_patch(rect)
            
            # Show mask if available
            if hasattr(self, 'combined_mask_dir'):
                mask_file = os.path.join(self.combined_mask_dir, os.path.splitext(os.path.basename(file_path))[0] + '.png')
                if os.path.exists(mask_file):
                    mask = cv2.imread(mask_file, cv2.IMREAD_GRAYSCALE)
                    if mask is not None:
                        if self.use_custom_colors_cb.isChecked():
                            # Create custom colormap (purple for 0, yellow for 1)
                            custom_cmap = np.zeros((256, 3))
                            custom_cmap[:128] = [0.5, 0, 0.5]  # Purple for values 0-127
                            custom_cmap[128:] = [1, 1, 0]      # Yellow for values 128-255
                            
                            # Display mask with custom colors
                            ax_mask.imshow(mask, cmap=ListedColormap(custom_cmap))
                        else:
                            # Display mask in grayscale
                            ax_mask.imshow(mask, cmap='gray')
                        
                        ax_mask.axis('off')
            
            # Update the canvases
            self.combined_image_canvas.draw()
            self.combined_mask_canvas.draw()
            
        except Exception as e:
            print(f"Error showing combined view: {str(e)}")
            QMessageBox.critical(self, 'Error', f'Error showing combined view: {str(e)}')

    def on_mouse_wheel_zoom(self, event):
        ax = event.inaxes
        if ax is None:
            return
        
        # Get the current x and y limits
        cur_xlim = ax.get_xlim()
        cur_ylim = ax.get_ylim()
        
        # Get the cursor position
        xdata = event.xdata
        ydata = event.ydata
        if xdata is None or ydata is None:
            return
        
        # Get the zoom factor
        base_scale = 1.1
        if event.button == 'up':
            scale_factor = 1/base_scale
        elif event.button == 'down':
            scale_factor = base_scale
        else:
            return
        
        # Calculate new limits
        new_width = (cur_xlim[1] - cur_xlim[0]) * scale_factor
        new_height = (cur_ylim[1] - cur_ylim[0]) * scale_factor
        
        relx = (cur_xlim[1] - xdata)/(cur_xlim[1] - cur_xlim[0])
        rely = (cur_ylim[1] - ydata)/(cur_ylim[1] - cur_ylim[0])
        
        ax.set_xlim([xdata - new_width * (1-relx), xdata + new_width * relx])
        ax.set_ylim([ydata - new_height * (1-rely), ydata + new_height * rely])
        
        # Redraw the figure
        ax.figure.canvas.draw_idle()

class ResizeWorker(QThread):
    progress = pyqtSignal(int)
    status = pyqtSignal(str)
    info = pyqtSignal(str)
    finished = pyqtSignal()

    def __init__(self, input_dir, output_dir, target_size, folder_type, file_extensions):
        super().__init__()
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.target_size = (target_size, target_size)
        self.folder_type = folder_type
        self.file_extensions = file_extensions

    def run(self):
        try:
            files = []
            for file in os.listdir(self.input_dir):
                if any(file.lower().endswith(ext) for ext in self.file_extensions):
                    files.append(file)

            total_files = len(files)
            for i, file in enumerate(files):
                input_path = os.path.join(self.input_dir, file)
                output_path = os.path.join(self.output_dir, file)

                # Read image
                img = cv2.imread(input_path)
                if img is None:
                    self.info.emit(f"Could not read {file}")
                    continue

                # Resize image
                resized = cv2.resize(img, self.target_size, interpolation=cv2.INTER_AREA)
                
                # Save resized image
                cv2.imwrite(output_path, resized)
                
                progress = int((i + 1) / total_files * 100)
                self.progress.emit(progress)
                self.status.emit(f'Processing: {file}')

            self.status.emit('Finished processing all files')
            self.finished.emit()

        except Exception as e:
            self.status.emit(f'Error: {str(e)}')
            self.finished.emit()

def main():
    app = QApplication(sys.argv)
    gui = SegmentationToolsGUI()
    gui.show()
    sys.exit(app.exec_())

if __name__ == '__main__':
    main()
