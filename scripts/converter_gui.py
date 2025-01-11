import sys
import os
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                           QPushButton, QLabel, QFileDialog, QProgressBar)
from PyQt5.QtCore import Qt, QThread, pyqtSignal
from PIL import Image

class ConversionWorker(QThread):
    progress = pyqtSignal(int)
    finished = pyqtSignal()
    error = pyqtSignal(str)

    def __init__(self, input_dir, output_dir):
        super().__init__()
        self.input_dir = input_dir
        self.output_dir = output_dir

    def run(self):
        try:
            # Get list of all .bmp files
            bmp_files = [f for f in os.listdir(self.input_dir) if f.lower().endswith('.bmp')]
            total_files = len(bmp_files)

            if total_files == 0:
                self.error.emit("No BMP files found in the input directory!")
                return

            for i, bmp_file in enumerate(bmp_files):
                try:
                    input_path = os.path.join(self.input_dir, bmp_file)
                    output_filename = os.path.splitext(bmp_file)[0] + '.png'
                    output_path = os.path.join(self.output_dir, output_filename)

                    # Convert image
                    with Image.open(input_path) as img:
                        img.save(output_path, 'PNG')

                    # Update progress
                    progress = int((i + 1) / total_files * 100)
                    self.progress.emit(progress)

                except Exception as e:
                    self.error.emit(f"Error converting {bmp_file}: {str(e)}")

            self.finished.emit()

        except Exception as e:
            self.error.emit(f"Conversion error: {str(e)}")

class ConverterGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.initUI()

    def initUI(self):
        self.setWindowTitle('Image Converter - BMP to PNG')
        self.setGeometry(100, 100, 600, 200)

        # Create central widget and layout
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        layout = QVBoxLayout(central_widget)

        # Create widgets
        self.input_label = QLabel('Input Directory: Not selected')
        self.output_label = QLabel('Output Directory: Not selected')
        
        self.select_input_btn = QPushButton('Select Input Directory')
        self.select_output_btn = QPushButton('Select Output Directory')
        self.convert_btn = QPushButton('Convert BMP to PNG')
        self.convert_btn.setEnabled(False)
        
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        
        self.status_label = QLabel('')
        self.status_label.setAlignment(Qt.AlignCenter)

        # Add widgets to layout
        layout.addWidget(self.input_label)
        layout.addWidget(self.select_input_btn)
        layout.addWidget(self.output_label)
        layout.addWidget(self.select_output_btn)
        layout.addWidget(self.convert_btn)
        layout.addWidget(self.progress_bar)
        layout.addWidget(self.status_label)

        # Connect signals
        self.select_input_btn.clicked.connect(self.select_input_directory)
        self.select_output_btn.clicked.connect(self.select_output_directory)
        self.convert_btn.clicked.connect(self.start_conversion)

        self.input_dir = ''
        self.output_dir = ''

    def select_input_directory(self):
        dir_path = QFileDialog.getExistingDirectory(self, "Select Input Directory")
        if dir_path:
            self.input_dir = dir_path
            self.input_label.setText(f'Input Directory: {dir_path}')
            self.update_convert_button()

    def select_output_directory(self):
        dir_path = QFileDialog.getExistingDirectory(self, "Select Output Directory")
        if dir_path:
            self.output_dir = dir_path
            self.output_label.setText(f'Output Directory: {dir_path}')
            self.update_convert_button()

    def update_convert_button(self):
        self.convert_btn.setEnabled(bool(self.input_dir and self.output_dir))

    def start_conversion(self):
        self.progress_bar.setVisible(True)
        self.progress_bar.setValue(0)
        self.convert_btn.setEnabled(False)
        self.select_input_btn.setEnabled(False)
        self.select_output_btn.setEnabled(False)
        self.status_label.setText('Converting...')

        # Create and start worker thread
        self.worker = ConversionWorker(self.input_dir, self.output_dir)
        self.worker.progress.connect(self.update_progress)
        self.worker.finished.connect(self.conversion_finished)
        self.worker.error.connect(self.show_error)
        self.worker.start()

    def update_progress(self, value):
        self.progress_bar.setValue(value)

    def conversion_finished(self):
        self.progress_bar.setValue(100)
        self.status_label.setText('Conversion completed successfully!')
        self.enable_controls()

    def show_error(self, error_message):
        self.status_label.setText(error_message)
        self.enable_controls()

    def enable_controls(self):
        self.convert_btn.setEnabled(True)
        self.select_input_btn.setEnabled(True)
        self.select_output_btn.setEnabled(True)

def main():
    app = QApplication(sys.argv)
    gui = ConverterGUI()
    gui.show()
    sys.exit(app.exec_())

if __name__ == '__main__':
    main()
