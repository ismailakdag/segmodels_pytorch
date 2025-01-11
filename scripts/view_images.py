import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import tkinter as tk
from tkinter import ttk, filedialog
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import cv2

class ImageMaskViewer(tk.Tk):
    def __init__(self):
        super().__init__()

        self.title("Image and YOLO Mask Viewer")
        self.geometry("1200x800")
        self.configure(bg='#2C3E50')

        # Configure style
        self.style = ttk.Style()
        self.style.configure('TFrame', background='#2C3E50')
        self.style.configure('TButton', 
                           padding=10, 
                           font=('Arial', 10, 'bold'),
                           background='#3498DB')
        self.style.configure('TLabel', 
                           font=('Arial', 10),
                           background='#2C3E50',
                           foreground='white')
        
        # Create main frame
        self.main_frame = ttk.Frame(self)
        self.main_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)

        # Create control frame
        self.control_frame = ttk.Frame(self.main_frame)
        self.control_frame.pack(fill=tk.X, pady=(0, 20))

        # Create directory selection frame
        self.dir_frame = ttk.Frame(self.control_frame)
        self.dir_frame.pack(fill=tk.X, pady=(0, 10))

        # Image directory selection
        self.img_dir_frame = ttk.Frame(self.dir_frame)
        self.img_dir_frame.pack(fill=tk.X, pady=5)
        ttk.Label(self.img_dir_frame, text="Images Directory:").pack(side=tk.LEFT, padx=5)
        self.img_dir_label = ttk.Label(self.img_dir_frame, text="Not selected", foreground='#E74C3C')
        self.img_dir_label.pack(side=tk.LEFT, padx=5)
        self.select_img_btn = ttk.Button(self.img_dir_frame, 
                                       text="Select Images Directory", 
                                       command=self.select_image_directory)
        self.select_img_btn.pack(side=tk.RIGHT, padx=5)

        # Mask directory selection
        self.mask_dir_frame = ttk.Frame(self.dir_frame)
        self.mask_dir_frame.pack(fill=tk.X, pady=5)
        ttk.Label(self.mask_dir_frame, text="Binary Masks Directory:").pack(side=tk.LEFT, padx=5)
        self.mask_dir_label = ttk.Label(self.mask_dir_frame, text="Not selected", foreground='#E74C3C')
        self.mask_dir_label.pack(side=tk.LEFT, padx=5)
        self.select_mask_btn = ttk.Button(self.mask_dir_frame, 
                                        text="Select Binary Masks Directory", 
                                        command=self.select_mask_directory)
        self.select_mask_btn.pack(side=tk.RIGHT, padx=5)

        # YOLO labels directory selection
        self.yolo_dir_frame = ttk.Frame(self.dir_frame)
        self.yolo_dir_frame.pack(fill=tk.X, pady=5)
        ttk.Label(self.yolo_dir_frame, text="YOLO Labels Directory:").pack(side=tk.LEFT, padx=5)
        self.yolo_dir_label = ttk.Label(self.yolo_dir_frame, text="Not selected", foreground='#E74C3C')
        self.yolo_dir_label.pack(side=tk.LEFT, padx=5)
        self.select_yolo_btn = ttk.Button(self.yolo_dir_frame, 
                                        text="Select YOLO Labels Directory", 
                                        command=self.select_yolo_directory)
        self.select_yolo_btn.pack(side=tk.RIGHT, padx=5)

        # Create list frame with a title
        self.list_frame = ttk.Frame(self.main_frame)
        self.list_frame.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 20))
        
        ttk.Label(self.list_frame, text="Available Images", 
                 font=('Arial', 12, 'bold')).pack(pady=(0, 5))

        # Create listbox with scrollbar
        self.listbox_frame = ttk.Frame(self.list_frame)
        self.listbox_frame.pack(fill=tk.Y)
        
        self.scrollbar = ttk.Scrollbar(self.listbox_frame)
        self.scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        self.listbox = tk.Listbox(self.listbox_frame, 
                                 yscrollcommand=self.scrollbar.set,
                                 width=40,
                                 bg='#34495E',
                                 fg='white',
                                 selectbackground='#E74C3C',
                                 font=('Arial', 10))
        self.listbox.pack(side=tk.LEFT, fill=tk.Y)
        self.scrollbar.config(command=self.listbox.yview)

        # Bind listbox selection
        self.listbox.bind('<<ListboxSelect>>', self.on_select)

        # Create figure
        self.fig = Figure(figsize=(12, 6), facecolor='#2C3E50')
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.main_frame)
        self.canvas.get_tk_widget().pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        # Initialize variables
        self.images_dir = None
        self.masks_dir = None
        self.yolo_dir = None

    def select_image_directory(self):
        self.images_dir = filedialog.askdirectory(title="Select Images Directory")
        if self.images_dir:
            self.img_dir_label.config(text=os.path.basename(self.images_dir), foreground='#2ECC71')
            self.update_file_list()

    def select_mask_directory(self):
        self.masks_dir = filedialog.askdirectory(title="Select Binary Masks Directory")
        if self.masks_dir:
            self.mask_dir_label.config(text=os.path.basename(self.masks_dir), foreground='#2ECC71')

    def select_yolo_directory(self):
        self.yolo_dir = filedialog.askdirectory(title="Select YOLO Labels Directory")
        if self.yolo_dir:
            self.yolo_dir_label.config(text=os.path.basename(self.yolo_dir), foreground='#2ECC71')

    def update_file_list(self):
        self.listbox.delete(0, tk.END)
        if not self.images_dir:
            return

        image_files = sorted([f for f in os.listdir(self.images_dir) 
                            if f.endswith(('.png', '.jpg', '.jpeg'))])
        for file in image_files:
            self.listbox.insert(tk.END, file)

    def create_mask_from_yolo(self, yolo_path, image_shape):
        mask = np.zeros(image_shape[:2], dtype=np.uint8)
        
        with open(yolo_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) > 1:  # Ensure we have points
                    # Convert points to pixel coordinates
                    points = []
                    for i in range(1, len(parts), 2):
                        if i + 1 < len(parts):
                            x = float(parts[i]) * image_shape[1]  # width
                            y = float(parts[i + 1]) * image_shape[0]  # height
                            points.append([int(x), int(y)])
                    
                    if len(points) > 2:
                        points = np.array(points, dtype=np.int32)
                        cv2.fillPoly(mask, [points], 1)
        
        return mask

    def on_select(self, event):
        if not self.listbox.curselection():
            return

        selection = self.listbox.get(self.listbox.curselection())
        image_path = os.path.join(self.images_dir, selection)
        
        # Get corresponding mask path
        mask_path = None
        if self.masks_dir:
            mask_path = os.path.join(self.masks_dir, selection)
        
        # Get corresponding YOLO label path
        yolo_path = None
        if self.yolo_dir:
            base_name = os.path.splitext(selection)[0]
            yolo_path = os.path.join(self.yolo_dir, f"{base_name}.txt")

        if os.path.exists(image_path):
            self.display_views(image_path, mask_path, yolo_path)

    def display_views(self, image_path, mask_path, yolo_path):
        # Clear previous plots
        self.fig.clear()
        self.fig.set_facecolor('#2C3E50')

        # Create three subplots
        ax1 = self.fig.add_subplot(131)
        ax2 = self.fig.add_subplot(132)
        ax3 = self.fig.add_subplot(133)

        # Set background color for subplots
        ax1.set_facecolor('#34495E')
        ax2.set_facecolor('#34495E')
        ax3.set_facecolor('#34495E')

        # Display original image
        image = Image.open(image_path)
        img_array = np.array(image)
        ax1.imshow(image)
        ax1.set_title('Original Image', color='white', pad=10)
        ax1.axis('off')

        # Display binary mask if available
        if mask_path and os.path.exists(mask_path):
            mask = np.array(Image.open(mask_path))
            if len(mask.shape) > 2:
                mask = mask[:, :, 0]
            mask = mask.astype(np.uint8)
            
            colored_mask = np.zeros((*mask.shape, 3), dtype=np.uint8)
            colored_mask[mask == 0] = [128, 0, 128]  # Purple for background
            colored_mask[mask == 1] = [255, 255, 0]  # Yellow for mask
            
            ax2.imshow(colored_mask)
            ax2.set_title('Binary Mask', color='white', pad=10)
        else:
            ax2.text(0.5, 0.5, 'No Binary Mask Available', 
                    ha='center', va='center', color='white')
            ax2.set_title('Binary Mask View', color='white', pad=10)
        ax2.axis('off')

        # Display YOLO mask if available
        if yolo_path and os.path.exists(yolo_path):
            mask = self.create_mask_from_yolo(yolo_path, img_array.shape)
            
            # Create colored visualization
            colored_mask = np.zeros((*mask.shape, 3), dtype=np.uint8)
            colored_mask[mask == 0] = [128, 0, 128]  # Purple for background
            colored_mask[mask == 1] = [255, 255, 0]  # Yellow for mask
            
            ax3.imshow(colored_mask)
            ax3.set_title('YOLO Mask', color='white', pad=10)
        else:
            ax3.text(0.5, 0.5, 'No YOLO Label Available', 
                    ha='center', va='center', color='white')
            ax3.set_title('YOLO Mask View', color='white', pad=10)
        ax3.axis('off')

        self.fig.tight_layout()
        self.canvas.draw()

if __name__ == "__main__":
    app = ImageMaskViewer()
    app.mainloop()
