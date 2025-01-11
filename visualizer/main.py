import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
import matplotlib.pyplot as plt
from pathlib import Path
import json

from visualizer.settings import VisualizerSettings
from visualizer.plots import Plotter
from visualizer.utils import load_results, get_best_metrics, format_experiment_info

class ResultsVisualizer:
    def __init__(self, root):
        self.root = root
        self.root.title("Training Results Visualizer")
        self.root.geometry("1200x800")
        
        # Configure style
        style = ttk.Style()
        style.configure(".", font=('Arial', 10))
        style.configure("Title.TLabel", font=('Arial', 12, 'bold'), foreground='#2C3E50')
        style.configure("Header.TLabel", font=('Arial', 11, 'bold'), foreground='#34495E')
        style.configure("Info.TLabel", font=('Arial', 10))
        style.configure("TNotebook.Tab", font=('Arial', 10), padding=[10, 2])
        style.configure("Export.TButton", font=('Arial', 10), background='#3498DB')
        
        # Configure colors
        self.colors = {
            'bg': '#F8F9F9',
            'frame_bg': '#FFFFFF',
            'header_bg': '#ECF0F1',
            'text': '#2C3E50',
            'accent': '#3498DB'
        }
        
        # Set background colors
        self.root.configure(bg=self.colors['bg'])
        
        # Initialize components
        self.results_list = []
        self.file_paths = []
        self.comparison_mode = False
        self.plotter = Plotter()
        self.settings = VisualizerSettings(root, self)
        
        # Create main container
        self.main_container = ttk.Frame(self.root)
        self.main_container.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Create file selection frame
        self.create_file_selection_frame()
        
        # Create settings button
        settings_btn = ttk.Button(self.main_container, text="Settings", 
                                command=self.settings.show_settings_dialog)
        settings_btn.pack(side=tk.TOP, anchor=tk.E, padx=5, pady=5)
        
        # Create main notebook
        self.main_notebook = ttk.Notebook(self.main_container)
        self.main_notebook.pack(fill=tk.BOTH, expand=True, pady=5)
        
        # Create frames for info and plots
        self.info_frame = ttk.Frame(self.main_notebook)
        self.plots_frame = ttk.Frame(self.main_notebook)
        
        self.main_notebook.add(self.info_frame, text="Experiment Info")
        self.main_notebook.add(self.plots_frame, text="Plots")
        
        self.results_frame = None
        self.plots_notebook = None

    def create_file_selection_frame(self):
        """Create file selection frame"""
        file_frame = ttk.Frame(self.main_container)
        file_frame.pack(fill=tk.X, pady=(0, 10))
        
        # Create buttons
        ttk.Button(file_frame, text="Add File", 
                  command=self.add_file).pack(side=tk.LEFT, padx=5)
        ttk.Button(file_frame, text="Remove Selected", 
                  command=self.remove_selected_files).pack(side=tk.LEFT, padx=5)
        ttk.Button(file_frame, text="Clear All", 
                  command=self.clear_files).pack(side=tk.LEFT, padx=5)
        
        # Create listbox
        self.file_listbox = tk.Listbox(file_frame, height=3)
        self.file_listbox.pack(fill=tk.X, expand=True, padx=5)

    def add_file(self):
        """Add a file to the visualization"""
        file_path = filedialog.askopenfilename(
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")])
        if file_path and file_path not in self.file_paths:
            self.file_paths.append(file_path)
            self.file_listbox.insert(tk.END, Path(file_path).name)
            self.load_results()

    def remove_selected_files(self):
        """Remove selected files from the visualization"""
        selected = self.file_listbox.curselection()
        for index in reversed(selected):
            self.file_paths.pop(index)
            self.file_listbox.delete(index)
        self.load_results()

    def clear_files(self):
        """Clear all files from the visualization"""
        self.file_paths = []
        self.file_listbox.delete(0, tk.END)
        if self.results_frame:
            self.results_frame.destroy()
            self.results_frame = None
        self.results_list = []

    def load_results(self):
        """Load results from selected files"""
        try:
            self.results_list = []
            for file_path in self.file_paths:
                result = load_results(file_path)
                result['file_name'] = Path(file_path).name
                self.results_list.append(result)
            
            self.comparison_mode = len(self.results_list) > 1
            
            # Clear existing frames
            if self.results_frame:
                self.results_frame.destroy()
            if self.plots_notebook:
                self.plots_notebook.destroy()
            
            # Create info frame
            self.results_frame = ttk.Frame(self.info_frame)
            self.results_frame.pack(fill=tk.BOTH, expand=True)
            
            self.create_experiment_info_frame()
            
            # Create plots notebook
            self.plots_notebook = ttk.Notebook(self.plots_frame)
            self.plots_notebook.pack(fill=tk.BOTH, expand=True)
            
            # Create plot tabs for each metric
            self.create_loss_plot()
            self.create_metrics_plots()
            
        except Exception as e:
            messagebox.showerror("Error", str(e))

    def create_experiment_info_frame(self):
        """Create experiment information frame"""
        # Create a canvas and scrollbar for scrolling
        canvas = tk.Canvas(self.results_frame)
        scrollbar = ttk.Scrollbar(self.results_frame, orient="vertical", command=canvas.yview)
        scrollable_frame = ttk.Frame(canvas)

        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )

        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)

        # Pack the canvas and scrollbar
        canvas.pack(side="left", fill="both", expand=True, padx=5)
        scrollbar.pack(side="right", fill="y")

        for i, result in enumerate(self.results_list):
            # Create frame with export option
            frame = ttk.LabelFrame(scrollable_frame, text=f"Experiment {i+1}: {Path(self.file_paths[i]).name}")
            frame.pack(fill=tk.X, padx=5, pady=5)
            
            # Add export button at the top
            export_frame = ttk.Frame(frame)
            export_frame.pack(fill=tk.X, padx=5, pady=5)
            
            export_btn = ttk.Button(export_frame, text="Export Info", style="Export.TButton",
                                  command=lambda r=result: self.export_experiment_info(r))
            export_btn.pack(side=tk.RIGHT, padx=5)
            
            # Overall Best Metrics Frame (stays at top)
            best_metrics_frame = ttk.Frame(frame)
            best_metrics_frame.pack(fill=tk.X, padx=5, pady=5)
            
            # Display overall best metrics
            best_metrics = self.get_best_metrics(result)
            ttk.Label(best_metrics_frame, text="Best Metrics:", 
                     style="Title.TLabel").pack(anchor=tk.W, padx=5, pady=2)
            for metric in best_metrics:
                ttk.Label(best_metrics_frame, text=metric,
                         style="Info.TLabel").pack(anchor=tk.W, padx=20, pady=1)
            
            # Create notebook for different info sections
            notebook = ttk.Notebook(frame)
            notebook.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
            
            # Create tabs
            self.create_basic_info_tab(notebook, result)
            self.create_dataset_info_tab(notebook, result)
            self.create_hardware_info_tab(notebook, result)
            self.create_training_info_tab(notebook, result)
            self.create_time_info_tab(notebook, result)
            self.create_detailed_metrics_tab(notebook, result)

        # Configure the canvas to handle mousewheel
        def _on_mousewheel(event):
            canvas.yview_scroll(int(-1*(event.delta/120)), "units")
        canvas.bind_all("<MouseWheel>", _on_mousewheel)

    def get_best_metrics(self, result):
        """Get formatted best metrics strings"""
        metrics = []
        metrics_list = ['IoU', 'Precision', 'Recall', 'Dice Score', 'Accuracy']
        
        # First try to get metrics from training_info
        training_info = result.get('training_info', {})
        if 'best_metrics' in training_info and 'best_epoch' in training_info:
            best_metrics = training_info['best_metrics']
            for metric, value in best_metrics.items():
                if metric != 'F1-Score':  # Skip F1-Score as it's redundant with Dice Score
                    metrics.append(f"Best {metric}: {float(value):.4f} (Epoch {training_info['best_epoch']})")
        
        # Calculate individual best epochs for each metric
        epochs = result.get('epochs', [])
        if epochs:
            metrics.append("\nBest Epochs per Metric:")
            for metric in metrics_list:
                try:
                    # Find best value and its epoch for this metric
                    best_val = max(float(e['valid_metrics'][metric]) for e in epochs)
                    best_epoch = next(e['epoch'] for e in epochs 
                                    if float(e['valid_metrics'][metric]) == best_val)
                    metrics.append(f"{metric}: Best at Epoch {best_epoch} ({best_val:.4f})")
                except (KeyError, ValueError):
                    continue
        
        return metrics

    def create_basic_info_tab(self, notebook, result):
        """Create basic info tab"""
        frame = ttk.Frame(notebook)
        notebook.add(frame, text="Basic Info")
        
        try:
            # Get architecture and encoder info
            arch = result['training_info']['architecture']
            encoder = result['training_info']['encoder']
            self.create_info_row(frame, "Architecture", f"{arch} with {encoder}")
            
            # Get image size
            img_size = f"{result['training_info']['image_size']['width']}x{result['training_info']['image_size']['height']}"
            self.create_info_row(frame, "Image Size", img_size)
            
            # Get hardware info
            device = result['hardware_info']['device']
            gpu_memory = result['hardware_info'].get('gpu_memory', 'Not available')
            peak_gpu_memory = result['hardware_info'].get('peak_gpu_memory', 'Not available')
            self.create_info_row(frame, "Device", device)
            self.create_info_row(frame, "GPU Memory", gpu_memory)
            self.create_info_row(frame, "Peak GPU Memory", peak_gpu_memory)
            
            # Get training info
            batch_size = result['training_info']['batch_size']
            total_epochs = len(result['epochs'])
            self.create_info_row(frame, "Batch Size", batch_size)
            self.create_info_row(frame, "Total Epochs", total_epochs)
            
            # Get timing info
            start_time = result.get('training_info', {}).get('start_time', 'Not available')
            self.create_info_row(frame, "Start Time", start_time)
            
            # Calculate total training time
            total_time = sum(epoch.get('time', 0) for epoch in result['epochs'])
            if total_time > 0:
                hours = int(total_time // 3600)
                minutes = int((total_time % 3600) // 60)
                seconds = int(total_time % 60)
                time_str = f"{hours}h {minutes}m {seconds}s"
            else:
                time_str = 'Not available'
            
            self.create_info_row(frame, "Total Training Time", time_str)
            
        except KeyError as e:
            self.create_info_row(frame, "Error", f"Missing data: {str(e)}")
        except Exception as e:
            self.create_info_row(frame, "Error", f"Failed to load info: {str(e)}")

    def create_dataset_info_tab(self, notebook, result):
        """Create dataset information tab"""
        frame = ttk.Frame(notebook)
        notebook.add(frame, text="Dataset Info")
        
        dataset_info = result.get('dataset_info', {})
        info = [
            ("Train Size", dataset_info.get('train_size', 'N/A')),
            ("Valid Size", dataset_info.get('valid_size', 'N/A')),
            ("Batch Size", dataset_info.get('batch_size', 'N/A')),
            ("Image Size", f"{dataset_info.get('image_size', {}).get('width', 'N/A')}x{dataset_info.get('image_size', {}).get('height', 'N/A')}")
        ]
        
        for label, value in info:
            self.create_info_row(frame, label, value)

    def create_hardware_info_tab(self, notebook, result):
        """Create hardware information tab"""
        frame = ttk.Frame(notebook)
        notebook.add(frame, text="Hardware Info")
        
        hardware_info = result.get('hardware_info', {})
        info = [
            ("Device", hardware_info.get('device', 'N/A')),
            ("GPU Memory", hardware_info.get('gpu_memory', 'N/A')),
            ("Peak GPU Memory", hardware_info.get('peak_gpu_memory', 'N/A'))
        ]
        
        for label, value in info:
            self.create_info_row(frame, label, value)

    def create_training_info_tab(self, notebook, result):
        """Create training information tab"""
        frame = ttk.Frame(notebook)
        notebook.add(frame, text="Training Info")
        
        training_info = result.get('training_info', {})
        info = [
            ("Architecture", training_info.get('architecture', 'N/A')),
            ("Encoder", training_info.get('encoder', 'N/A')),
            ("Pretrained", training_info.get('pretrained_weights', 'N/A')),
            ("Initial Epochs", training_info.get('initial_epochs', 'N/A'))
        ]
        
        for label, value in info:
            self.create_info_row(frame, label, value)

    def create_time_info_tab(self, notebook, result):
        """Create time information tab"""
        frame = ttk.Frame(notebook)
        notebook.add(frame, text="Time Info")
        
        try:
            # Get start date and time from training_info
            start_time = result.get('training_info', {}).get('start_time', 'Not available')
            end_time = result.get('training_info', {}).get('end_time', 'Not available')
            
            # Calculate total training time from epochs
            total_time = sum(epoch.get('time', 0) for epoch in result['epochs'])
            if total_time > 0:
                hours = int(total_time // 3600)
                minutes = int((total_time % 3600) // 60)
                seconds = int(total_time % 60)
                time_str = f"{hours}h {minutes}m {seconds}s"
            else:
                time_str = 'Not available'
            
            self.create_info_row(frame, "Start Time", start_time)
            self.create_info_row(frame, "End Time", end_time)
            self.create_info_row(frame, "Total Training Time", time_str)
            
            # Add average epoch time if available
            if total_time > 0 and len(result['epochs']) > 0:
                avg_epoch_time = total_time / len(result['epochs'])
                avg_minutes = int(avg_epoch_time // 60)
                avg_seconds = int(avg_epoch_time % 60)
                avg_time_str = f"{avg_minutes}m {avg_seconds}s"
                self.create_info_row(frame, "Average Epoch Time", avg_time_str)
            
        except Exception as e:
            self.create_info_row(frame, "Error", f"Failed to load time info: {str(e)}")

    def create_detailed_metrics_tab(self, notebook, result):
        """Create detailed metrics tab"""
        frame = ttk.Frame(notebook)
        notebook.add(frame, text="Detailed Metrics")
        
        # First try to get metrics from training_info
        training_info = result.get('training_info', {})
        if 'best_metrics' in training_info and 'best_epoch' in training_info:
            best_metrics = training_info['best_metrics']
            best_epoch = training_info['best_epoch']
            
            for metric, value in best_metrics.items():
                if metric != 'F1-Score':  # Skip F1-Score as it's redundant with Dice Score
                    self.create_info_row(frame, f"Best {metric}", 
                                       f"{float(value):.4f} (Epoch {best_epoch})")
        else:
            # Calculate from epochs if training_info metrics not available
            metrics = [
                ("IoU", "IoU"),
                ("Precision", "Precision"),
                ("Recall", "Recall"),
                ("Dice Score", "Dice Score"),
                ("Accuracy", "Accuracy")
            ]
            
            epochs = result.get('epochs', [])
            if epochs:
                for metric_label, metric_key in metrics:
                    try:
                        best_val = max(float(e['valid_metrics'][metric_key]) for e in epochs)
                        best_epoch = next(e['epoch'] for e in epochs 
                                        if float(e['valid_metrics'][metric_key]) == best_val)
                        self.create_info_row(frame, f"Best {metric_label}", 
                                           f"{best_val:.4f} (Epoch {best_epoch})")
                    except (KeyError, ValueError):
                        continue

    def create_info_row(self, parent, label, value):
        """Create a row with label and value"""
        frame = ttk.Frame(parent)
        frame.pack(fill=tk.X, padx=5, pady=2)
        
        ttk.Label(frame, text=f"{label}:", style="Header.TLabel").pack(side=tk.LEFT, padx=5)
        ttk.Label(frame, text=str(value), style="Info.TLabel").pack(side=tk.LEFT, padx=5)

    def create_loss_plot(self):
        """Create loss plot tab"""
        fig = plt.figure(figsize=(10, 6))
        canvas_frame = ttk.Frame(self.plots_notebook)
        self.plots_notebook.add(canvas_frame, text="Loss Plot")
        
        # Create control frame
        control_frame = ttk.Frame(canvas_frame)
        control_frame.pack(fill=tk.X, padx=5, pady=5)
        
        # Add export button
        export_btn = ttk.Button(control_frame, text="Export Plot", 
                              command=lambda: self.export_plot(fig, "loss_plot"))
        export_btn.pack(side=tk.LEFT, padx=5)
        
        # Add configure button
        configure_btn = ttk.Button(control_frame, text="Configure Plot", 
                                 command=lambda: self.configure_plot(fig))
        configure_btn.pack(side=tk.LEFT, padx=5)
        
        ax = fig.add_subplot(111)
        ax.grid(True, linestyle='--', alpha=0.7)
        
        for i, result in enumerate(self.results_list):
            epochs_data = result['epochs']
            epochs = [e['epoch'] for e in epochs_data]
            label_suffix = f" ({result['file_name']})" if len(self.results_list) > 1 else ""
            
            # Plot validation loss
            valid_loss = [e['valid_loss'] for e in epochs_data]
            ax.plot(epochs, valid_loss, label=f'Validation Loss{label_suffix}', 
                   linewidth=2, marker='o', markersize=4)
        
        ax.set_xlabel('Epoch', fontsize=12, labelpad=10)
        ax.set_ylabel('Loss', fontsize=12, labelpad=10)
        ax.set_title('Validation Loss Over Time', fontsize=14, pad=15)
        ax.legend(frameon=True, fancybox=True, shadow=True, fontsize=10)
        ax.tick_params(axis='both', which='major', labelsize=10)
        
        # Adjust layout to prevent label cutoff
        plt.tight_layout()
        
        canvas = FigureCanvasTkAgg(fig, master=canvas_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

    def create_metrics_plots(self):
        """Create metrics plot tabs"""
        metrics = ['IoU', 'Precision', 'Recall', 'Dice Score', 'Accuracy']
        
        for metric in metrics:
            fig = plt.figure(figsize=(10, 6))
            canvas_frame = ttk.Frame(self.plots_notebook)
            self.plots_notebook.add(canvas_frame, text=f"{metric} Plot")
            
            # Create control frame
            control_frame = ttk.Frame(canvas_frame)
            control_frame.pack(fill=tk.X, padx=5, pady=5)
            
            # Add export button
            export_btn = ttk.Button(control_frame, text="Export Plot", 
                                  command=lambda f=fig, m=metric: self.export_plot(f, f"{m.lower()}_plot"))
            export_btn.pack(side=tk.LEFT, padx=5)
            
            # Add configure button
            configure_btn = ttk.Button(control_frame, text="Configure Plot", 
                                     command=lambda f=fig: self.configure_plot(f))
            configure_btn.pack(side=tk.LEFT, padx=5)
            
            ax = fig.add_subplot(111)
            ax.grid(True, linestyle='--', alpha=0.7)
            
            for i, result in enumerate(self.results_list):
                epochs_data = result['epochs']
                epochs = [e['epoch'] for e in epochs_data]
                label_suffix = f" ({result['file_name']})" if len(self.results_list) > 1 else ""
                
                # Plot validation metrics
                valid_values = [float(e['valid_metrics'][metric]) for e in epochs_data]  # Convert to float
                ax.plot(epochs, valid_values, label=f'Validation {metric}{label_suffix}', 
                       linewidth=2, marker='o', markersize=4)
            
            ax.set_xlabel('Epoch', fontsize=12, labelpad=10)
            ax.set_ylabel(f'{metric} (Raw Value)', fontsize=12, labelpad=10)  # Updated label
            ax.set_title(f'{metric} Over Time (Raw Values)', fontsize=14, pad=15)  # Updated title
            ax.legend(frameon=True, fancybox=True, shadow=True, fontsize=10)
            ax.tick_params(axis='both', which='major', labelsize=10)
            
            # Set y-axis limits for metrics (between 0 and 1)
            if metric in ['IoU', 'Precision', 'Recall', 'Dice Score', 'Accuracy']:
                ax.set_ylim(0, 1)
                ax.yaxis.set_major_locator(plt.LinearLocator(11))  # Show 11 ticks from 0 to 1
                ax.yaxis.set_major_formatter(plt.FormatStrFormatter('%.2f'))  # Format as decimal
            
            # Adjust layout to prevent label cutoff
            plt.tight_layout()
            
            canvas = FigureCanvasTkAgg(fig, master=canvas_frame)
            canvas.draw()
            canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

    def export_plot(self, fig, default_name):
        """Export plot to file"""
        file_path = filedialog.asksaveasfilename(
            defaultextension=".png",
            filetypes=[("PNG files", "*.png"), 
                      ("PDF files", "*.pdf"),
                      ("SVG files", "*.svg")],
            initialfile=default_name
        )
        if file_path:
            try:
                fig.savefig(file_path, bbox_inches='tight', dpi=300)
                messagebox.showinfo("Success", "Plot exported successfully!")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to export plot: {str(e)}")

    def configure_plot(self, fig):
        """Configure plot settings"""
        config_window = tk.Toplevel(self.root)
        config_window.title("Configure Plot")
        config_window.geometry("300x400")
        
        # Title settings
        title_frame = ttk.LabelFrame(config_window, text="Title")
        title_frame.pack(fill=tk.X, padx=5, pady=5)
        
        title_var = tk.StringVar(value=fig.axes[0].get_title())
        ttk.Label(title_frame, text="Title:").pack(side=tk.LEFT, padx=5)
        title_entry = ttk.Entry(title_frame, textvariable=title_var)
        title_entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5, pady=5)
        
        # Font size settings
        font_frame = ttk.LabelFrame(config_window, text="Font Sizes")
        font_frame.pack(fill=tk.X, padx=5, pady=5)
        
        title_size_var = tk.StringVar(value="14")
        label_size_var = tk.StringVar(value="12")
        tick_size_var = tk.StringVar(value="10")
        
        ttk.Label(font_frame, text="Title Size:").grid(row=0, column=0, padx=5, pady=2)
        ttk.Entry(font_frame, textvariable=title_size_var, width=5).grid(row=0, column=1, padx=5, pady=2)
        
        ttk.Label(font_frame, text="Label Size:").grid(row=1, column=0, padx=5, pady=2)
        ttk.Entry(font_frame, textvariable=label_size_var, width=5).grid(row=1, column=1, padx=5, pady=2)
        
        ttk.Label(font_frame, text="Tick Size:").grid(row=2, column=0, padx=5, pady=2)
        ttk.Entry(font_frame, textvariable=tick_size_var, width=5).grid(row=2, column=1, padx=5, pady=2)
        
        # Grid settings
        grid_frame = ttk.LabelFrame(config_window, text="Grid")
        grid_frame.pack(fill=tk.X, padx=5, pady=5)
        
        grid_var = tk.BooleanVar(value=True)
        grid_alpha_var = tk.DoubleVar(value=0.7)
        
        ttk.Checkbutton(grid_frame, text="Show Grid", variable=grid_var).pack(padx=5, pady=2)
        ttk.Label(grid_frame, text="Grid Alpha:").pack(side=tk.LEFT, padx=5)
        ttk.Entry(grid_frame, textvariable=grid_alpha_var, width=5).pack(side=tk.LEFT, padx=5)
        
        def apply_settings():
            try:
                ax = fig.axes[0]
                
                # Update title
                ax.set_title(title_var.get(), fontsize=int(title_size_var.get()), pad=15)
                
                # Update font sizes
                ax.set_xlabel(ax.get_xlabel(), fontsize=int(label_size_var.get()), labelpad=10)
                ax.set_ylabel(ax.get_ylabel(), fontsize=int(label_size_var.get()), labelpad=10)
                ax.tick_params(axis='both', which='major', labelsize=int(tick_size_var.get()))
                
                # Update grid
                ax.grid(grid_var.get(), linestyle='--', alpha=grid_alpha_var.get())
                
                # Redraw
                plt.tight_layout()
                fig.canvas.draw()
                
                config_window.destroy()
                
            except Exception as e:
                messagebox.showerror("Error", f"Failed to apply settings: {str(e)}")
        
        # Apply button
        ttk.Button(config_window, text="Apply", command=apply_settings).pack(pady=10)

    def on_plot_select(self, event=None):
        """Update entry fields when a plot is selected"""
        selected_plot = self.settings.plot_select.get()
        if selected_plot:
            self.settings.title_var.set(self.settings.plot_titles.get(selected_plot, ''))
            self.settings.legend_var.set(self.settings.legend_labels.get(selected_plot, ''))

    def update_plot_title(self):
        """Update the title for the selected plot"""
        selected_plot = self.settings.plot_select.get()
        if selected_plot and self.settings.title_var.get():
            self.settings.plot_titles[selected_plot] = self.settings.title_var.get()
            self.settings.save_settings()

    def update_legend_label(self):
        """Update the legend label for the selected plot"""
        selected_plot = self.settings.plot_select.get()
        if selected_plot and self.settings.legend_var.get():
            self.settings.legend_labels[selected_plot] = self.settings.legend_var.get()
            self.settings.save_settings()

if __name__ == "__main__":
    root = tk.Tk()
    app = ResultsVisualizer(root)
    root.mainloop()
