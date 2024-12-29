import tkinter as tk
from tkinter import ttk, filedialog, messagebox, colorchooser
import json
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import numpy as np
from matplotlib.font_manager import FontProperties
import seaborn as sns
from pathlib import Path

class ResultsVisualizer:
    def __init__(self, root):
        self.root = root
        self.root.title("Training Results Visualizer")
        self.root.geometry("1200x800")
        
        # Initialize results storage
        self.results_list = []
        self.file_paths = []
        self.plot_colors = {}
        self.plot_titles = {}
        self.legend_labels = {}
        
        # Load saved settings or use defaults
        self.load_settings()
        
        # Set style
        plt.style.use('seaborn-v0_8-darkgrid')
        self.default_colors = sns.color_palette("husl", n_colors=10)
        
        # Configure fonts
        plt.rcParams['font.family'] = 'sans-serif'
        plt.rcParams['font.sans-serif'] = ['Arial']
        plt.rcParams['axes.labelsize'] = 10
        plt.rcParams['axes.titlesize'] = 12
        plt.rcParams['xtick.labelsize'] = 9
        plt.rcParams['ytick.labelsize'] = 9
        
        # Configure style
        style = ttk.Style()
        style.configure("TLabel", font=('Arial', 10))
        style.configure("TButton", font=('Arial', 10))
        style.configure("TNotebook.Tab", font=('Arial', 10))
        
        # Create main container
        self.main_container = ttk.Frame(self.root)
        self.main_container.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Create file selection frame
        self.create_file_selection_frame()
        
        # Create placeholder for other widgets
        self.results_frame = None
        self.notebook = None
        self.comparison_mode = False
        
        # Create settings button
        settings_btn = ttk.Button(self.main_container, text="Settings", command=self.show_settings)
        settings_btn.pack(side=tk.TOP, anchor=tk.E, padx=5, pady=5)
        
    def load_settings(self):
        """Load settings from JSON file"""
        try:
            with open('visualizer_settings.json', 'r') as f:
                self.settings = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            self.settings = {
                'metric_type': 'valid',
                'plot_titles': {},
                'legend_labels': {},
                'plot_colors': {},
                'title_size': '14',
                'label_size': '12',
                'legend_size': '10'
            }
        
        # Apply loaded settings
        self.metric_type = tk.StringVar(value=self.settings.get('metric_type', 'valid'))
        self.plot_titles = self.settings.get('plot_titles', {})
        self.legend_labels = self.settings.get('legend_labels', {})
        self.plot_colors = self.settings.get('plot_colors', {})
        self.title_size = tk.StringVar(value=self.settings.get('title_size', '14'))
        self.label_size = tk.StringVar(value=self.settings.get('label_size', '12'))
        self.legend_size = tk.StringVar(value=self.settings.get('legend_size', '10'))
        
    def save_settings(self):
        """Save current settings to JSON file"""
        settings = {
            'metric_type': self.metric_type.get(),
            'plot_titles': self.plot_titles,
            'legend_labels': self.legend_labels,
            'plot_colors': self.plot_colors,
            'title_size': self.title_size.get(),
            'label_size': self.label_size.get(),
            'legend_size': self.legend_size.get()
        }
        
        with open('visualizer_settings.json', 'w') as f:
            json.dump(settings, f, indent=4)
            
    def show_settings(self):
        """Show settings dialog"""
        settings_window = tk.Toplevel(self.root)
        settings_window.title("Settings")
        settings_window.geometry("600x500")
        
        settings_frame = ttk.Frame(settings_window)
        settings_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Metric type selection
        type_frame = ttk.LabelFrame(settings_frame, text="Metric Display Options")
        type_frame.pack(fill=tk.X, pady=(0, 10))
        
        ttk.Label(type_frame, text="Default Metrics:").pack(side=tk.LEFT, padx=5, pady=5)
        ttk.Radiobutton(type_frame, text="Valid Only", variable=self.metric_type, 
                       value='valid').pack(side=tk.LEFT, padx=5)
        ttk.Radiobutton(type_frame, text="Train Only", variable=self.metric_type, 
                       value='train').pack(side=tk.LEFT, padx=5)
        ttk.Radiobutton(type_frame, text="Both", variable=self.metric_type, 
                       value='both').pack(side=tk.LEFT, padx=5)
        
        # Plot customization
        plot_frame = ttk.LabelFrame(settings_frame, text="Plot Customization")
        plot_frame.pack(fill=tk.X, pady=10)
        
        # Plot selection
        select_frame = ttk.Frame(plot_frame)
        select_frame.pack(fill=tk.X, padx=5, pady=5)
        
        ttk.Label(select_frame, text="Select Plot:").pack(side=tk.LEFT, padx=5)
        self.plot_select = ttk.Combobox(select_frame, 
            values=['Loss', 'IoU', 'Precision', 'Recall', 'Dice Score', 'Accuracy'])
        self.plot_select.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
        self.plot_select.bind('<<ComboboxSelected>>', self.on_plot_select)
        
        # Title customization
        title_frame = ttk.Frame(plot_frame)
        title_frame.pack(fill=tk.X, padx=5, pady=5)
        
        ttk.Label(title_frame, text="Default Title:").pack(side=tk.LEFT, padx=5)
        self.title_var = tk.StringVar()
        title_entry = ttk.Entry(title_frame, textvariable=self.title_var)
        title_entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
        ttk.Button(title_frame, text="Set", 
                  command=self.update_plot_title).pack(side=tk.LEFT, padx=5)
        
        # Legend customization
        legend_frame = ttk.Frame(plot_frame)
        legend_frame.pack(fill=tk.X, padx=5, pady=5)
        
        ttk.Label(legend_frame, text="Default Legend:").pack(side=tk.LEFT, padx=5)
        self.legend_var = tk.StringVar()
        legend_entry = ttk.Entry(legend_frame, textvariable=self.legend_var)
        legend_entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
        ttk.Button(legend_frame, text="Set", 
                  command=self.update_legend_label).pack(side=tk.LEFT, padx=5)
        
        # Color settings
        color_frame = ttk.LabelFrame(settings_frame, text="Color Settings")
        color_frame.pack(fill=tk.X, pady=10)
        
        def choose_color(element):
            color = colorchooser.askcolor(
                color=self.plot_colors.get(element),
                title=f"Choose color for {element}")[1]
            if color:
                self.plot_colors[element] = color
        
        elements = ['Train Line', 'Validation Line', 'Grid', 'Background']
        for i, element in enumerate(elements):
            frame = ttk.Frame(color_frame)
            frame.pack(fill=tk.X, padx=5, pady=2)
            ttk.Label(frame, text=f"{element}:").pack(side=tk.LEFT, padx=5)
            ttk.Button(frame, text="Select Color", 
                      command=lambda e=element: choose_color(e)).pack(side=tk.LEFT, padx=5)
        
        # Font settings
        font_frame = ttk.LabelFrame(settings_frame, text="Font Settings")
        font_frame.pack(fill=tk.X, pady=10)
        
        sizes_frame = ttk.Frame(font_frame)
        sizes_frame.pack(fill=tk.X, padx=5, pady=5)
        
        ttk.Label(sizes_frame, text="Title Size:").grid(row=0, column=0, padx=5, pady=2)
        ttk.Entry(sizes_frame, textvariable=self.title_size, width=5).grid(row=0, column=1, padx=5, pady=2)
        
        ttk.Label(sizes_frame, text="Label Size:").grid(row=0, column=2, padx=5, pady=2)
        ttk.Entry(sizes_frame, textvariable=self.label_size, width=5).grid(row=0, column=3, padx=5, pady=2)
        
        ttk.Label(sizes_frame, text="Legend Size:").grid(row=0, column=4, padx=5, pady=2)
        ttk.Entry(sizes_frame, textvariable=self.legend_size, width=5).grid(row=0, column=5, padx=5, pady=2)
        
        # Apply and Close buttons
        button_frame = ttk.Frame(settings_frame)
        button_frame.pack(fill=tk.X, pady=10)
        
        def apply_settings():
            self.save_settings()
            if hasattr(self, 'notebook') and self.notebook:
                # Remove existing tabs
                for tab_id in self.notebook.tabs():
                    self.notebook.forget(tab_id)
                
                # Recreate plots
                self.create_loss_plot()
                self.create_metrics_plots()
                
                # Update comparison tabs if in comparison mode
                if self.comparison_mode:
                    self.create_comparison_tabs()
        
        ttk.Button(button_frame, text="Apply", command=apply_settings).pack(side=tk.RIGHT, padx=5)
        ttk.Button(button_frame, text="Close", command=settings_window.destroy).pack(side=tk.RIGHT, padx=5)
        
    def save_and_update(self):
        """Save settings and update plots if they exist"""
        self.save_settings()
        if hasattr(self, 'notebook') and self.notebook:
            for tab_id in self.notebook.tabs():
                tab = self.notebook.select(tab_id)
                for widget in self.notebook.children[tab_id.split(".")[-1]].winfo_children():
                    if isinstance(widget, FigureCanvasTkAgg):
                        widget.figure.clear()
                        if "Loss" in self.notebook.tab(tab_id, "text"):
                            self.plot_loss(widget.figure)
                        elif "Metrics" in self.notebook.tab(tab_id, "text"):
                            self.plot_metrics(widget.figure)
                        widget.draw()
            
    def update_plots(self):
        """Update all plots with current settings"""
        if hasattr(self, 'notebook') and self.notebook:
            for tab_id in self.notebook.tabs():
                tab = self.notebook.select(tab_id)
                for widget in self.notebook.children[tab_id.split(".")[-1]].winfo_children():
                    if isinstance(widget, FigureCanvasTkAgg):
                        widget.figure.clear()
                        if "Loss" in self.notebook.tab(tab_id, "text"):
                            self.plot_loss(widget.figure)
                        elif "Metrics" in self.notebook.tab(tab_id, "text"):
                            self.plot_metrics(widget.figure)
                        widget.draw()

    def create_file_selection_frame(self):
        """Create file selection frame"""
        file_frame = ttk.Frame(self.main_container)
        file_frame.pack(fill=tk.X, padx=5, pady=5)
        
        # File list frame
        list_frame = ttk.Frame(file_frame)
        list_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        # Create listbox for files
        self.file_listbox = tk.Listbox(list_frame, height=3, selectmode=tk.MULTIPLE)
        self.file_listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        # Scrollbar for listbox
        scrollbar = ttk.Scrollbar(list_frame, orient=tk.VERTICAL, command=self.file_listbox.yview)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.file_listbox.config(yscrollcommand=scrollbar.set)
        
        # Buttons frame
        btn_frame = ttk.Frame(file_frame)
        btn_frame.pack(side=tk.LEFT, fill=tk.Y, padx=5)
        
        add_btn = ttk.Button(btn_frame, text="Add File", command=self.add_file)
        add_btn.pack(side=tk.TOP, padx=5, pady=5)
        
        remove_btn = ttk.Button(btn_frame, text="Remove Selected", command=self.remove_selected_files)
        remove_btn.pack(side=tk.TOP, padx=5, pady=5)
        
        clear_btn = ttk.Button(btn_frame, text="Clear All", command=self.clear_files)
        clear_btn.pack(side=tk.TOP, padx=5, pady=5)
        
        load_btn = ttk.Button(btn_frame, text="Load Results", command=self.load_results)
        load_btn.pack(side=tk.TOP, padx=5, pady=5)
        
    def add_file(self):
        filename = filedialog.askopenfilename(
            title="Select Results JSON File",
            filetypes=[("JSON files", "*.json")]
        )
        if filename and filename not in self.file_paths:
            self.file_paths.append(filename)
            self.file_listbox.insert(tk.END, Path(filename).name)
            
    def remove_selected_files(self):
        selected = self.file_listbox.curselection()
        for idx in reversed(selected):
            self.file_paths.pop(idx)
            self.file_listbox.delete(idx)
            
    def clear_files(self):
        self.file_paths.clear()
        self.file_listbox.delete(0, tk.END)
        
    def load_results(self):
        if not self.file_paths:
            messagebox.showwarning("Warning", "Please add at least one results file!")
            return
            
        try:
            # Clear previous results
            if self.results_frame:
                self.results_frame.destroy()
            if self.notebook:
                self.notebook.destroy()
                
            self.results_list = []
            self.comparison_mode = len(self.file_paths) > 1
            
            # Load all results files
            for file_path in self.file_paths:
                with open(file_path, 'r') as f:
                    result_data = json.load(f)
                    result_data['file_name'] = Path(file_path).name
                    self.results_list.append(result_data)
                    
            # Create results frame
            self.results_frame = ttk.Frame(self.main_container)
            self.results_frame.pack(fill=tk.BOTH, expand=True)
            
            # Create experiment info frame
            self.create_experiment_info_frame()
            
            # Create best metrics frame
            self.create_best_metrics_frame()
            
            # Create notebook for plots
            self.notebook = ttk.Notebook(self.results_frame)
            self.notebook.pack(fill=tk.BOTH, expand=True, pady=(10, 0))
            
            # Add comparison tabs
            if len(self.results_list) > 1:
                self.create_comparison_tabs()
            
            # Add plot tabs
            self.create_loss_plot()
            self.create_metrics_plots()
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load results: {str(e)}")
            
    def plot_loss(self, fig=None):
        """Plot loss curves"""
        if fig is None:
            fig = plt.figure(figsize=(10, 6))
            
        ax = fig.add_subplot(111)
        
        for i, result in enumerate(self.results_list):
            epochs_data = result['epochs']
            epochs = [e['epoch'] for e in epochs_data]
            label_suffix = f" ({result['file_name']})" if self.comparison_mode else ""
            train_color = self.plot_colors.get('Train Line', self.default_colors[i*2])
            valid_color = self.plot_colors.get('Validation Line', self.default_colors[i*2+1])
            
            if self.metric_type.get() in ['both', 'train']:
                train_loss = [e['train_loss'] for e in epochs_data]
                ax.plot(epochs, train_loss, label=f'Train Loss{label_suffix}', 
                       color=train_color, linewidth=2)
                
            if self.metric_type.get() in ['both', 'valid']:
                valid_loss = [e['valid_loss'] for e in epochs_data]
                ax.plot(epochs, valid_loss, label=f'Validation Loss{label_suffix}', 
                       color=valid_color, linewidth=2)
        
        ax.set_xlabel('Epoch', fontsize=int(self.label_size.get()))
        ax.set_ylabel('Loss', fontsize=int(self.label_size.get()))
        ax.set_title(self.plot_titles.get('Loss', 'Training and Validation Loss'), 
                    fontsize=int(self.title_size.get()), pad=15)
        ax.legend(frameon=True, fancybox=True, shadow=True, fontsize=int(self.legend_size.get()))
        
        if 'Grid' in self.plot_colors:
            ax.grid(True, color=self.plot_colors['Grid'], alpha=0.3)
        else:
            ax.grid(True, alpha=0.3)
            
        if 'Background' in self.plot_colors:
            ax.set_facecolor(self.plot_colors['Background'])
            fig.patch.set_facecolor(self.plot_colors['Background'])
            
        fig.tight_layout()
        return fig

    def plot_metrics(self, fig=None):
        """Plot all metrics"""
        if fig is None:
            fig = plt.figure(figsize=(15, 10))
            
        gs = fig.add_gridspec(2, 3, hspace=0.3, wspace=0.3)
        metrics = ['IoU', 'Precision', 'Recall', 'Dice Score', 'Accuracy']
        
        for i, metric in enumerate(metrics):
            ax = fig.add_subplot(gs[i//3, i%3])
            
            for j, result in enumerate(self.results_list):
                epochs_data = result['epochs']
                epochs = [e['epoch'] for e in epochs_data]
                label_suffix = f" ({result['file_name']})" if self.comparison_mode else ""
                train_color = self.plot_colors.get('Train Line', self.default_colors[j*2])
                valid_color = self.plot_colors.get('Validation Line', self.default_colors[j*2+1])
                
                if self.metric_type.get() in ['both', 'train']:
                    train_metric = [e['train_metrics'][metric] for e in epochs_data]
                    ax.plot(epochs, train_metric, 
                           label=self.legend_labels.get(f'{metric}_train', f'Train {metric}{label_suffix}'),
                           color=train_color, linewidth=2)
                
                if self.metric_type.get() in ['both', 'valid']:
                    valid_metric = [e['valid_metrics'][metric] for e in epochs_data]
                    ax.plot(epochs, valid_metric, 
                           label=self.legend_labels.get(f'{metric}_valid', f'Valid {metric}{label_suffix}'),
                           color=valid_color, linewidth=2)
            
            ax.set_xlabel('Epoch', fontsize=int(self.label_size.get()))
            ax.set_ylabel(metric.replace(' ', '\n'), fontsize=int(self.label_size.get()))
            ax.set_title(self.plot_titles.get(metric, f'{metric} over Epochs'), 
                        fontsize=int(self.title_size.get()), pad=10)
            ax.legend(frameon=True, fancybox=True, shadow=True, fontsize=int(self.legend_size.get()))
            
            if 'Grid' in self.plot_colors:
                ax.grid(True, color=self.plot_colors['Grid'], alpha=0.3)
            else:
                ax.grid(True, alpha=0.3)
                
            if 'Background' in self.plot_colors:
                ax.set_facecolor(self.plot_colors['Background'])
                fig.patch.set_facecolor(self.plot_colors['Background'])
        
        fig.tight_layout()
        return fig

    def create_experiment_info_frame(self):
        info_frame = ttk.LabelFrame(self.results_frame, text="Experiment Information")
        info_frame.pack(fill=tk.X, pady=(0, 10))
        
        # Create a notebook for multiple experiment info
        info_notebook = ttk.Notebook(info_frame)
        info_notebook.pack(fill=tk.BOTH, expand=True)
        
        # Create a tab for each result file
        for i, result in enumerate(self.results_list):
            exp_frame = ttk.Frame(info_notebook)
            info_notebook.add(exp_frame, text=f"Experiment {i+1}")
            
            # Extract info
            arch = result['training_info']['architecture']
            encoder = result['training_info']['encoder']
            img_size = f"{result['training_info']['image_size']['width']}x{result['training_info']['image_size']['height']}"
            device = result['hardware_info']['device']
            gpu_memory = result['hardware_info'].get('gpu_memory', 'Not available')
            peak_gpu_memory = result['hardware_info'].get('peak_gpu_memory', 'Not available')
            batch_size = result['training_info']['batch_size']
            total_epochs = len(result['epochs'])
            
            # Get start date and time
            start_time = result.get('training_info', {}).get('start_time', 'Not available')
            
            # Calculate total training time
            total_time = sum(epoch.get('time', 0) for epoch in result['epochs'])
            hours = int(total_time // 3600)
            minutes = int((total_time % 3600) // 60)
            seconds = int(total_time % 60)
            time_str = f"{hours}h {minutes}m {seconds}s"
            
            # Training info - Row 1
            ttk.Label(exp_frame, text="Architecture:", font=('Arial', 10, 'bold')).grid(row=0, column=0, padx=5, pady=2, sticky='e')
            ttk.Label(exp_frame, text=arch).grid(row=0, column=1, padx=5, pady=2, sticky='w')
            
            ttk.Label(exp_frame, text="Encoder:", font=('Arial', 10, 'bold')).grid(row=0, column=2, padx=5, pady=2, sticky='e')
            ttk.Label(exp_frame, text=encoder).grid(row=0, column=3, padx=5, pady=2, sticky='w')
            
            ttk.Label(exp_frame, text="Device:", font=('Arial', 10, 'bold')).grid(row=0, column=4, padx=5, pady=2, sticky='e')
            ttk.Label(exp_frame, text=device).grid(row=0, column=5, padx=5, pady=2, sticky='w')
            
            # Training info - Row 2
            ttk.Label(exp_frame, text="Start Time:", font=('Arial', 10, 'bold')).grid(row=1, column=0, padx=5, pady=2, sticky='e')
            ttk.Label(exp_frame, text=start_time).grid(row=1, column=1, padx=5, pady=2, sticky='w')
            
            ttk.Label(exp_frame, text="Total Time:", font=('Arial', 10, 'bold')).grid(row=1, column=2, padx=5, pady=2, sticky='e')
            ttk.Label(exp_frame, text=time_str).grid(row=1, column=3, padx=5, pady=2, sticky='w')
            
            ttk.Label(exp_frame, text="Peak GPU Memory:", font=('Arial', 10, 'bold')).grid(row=1, column=4, padx=5, pady=2, sticky='e')
            ttk.Label(exp_frame, text=peak_gpu_memory).grid(row=1, column=5, padx=5, pady=2, sticky='w')
            
            # Training info - Row 3
            ttk.Label(exp_frame, text="Image Size:", font=('Arial', 10, 'bold')).grid(row=2, column=0, padx=5, pady=2, sticky='e')
            ttk.Label(exp_frame, text=img_size).grid(row=2, column=1, padx=5, pady=2, sticky='w')
            
            ttk.Label(exp_frame, text="Batch Size:", font=('Arial', 10, 'bold')).grid(row=2, column=2, padx=5, pady=2, sticky='e')
            ttk.Label(exp_frame, text=str(batch_size)).grid(row=2, column=3, padx=5, pady=2, sticky='w')
            
            ttk.Label(exp_frame, text="GPU Memory:", font=('Arial', 10, 'bold')).grid(row=2, column=4, padx=5, pady=2, sticky='e')
            ttk.Label(exp_frame, text=gpu_memory).grid(row=2, column=5, padx=5, pady=2, sticky='w')
            
            # Dataset info - Row 4
            train_size = result['dataset_info']['train_size']
            valid_size = result['dataset_info']['valid_size']
            
            ttk.Label(exp_frame, text="Train Size:", font=('Arial', 10, 'bold')).grid(row=3, column=0, padx=5, pady=2, sticky='e')
            ttk.Label(exp_frame, text=str(train_size)).grid(row=3, column=1, padx=5, pady=2, sticky='w')
            
            ttk.Label(exp_frame, text="Valid Size:", font=('Arial', 10, 'bold')).grid(row=3, column=2, padx=5, pady=2, sticky='e')
            ttk.Label(exp_frame, text=str(valid_size)).grid(row=3, column=3, padx=5, pady=2, sticky='w')
            
            ttk.Label(exp_frame, text="Total Epochs:", font=('Arial', 10, 'bold')).grid(row=3, column=4, padx=5, pady=2, sticky='e')
            ttk.Label(exp_frame, text=str(total_epochs)).grid(row=3, column=5, padx=5, pady=2, sticky='w')
        
    def create_best_metrics_frame(self):
        """Create frame for best metrics"""
        for i, result in enumerate(self.results_list):
            # Get metrics from training_info if available
            training_info = result.get('training_info', {})
            best_metrics = training_info.get('best_metrics', {})
            best_epoch = training_info.get('best_epoch')
            
            if best_metrics:
                # Create frame
                metrics_frame = ttk.LabelFrame(self.results_frame, 
                                             text=f"Best Metrics - {result['file_name']}")
                metrics_frame.pack(fill=tk.X, padx=5, pady=5)
                
                # Display metrics
                for metric, value in best_metrics.items():
                    if metric != 'F1-Score':  # Skip F1-Score as it's redundant with Dice Score
                        label = ttk.Label(metrics_frame, 
                                        text=f"Best {metric}: {value:.4f} (Epoch {best_epoch})",
                                        font=('Arial', 10))
                        label.pack(anchor=tk.W, padx=5, pady=2)
            else:
                # Fall back to calculating from epochs
                metrics_frame = ttk.LabelFrame(self.results_frame, 
                                             text=f"Best Metrics - {result['file_name']}")
                metrics_frame.pack(fill=tk.X, padx=5, pady=5)
                
                epochs = result.get('epochs', [])
                if epochs:
                    metrics = {
                        'IoU': ('max', None, None),
                        'Precision': ('max', None, None),
                        'Recall': ('max', None, None),
                        'Dice Score': ('max', None, None),
                        'Accuracy': ('max', None, None)
                    }
                    
                    # Find best values and epochs
                    for epoch_idx, epoch in enumerate(epochs):
                        valid_metrics = epoch.get('valid_metrics', {})
                        for metric in metrics:
                            if metric in valid_metrics:
                                value = float(valid_metrics[metric])
                                compare_func = max if metrics[metric][0] == 'max' else min
                                if metrics[metric][1] is None or compare_func(value, metrics[metric][1]) == value:
                                    metrics[metric] = (metrics[metric][0], value, epoch_idx)
                    
                    # Display metrics
                    for metric, (_, value, epoch) in metrics.items():
                        if value is not None:
                            label = ttk.Label(metrics_frame, 
                                            text=f"Best {metric}: {value:.4f} (Epoch {epoch})",
                                            font=('Arial', 10))
                            label.pack(anchor=tk.W, padx=5, pady=2)
        
    def create_comparison_tabs(self):
        # Create detailed comparison tab
        compare_frame = ttk.Frame(self.notebook)
        self.notebook.add(compare_frame, text="Detailed Comparison")
        
        # Create inner notebook for comparison categories
        compare_notebook = ttk.Notebook(compare_frame)
        compare_notebook.pack(fill=tk.BOTH, expand=True)
        
        # Hardware Comparison Tab
        hw_frame = ttk.Frame(compare_notebook)
        compare_notebook.add(hw_frame, text="Hardware Metrics")
        
        # Create hardware comparison table
        hw_cols = ['Experiment', 'Peak GPU Memory', 'Total Time', 'Batch Size', 'Image Size']
        for i, col in enumerate(hw_cols):
            ttk.Label(hw_frame, text=col, font=('Arial', 10, 'bold')).grid(row=0, column=i, padx=5, pady=5, sticky='w')
        
        for i, result in enumerate(self.results_list):
            exp_name = Path(result['file_name']).stem
            peak_gpu = result['hardware_info'].get('peak_gpu_memory', 'N/A')
            total_time = sum(epoch.get('time', 0) for epoch in result['epochs'])
            time_str = f"{int(total_time//3600)}h {int((total_time%3600)//60)}m"
            batch_size = result['training_info']['batch_size']
            img_size = f"{result['training_info']['image_size']['width']}x{result['training_info']['image_size']['height']}"
            
            values = [exp_name, peak_gpu, time_str, batch_size, img_size]
            for j, val in enumerate(values):
                ttk.Label(hw_frame, text=str(val)).grid(row=i+1, column=j, padx=5, pady=2, sticky='w')
        
        # Best Epoch Comparison Tab
        epoch_frame = ttk.Frame(compare_notebook)
        compare_notebook.add(epoch_frame, text="Best Epochs")
        
        # Create epoch selection frame
        select_frame = ttk.Frame(epoch_frame)
        select_frame.pack(fill=tk.X, padx=5, pady=5)
        
        ttk.Label(select_frame, text="Compare at epoch:").pack(side=tk.LEFT, padx=5)
        epoch_var = tk.StringVar(value='best')
        epoch_combo = ttk.Combobox(select_frame, textvariable=epoch_var, values=['best'] + 
                                 [str(i) for i in range(1, len(self.results_list[0]['epochs'])+1)])
        epoch_combo.pack(side=tk.LEFT, padx=5)
        
        # Create scrollable frame for metrics
        canvas = tk.Canvas(epoch_frame)
        scrollbar = ttk.Scrollbar(epoch_frame, orient="vertical", command=canvas.yview)
        scrollable_frame = ttk.Frame(canvas)
        
        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        
        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        
        # Pack scrollbar and canvas
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        def update_epoch_comparison(*args):
            # Clear previous content
            for widget in scrollable_frame.winfo_children():
                widget.destroy()
            
            # Headers
            ttk.Label(scrollable_frame, text="Metric", font=('Arial', 10, 'bold')).grid(
                row=0, column=0, padx=5, pady=2, sticky='w')
            
            for i, result in enumerate(self.results_list):
                exp_name = Path(result['file_name']).stem
                ttk.Label(scrollable_frame, text=exp_name, font=('Arial', 10, 'bold')).grid(
                    row=0, column=i+1, padx=5, pady=2)
            
            # Metrics comparison
            metrics = ['IoU', 'Precision', 'Recall', 'Dice Score', 'Accuracy']
            for row, metric in enumerate(metrics, 1):
                ttk.Label(scrollable_frame, text=metric, font=('Arial', 10, 'bold')).grid(
                    row=row, column=0, padx=5, pady=2, sticky='w')
                
                for col, result in enumerate(self.results_list, 1):
                    if epoch_var.get() == 'best':
                        val = max(e['valid_metrics'][metric] for e in result['epochs'])
                    else:
                        epoch_idx = int(epoch_var.get()) - 1
                        val = result['epochs'][epoch_idx]['valid_metrics'][metric]
                    ttk.Label(scrollable_frame, text=f"{val:.4f}").grid(
                        row=row, column=col, padx=5, pady=2)
        
        epoch_var.trace('w', update_epoch_comparison)
        update_epoch_comparison()
        
        # Training Progress Comparison Tab
        progress_frame = ttk.Frame(compare_notebook)
        compare_notebook.add(progress_frame, text="Training Progress")
        
        # Create progress comparison plots
        fig = plt.figure(figsize=(12, 8))
        gs = fig.add_gridspec(2, 2)
        
        # Loss comparison
        ax1 = fig.add_subplot(gs[0, 0])
        ax2 = fig.add_subplot(gs[0, 1])
        ax3 = fig.add_subplot(gs[1, :])
        
        # Plot loss curves
        for result in self.results_list:
            exp_name = Path(result['file_name']).stem
            epochs = [e['epoch'] for e in result['epochs']]
            train_loss = [e['train_loss'] for e in result['epochs']]
            valid_loss = [e['valid_loss'] for e in result['epochs']]
            
            ax1.plot(epochs, train_loss, label=f'{exp_name} (Train)', linewidth=2)
            ax2.plot(epochs, valid_loss, label=f'{exp_name} (Valid)', linewidth=2)
        
        ax1.set_title('Training Loss Comparison')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.legend()
        ax1.grid(True)
        
        ax2.set_title('Validation Loss Comparison')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Loss')
        ax2.legend()
        ax2.grid(True)
        
        # Plot IoU progress
        for result in self.results_list:
            exp_name = Path(result['file_name']).stem
            epochs = [e['epoch'] for e in result['epochs']]
            iou = [e['valid_metrics']['IoU'] for e in result['epochs']]
            ax3.plot(epochs, iou, label=exp_name, linewidth=2)
        
        ax3.set_title('IoU Progress Comparison')
        ax3.set_xlabel('Epoch')
        ax3.set_ylabel('IoU')
        ax3.legend()
        ax3.grid(True)
        
        fig.tight_layout()
        canvas = FigureCanvasTkAgg(fig, progress_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
    def create_loss_plot(self):
        # Remove existing Loss tab if it exists
        for tab_id in self.notebook.tabs():
            if self.notebook.tab(tab_id, "text") == "Loss Curves":
                self.notebook.forget(tab_id)
                
        loss_frame = ttk.Frame(self.notebook)
        self.notebook.add(loss_frame, text="Loss Curves")
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        for i, result in enumerate(self.results_list):
            epochs_data = result['epochs']
            epochs = [e['epoch'] for e in epochs_data]
            train_loss = [e['train_loss'] for e in epochs_data]
            valid_loss = [e['valid_loss'] for e in epochs_data]
            
            label_suffix = f" ({result['file_name']})" if self.comparison_mode else ""
            train_color = self.plot_colors.get('Train Line', self.default_colors[i*2])
            valid_color = self.plot_colors.get('Validation Line', self.default_colors[i*2+1])
            
            if self.metric_type.get() in ['both', 'train']:
                ax.plot(epochs, train_loss, label=f'Train Loss{label_suffix}', 
                       color=train_color, linewidth=2)
            if self.metric_type.get() in ['both', 'valid']:
                ax.plot(epochs, valid_loss, label=f'Validation Loss{label_suffix}', 
                       color=valid_color, linewidth=2)
                   
        ax.set_xlabel('Epoch', fontsize=int(self.label_size.get()))
        ax.set_ylabel('Loss', fontsize=int(self.label_size.get()))
        ax.set_title(self.plot_titles.get('Loss', 'Training and Validation Loss'), fontsize=int(self.title_size.get()), pad=15)
        ax.legend(frameon=True, fancybox=True, shadow=True, fontsize=int(self.legend_size.get()))
        
        if 'Grid' in self.plot_colors:
            ax.grid(True, linestyle='--', alpha=0.7, color=self.plot_colors['Grid'])
        else:
            ax.grid(True, linestyle='--', alpha=0.7)
            
        if 'Background' in self.plot_colors:
            ax.set_facecolor(self.plot_colors['Background'])
            fig.patch.set_facecolor(self.plot_colors['Background'])
            
        canvas = FigureCanvasTkAgg(fig, loss_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
    def create_metrics_plots(self):
        metrics_frame = ttk.Frame(self.notebook)
        self.notebook.add(metrics_frame, text="Metrics")
        
        # Create a canvas with scrollbar for better responsiveness
        canvas = tk.Canvas(metrics_frame)
        scrollbar = ttk.Scrollbar(metrics_frame, orient="vertical", command=canvas.yview)
        scrollable_frame = ttk.Frame(canvas)
        
        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        
        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        
        # Pack scrollbar and canvas
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        # Create metrics plots
        fig = plt.figure(figsize=(15, 10))
        gs = fig.add_gridspec(2, 3, hspace=0.3, wspace=0.3)
        
        metrics = ['IoU', 'Precision', 'Recall', 'Dice Score', 'Accuracy']
        axes = [fig.add_subplot(gs[i//3, i%3]) for i in range(5)]
        
        for i, (metric, ax) in enumerate(zip(metrics, axes)):
            for j, result in enumerate(self.results_list):
                epochs_data = result['epochs']
                epochs = [e['epoch'] for e in epochs_data]
                
                if self.metric_type.get() in ['both', 'train']:
                    train_metric = [e['train_metrics'][metric] for e in epochs_data]
                    ax.plot(epochs, train_metric, label=self.legend_labels.get(f'{metric}_train', f'Train {metric} ({result["file_name"]})'), 
                           color=self.plot_colors.get('Train Line', self.default_colors[j*2]), linewidth=2)
                
                if self.metric_type.get() in ['both', 'valid']:
                    valid_metric = [e['valid_metrics'][metric] for e in epochs_data]
                    ax.plot(epochs, valid_metric, label=self.legend_labels.get(f'{metric}_valid', f'Valid {metric} ({result["file_name"]})'), 
                           color=self.plot_colors.get('Validation Line', self.default_colors[j*2+1]), linewidth=2)
                
            ax.set_xlabel('Epoch', fontsize=int(self.label_size.get()))
            ax.set_ylabel(metric.replace(' ', '\n'), fontsize=int(self.label_size.get()))
            ax.set_title(self.plot_titles.get(metric, f'{metric} over Epochs'), fontsize=int(self.title_size.get()), pad=10)
            ax.legend(frameon=True, fancybox=True, shadow=True, fontsize=int(self.legend_size.get()))
            
            if 'Grid' in self.plot_colors:
                ax.grid(True, linestyle='--', alpha=0.7, color=self.plot_colors['Grid'])
            else:
                ax.grid(True, linestyle='--', alpha=0.7)
                
            if 'Background' in self.plot_colors:
                ax.set_facecolor(self.plot_colors['Background'])
                fig.patch.set_facecolor(self.plot_colors['Background'])
        
        fig.tight_layout()
        canvas_plot = FigureCanvasTkAgg(fig, scrollable_frame)
        canvas_plot.draw()
        canvas_plot.get_tk_widget().pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
    def on_plot_select(self, event=None):
        """Update entry fields when a plot is selected"""
        selected_plot = self.plot_select.get()
        if selected_plot:
            self.title_var.set(self.plot_titles.get(selected_plot, ''))
            self.legend_var.set(self.legend_labels.get(selected_plot, ''))
    
    def update_plot_title(self):
        """Update the title for the selected plot"""
        selected_plot = self.plot_select.get()
        if selected_plot and self.title_var.get():
            self.plot_titles[selected_plot] = self.title_var.get()
            self.save_settings()
            if hasattr(self, 'notebook') and self.notebook:
                self.update_plots()
    
    def update_legend_label(self):
        """Update the legend label for the selected plot"""
        selected_plot = self.plot_select.get()
        if selected_plot and self.legend_var.get():
            self.legend_labels[selected_plot] = self.legend_var.get()
            self.save_settings()
            if hasattr(self, 'notebook') and self.notebook:
                self.update_plots()
                
if __name__ == "__main__":
    root = tk.Tk()
    app = ResultsVisualizer(root)
    root.mainloop()
