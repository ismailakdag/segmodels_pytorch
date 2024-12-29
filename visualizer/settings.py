import tkinter as tk
from tkinter import ttk, colorchooser
import json

class VisualizerSettings:
    def __init__(self, root, parent):
        self.root = root
        self.parent = parent
        self.load_settings()

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

    def show_settings_dialog(self):
        """Show settings dialog"""
        settings_window = tk.Toplevel(self.root)
        settings_window.title("Settings")
        settings_window.geometry("600x500")
        
        settings_frame = ttk.Frame(settings_window)
        settings_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Metric type selection
        self._create_metric_selection(settings_frame)
        
        # Plot customization
        self._create_plot_customization(settings_frame)
        
        # Color settings
        self._create_color_settings(settings_frame)
        
        # Font settings
        self._create_font_settings(settings_frame)
        
        # Apply and Close buttons
        self._create_buttons(settings_frame, settings_window)

    def _create_metric_selection(self, parent):
        type_frame = ttk.LabelFrame(parent, text="Metric Display Options")
        type_frame.pack(fill=tk.X, pady=(0, 10))
        
        ttk.Label(type_frame, text="Default Metrics:").pack(side=tk.LEFT, padx=5, pady=5)
        for value, text in [('valid', 'Valid Only'), ('train', 'Train Only'), ('both', 'Both')]:
            ttk.Radiobutton(type_frame, text=text, variable=self.metric_type, 
                           value=value).pack(side=tk.LEFT, padx=5)

    def _create_plot_customization(self, parent):
        plot_frame = ttk.LabelFrame(parent, text="Plot Customization")
        plot_frame.pack(fill=tk.X, pady=10)
        
        # Plot selection
        select_frame = ttk.Frame(plot_frame)
        select_frame.pack(fill=tk.X, padx=5, pady=5)
        
        ttk.Label(select_frame, text="Select Plot:").pack(side=tk.LEFT, padx=5)
        self.plot_select = ttk.Combobox(select_frame, 
            values=['Loss', 'IoU', 'Precision', 'Recall', 'Dice Score', 'Accuracy'])
        self.plot_select.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
        self.plot_select.bind('<<ComboboxSelected>>', self.parent.on_plot_select)
        
        # Title and Legend customization
        for label, var, cmd in [("Default Title:", "title_var", "update_plot_title"),
                              ("Default Legend:", "legend_var", "update_legend_label")]:
            frame = ttk.Frame(plot_frame)
            frame.pack(fill=tk.X, padx=5, pady=5)
            ttk.Label(frame, text=label).pack(side=tk.LEFT, padx=5)
            entry_var = tk.StringVar()
            setattr(self, var, entry_var)
            ttk.Entry(frame, textvariable=entry_var).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
            ttk.Button(frame, text="Set", 
                      command=getattr(self.parent, cmd)).pack(side=tk.LEFT, padx=5)

    def _create_color_settings(self, parent):
        color_frame = ttk.LabelFrame(parent, text="Color Settings")
        color_frame.pack(fill=tk.X, pady=10)
        
        def choose_color(element):
            color = colorchooser.askcolor(
                color=self.plot_colors.get(element),
                title=f"Choose color for {element}")[1]
            if color:
                self.plot_colors[element] = color
        
        for element in ['Train Line', 'Validation Line', 'Grid', 'Background']:
            frame = ttk.Frame(color_frame)
            frame.pack(fill=tk.X, padx=5, pady=2)
            ttk.Label(frame, text=f"{element}:").pack(side=tk.LEFT, padx=5)
            ttk.Button(frame, text="Select Color", 
                      command=lambda e=element: choose_color(e)).pack(side=tk.LEFT, padx=5)

    def _create_font_settings(self, parent):
        font_frame = ttk.LabelFrame(parent, text="Font Settings")
        font_frame.pack(fill=tk.X, pady=10)
        
        sizes_frame = ttk.Frame(font_frame)
        sizes_frame.pack(fill=tk.X, padx=5, pady=5)
        
        for i, (text, var) in enumerate([("Title Size:", self.title_size),
                                       ("Label Size:", self.label_size),
                                       ("Legend Size:", self.legend_size)]):
            ttk.Label(sizes_frame, text=text).grid(row=0, column=i*2, padx=5, pady=2)
            ttk.Entry(sizes_frame, textvariable=var, width=5).grid(row=0, column=i*2+1, padx=5, pady=2)

    def _create_buttons(self, parent, window):
        button_frame = ttk.Frame(parent)
        button_frame.pack(fill=tk.X, pady=10)
        
        def apply_settings():
            self.save_settings()
            if hasattr(self.parent, 'notebook') and self.parent.notebook:
                # Remove existing tabs
                for tab_id in self.parent.notebook.tabs():
                    self.parent.notebook.forget(tab_id)
                
                # Recreate plots
                self.parent.create_loss_plot()
                self.parent.create_metrics_plots()
                
                # Update comparison tabs if in comparison mode
                if self.parent.comparison_mode:
                    self.parent.create_comparison_tabs()
        
        ttk.Button(button_frame, text="Apply", command=apply_settings).pack(side=tk.RIGHT, padx=5)
        ttk.Button(button_frame, text="Close", command=window.destroy).pack(side=tk.RIGHT, padx=5)
