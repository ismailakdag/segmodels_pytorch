import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import seaborn as sns

class Plotter:
    def __init__(self):
        plt.style.use('seaborn-v0_8-darkgrid')
        self.default_colors = sns.color_palette("husl", n_colors=20)
        plt.rcParams.update({
            'font.size': 12,
            'axes.labelsize': 14,
            'axes.titlesize': 16,
            'xtick.labelsize': 12,
            'ytick.labelsize': 12,
            'legend.fontsize': 12,
            'font.family': 'sans-serif',
            'font.sans-serif': ['Arial', 'DejaVu Sans'],
            'figure.titlesize': 16,
            'lines.linewidth': 2.5,
            'grid.linewidth': 0.8,
            'grid.alpha': 0.3
        })

    def _get_colors(self, result_idx, comparison_mode):
        """Get train and validation colors based on result index and mode"""
        if comparison_mode:
            train_color = self.default_colors[result_idx * 2]
            valid_color = self.default_colors[result_idx * 2 + 1]
        else:
            train_color = '#2ecc71'  # A nice green color
            valid_color = '#e74c3c'  # A nice red color
        return train_color, valid_color

    def plot_loss(self, fig, data, settings, comparison_mode=False):
        """Plot loss curves"""
        if fig is None:
            fig = plt.figure(figsize=(10, 6))
            
        ax = fig.add_subplot(111)
        self.plot_colors = settings.plot_colors
        
        for i, result in enumerate(data):
            epochs_data = result['epochs']
            epochs = [e['epoch'] for e in epochs_data]
            label_suffix = f" ({result['file_name']})" if comparison_mode else ""
            
            train_color, valid_color = self._get_colors(i, comparison_mode)
            
            if settings.metric_type.get() in ['both', 'train']:
                train_loss = [e['train_loss'] for e in epochs_data]
                ax.plot(epochs, train_loss, label=f'Train Loss{label_suffix}', 
                       color=train_color, linewidth=2.5)
                
            if settings.metric_type.get() in ['both', 'valid']:
                valid_loss = [e['valid_loss'] for e in epochs_data]
                ax.plot(epochs, valid_loss, label=f'Validation Loss{label_suffix}', 
                       color=valid_color, linewidth=2.5)
        
        self._style_plot(fig, ax, 'Loss', settings)
        
        # Adjust layout to prevent cutoff
        fig.tight_layout()
        return fig

    def plot_metrics(self, fig, data, settings, comparison_mode=False):
        """Plot all metrics"""
        if fig is None:
            fig = plt.figure(figsize=(15, 10))
            
        gs = fig.add_gridspec(2, 3, hspace=0.4, wspace=0.3)
        metrics = ['IoU', 'Precision', 'Recall', 'Dice Score', 'Accuracy']
        self.plot_colors = settings.plot_colors
        
        for i, metric in enumerate(metrics):
            ax = fig.add_subplot(gs[i//3, i%3])
            
            for j, result in enumerate(data):
                epochs_data = result['epochs']
                epochs = [e['epoch'] for e in epochs_data]
                label_suffix = f" ({result['file_name']})" if comparison_mode else ""
                
                train_color, valid_color = self._get_colors(j, comparison_mode)
                
                if settings.metric_type.get() in ['both', 'train']:
                    train_metric = [e['train_metrics'][metric] for e in epochs_data]
                    ax.plot(epochs, train_metric, 
                           label=settings.legend_labels.get(f'{metric}_train', f'Train {metric}{label_suffix}'),
                           color=train_color, linewidth=2.5)
                
                if settings.metric_type.get() in ['both', 'valid']:
                    valid_metric = [e['valid_metrics'][metric] for e in epochs_data]
                    ax.plot(epochs, valid_metric, 
                           label=settings.legend_labels.get(f'{metric}_valid', f'Valid {metric}{label_suffix}'),
                           color=valid_color, linewidth=2.5)
            
            self._style_plot(fig, ax, metric, settings)
        
        # Adjust layout to prevent cutoff
        fig.tight_layout(rect=[0, 0, 0.9, 1])  # Leave space for legend
        return fig

    def plot_single_metric(self, fig, data, settings, metric, comparison_mode=False):
        """Plot a single metric"""
        if fig is None:
            fig = plt.figure(figsize=(12, 8))
            
        ax = fig.add_subplot(111)
        self.plot_colors = settings.plot_colors
        
        for i, result in enumerate(data):
            epochs_data = result['epochs']
            epochs = [e['epoch'] for e in epochs_data]
            label_suffix = f" ({result['file_name']})" if comparison_mode else ""
            
            train_color, valid_color = self._get_colors(i, comparison_mode)
            
            if settings.metric_type.get() in ['both', 'train']:
                train_metric = [e['train_metrics'][metric] for e in epochs_data]
                ax.plot(epochs, train_metric, 
                       label=settings.legend_labels.get(f'{metric}_train', f'Train {metric}{label_suffix}'),
                       color=train_color, linewidth=2.5)
            
            if settings.metric_type.get() in ['both', 'valid']:
                valid_metric = [e['valid_metrics'][metric] for e in epochs_data]
                ax.plot(epochs, valid_metric, 
                       label=settings.legend_labels.get(f'{metric}_valid', f'Valid {metric}{label_suffix}'),
                       color=valid_color, linewidth=2.5)
        
        self._style_plot(fig, ax, metric, settings)
        
        # Adjust layout to prevent cutoff
        fig.tight_layout()
        return fig

    def _style_plot(self, fig, ax, metric, settings):
        """Apply common styling to plots"""
        ax.set_xlabel('Epoch', fontsize=14, labelpad=10)
        ax.set_ylabel(metric, fontsize=14, labelpad=10)
        ax.set_title(settings.plot_titles.get(metric, f'{metric} over Epochs'), 
                    fontsize=16, pad=15)
        
        # Improve legend
        ax.legend(frameon=True, fancybox=True, shadow=True, fontsize=12,
                 loc='best')
        
        # Improve grid
        ax.grid(True, linestyle='--', alpha=0.3)
        
        # Improve ticks
        ax.tick_params(axis='both', which='major', labelsize=12)
        
        # Add margins to prevent cutoff
        ax.margins(x=0.02)
        
        # Set background color
        if 'Background' in settings.plot_colors:
            ax.set_facecolor(settings.plot_colors['Background'])
            fig.patch.set_facecolor(settings.plot_colors['Background'])
