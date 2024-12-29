import json
from pathlib import Path
import numpy as np

def load_results(file_path):
    """Load results from a JSON file"""
    try:
        with open(file_path, 'r') as f:
            return json.load(f)
    except Exception as e:
        raise Exception(f"Failed to load results: {str(e)}")

def get_best_metrics(results):
    """Get best metrics from results"""
    best_metrics = {}
    for result in results:
        epochs_data = result['epochs']
        metrics = []
        for epoch in epochs_data:
            metrics.append({
                'epoch': epoch['epoch'],
                'valid_loss': epoch['valid_loss'],
                **epoch['valid_metrics']
            })
        
        best_epoch = min(metrics, key=lambda x: x['valid_loss'])
        best_metrics[Path(result['file_name']).stem] = best_epoch
        
    return best_metrics

def format_experiment_info(result):
    """Format experiment information for display"""
    info = []
    
    # Dataset Info
    if 'dataset_info' in result:
        ds_info = result['dataset_info']
        info.append(("Dataset Info", [
            f"Train Size: {ds_info.get('train_size')}",
            f"Valid Size: {ds_info.get('valid_size')}",
            f"Batch Size: {ds_info.get('batch_size')}",
            f"Image Size: {ds_info.get('image_size', {}).get('width', 'N/A')}x{ds_info.get('image_size', {}).get('height', 'N/A')}"
        ]))
    
    # Hardware Info
    if 'hardware_info' in result:
        hw_info = result['hardware_info']
        gpu_info = []
        if 'device' in hw_info:
            gpu_info.append(f"Device: {hw_info['device']}")
        if 'gpu_memory' in hw_info:
            gpu_info.append(f"Available GPU Memory: {hw_info['gpu_memory']}")
        if 'peak_gpu_memory' in hw_info:
            gpu_info.append(f"Peak GPU Memory: {hw_info['peak_gpu_memory']}")
        if gpu_info:
            info.append(("Hardware Info", gpu_info))
    
    # Training Info
    if 'training_info' in result:
        train_info = result['training_info']
        info.append(("Training Info", [
            f"Architecture: {train_info.get('architecture')}",
            f"Encoder: {train_info.get('encoder')}",
            f"Pretrained: {train_info.get('pretrained_weights')}",
            f"Initial Epochs: {train_info.get('initial_epochs')}"
        ]))
    
    # Time Info
    time_info = []
    
    # Start and end time
    if 'start_time' in result:
        time_info.append(f"Start Time: {result['start_time']}")
        if 'end_time' in result:
            time_info.append(f"End Time: {result['end_time']}")
            from datetime import datetime
            try:
                start = datetime.strptime(result['start_time'], "%Y-%m-%d %H:%M:%S")
                end = datetime.strptime(result['end_time'], "%Y-%m-%d %H:%M:%S")
                duration = end - start
                hours = duration.total_seconds() / 3600
                time_info.append(f"Total Time: {int(hours)}h {int((hours % 1) * 60)}m {int(((hours * 60) % 1) * 60)}s")
            except ValueError:
                pass
    
    # Training time
    if 'time' in result:
        total_time = float(result['time'])
        hours = total_time / 3600
        time_info.append(f"Training Time: {int(hours)}h {int((hours % 1) * 60)}m {int(((hours * 60) % 1) * 60)}s")
    
    # Epoch times
    if 'epochs' in result:
        epochs = result['epochs']
        epoch_times = [float(epoch.get('epoch_time', 0)) for epoch in epochs if epoch.get('epoch_time')]
        if epoch_times:
            avg_time = np.mean(epoch_times)
            total_time = sum(epoch_times)
            time_info.extend([
                f"Average Epoch Time: {int(avg_time)}s",
                f"Total Epoch Time: {int(total_time/3600)}h {int((total_time/3600 % 1) * 60)}m {int(((total_time/3600 * 60) % 1) * 60)}s"
            ])
    
    if time_info:
        info.append(("Time Info", time_info))
    
    return info
