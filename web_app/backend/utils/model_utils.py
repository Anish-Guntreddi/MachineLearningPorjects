"""
Model Utility Functions
Helper functions for model operations and metrics
"""

import torch
from typing import Dict, Any, List
import platform


def get_device_info() -> Dict[str, Any]:
    """
    Get information about available compute devices

    Returns:
        Dictionary with device information
    """
    info = {
        'platform': platform.system(),
        'python_version': platform.python_version(),
        'pytorch_version': torch.__version__,
        'cuda_available': torch.cuda.is_available(),
    }

    if torch.cuda.is_available():
        info['cuda_version'] = torch.version.cuda
        info['cudnn_version'] = torch.backends.cudnn.version()
        info['gpu_count'] = torch.cuda.device_count()
        info['gpus'] = []

        for i in range(torch.cuda.device_count()):
            gpu_info = {
                'index': i,
                'name': torch.cuda.get_device_name(i),
                'memory_total_gb': torch.cuda.get_device_properties(i).total_memory / (1024**3),
                'memory_allocated_gb': torch.cuda.memory_allocated(i) / (1024**3),
                'memory_cached_gb': torch.cuda.memory_reserved(i) / (1024**3),
            }
            info['gpus'].append(gpu_info)

        info['current_device'] = torch.cuda.current_device()
    else:
        info['cpu_count'] = torch.get_num_threads()

    return info


def format_metrics(metrics: Dict[str, Any]) -> Dict[str, Any]:
    """
    Format metrics for API response

    Args:
        metrics: Raw metrics dictionary

    Returns:
        Formatted metrics dictionary
    """
    formatted = {}

    # Round numeric values
    for key, value in metrics.items():
        if isinstance(value, (int, float)):
            if isinstance(value, float):
                formatted[key] = round(value, 4)
            else:
                formatted[key] = value
        elif isinstance(value, dict):
            formatted[key] = format_metrics(value)
        elif isinstance(value, list):
            formatted[key] = [
                format_metrics(item) if isinstance(item, dict) else item
                for item in value
            ]
        else:
            formatted[key] = value

    return formatted


def calculate_model_size(model: torch.nn.Module) -> Dict[str, int]:
    """
    Calculate model size information

    Args:
        model: PyTorch model

    Returns:
        Dictionary with model size metrics
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    non_trainable_params = total_params - trainable_params

    # Calculate model size in bytes
    param_size = sum(p.numel() * p.element_size() for p in model.parameters())
    buffer_size = sum(b.numel() * b.element_size() for b in model.buffers())
    total_size = param_size + buffer_size

    return {
        'total_parameters': total_params,
        'trainable_parameters': trainable_params,
        'non_trainable_parameters': non_trainable_params,
        'size_mb': total_size / (1024**2),
        'parameter_size_mb': param_size / (1024**2),
        'buffer_size_mb': buffer_size / (1024**2),
    }


def validate_model_output(output: Any, expected_keys: List[str] = None) -> bool:
    """
    Validate model output format

    Args:
        output: Model output to validate
        expected_keys: List of expected keys if output is a dict

    Returns:
        True if valid, False otherwise
    """
    if expected_keys is not None:
        if not isinstance(output, dict):
            return False

        for key in expected_keys:
            if key not in output:
                return False

    return True


def get_top_k_predictions(
    probabilities: torch.Tensor,
    class_names: List[str],
    k: int = 5
) -> List[Dict[str, Any]]:
    """
    Get top K predictions from model output

    Args:
        probabilities: Tensor of class probabilities
        class_names: List of class names
        k: Number of top predictions to return

    Returns:
        List of dictionaries with class and confidence
    """
    top_k = min(k, len(class_names))
    top_probs, top_indices = torch.topk(probabilities, top_k)

    predictions = []
    for prob, idx in zip(top_probs, top_indices):
        predictions.append({
            'class': class_names[idx.item()],
            'confidence': prob.item(),
            'index': idx.item()
        })

    return predictions


def batch_tensor(data: List[torch.Tensor], max_batch_size: int = 32) -> List[torch.Tensor]:
    """
    Split data into batches

    Args:
        data: List of tensors to batch
        max_batch_size: Maximum batch size

    Returns:
        List of batched tensors
    """
    batches = []

    for i in range(0, len(data), max_batch_size):
        batch = data[i:i + max_batch_size]
        batches.append(torch.stack(batch))

    return batches


def move_to_device(data: Any, device: torch.device) -> Any:
    """
    Recursively move data to device

    Args:
        data: Data to move (tensor, dict, list, etc.)
        device: Target device

    Returns:
        Data on the target device
    """
    if isinstance(data, torch.Tensor):
        return data.to(device)
    elif isinstance(data, dict):
        return {key: move_to_device(value, device) for key, value in data.items()}
    elif isinstance(data, list):
        return [move_to_device(item, device) for item in data]
    elif isinstance(data, tuple):
        return tuple(move_to_device(item, device) for item in data)
    else:
        return data


def clear_gpu_cache():
    """
    Clear GPU cache to free up memory
    """
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
