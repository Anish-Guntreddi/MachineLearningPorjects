"""
Utility functions for instance segmentation
"""
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import Rectangle, Polygon
import cv2
import os
import shutil
from typing import Dict, List, Tuple, Optional
from collections import defaultdict
from pycocotools import mask as maskUtils


class AverageMeter:
    """Computes and stores the average and current value"""
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
    
    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def save_checkpoint(state: dict, is_best: bool, checkpoint_dir: str = './checkpoints'):
    """Save model checkpoint"""
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    filename = os.path.join(checkpoint_dir, 'checkpoint.pth')
    torch.save(state, filename)
    
    if is_best:
        best_filename = os.path.join(checkpoint_dir, 'best_model.pth')
        shutil.copyfile(filename, best_filename)


def load_checkpoint(checkpoint_path: str, model: nn.Module, optimizer: Optional = None):
    """Load model checkpoint"""
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    model.load_state_dict(checkpoint['model_state_dict'])
    
    if optimizer and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    return checkpoint


def calculate_mask_iou(mask1: torch.Tensor, mask2: torch.Tensor) -> float:
    """
    Calculate IoU between two binary masks
    
    Args:
        mask1: Binary mask tensor
        mask2: Binary mask tensor
    
    Returns:
        IoU value
    """
    intersection = (mask1 & mask2).float().sum()
    union = (mask1 | mask2).float().sum()
    
    if union == 0:
        return 0.0
    
    return (intersection / union).item()


def evaluate_segmentation(
    predictions: List[Dict],
    targets: List[Dict],
    iou_threshold: float = 0.5
) -> Dict:
    """
    Evaluate instance segmentation performance
    
    Args:
        predictions: List of prediction dictionaries
        targets: List of target dictionaries
        iou_threshold: IoU threshold for matching
    
    Returns:
        Dictionary with evaluation metrics
    """
    metrics = {
        'mask_map': 0.0,
        'box_map': 0.0,
        'mask_map_50': 0.0,
        'mask_map_75': 0.0
    }
    
    # Simplified evaluation - in practice use COCO evaluation
    total_mask_iou = 0
    total_box_iou = 0
    count = 0
    
    for pred, target in zip(predictions, targets):
        if 'masks' in pred and 'masks' in target:
            # Match predictions to targets
            pred_masks = pred['masks']
            target_masks = target['masks']
            
            if len(pred_masks) > 0 and len(target_masks) > 0:
                # Calculate IoU for each pair
                for pm in pred_masks:
                    best_iou = 0
                    for tm in target_masks:
                        iou = calculate_mask_iou(pm > 0.5, tm > 0.5)
                        best_iou = max(best_iou, iou)
                    total_mask_iou += best_iou
                    count += 1
        
        if 'boxes' in pred and 'boxes' in target:
            # Similar for boxes
            pred_boxes = pred['boxes']
            target_boxes = target['boxes']
            
            if len(pred_boxes) > 0 and len(target_boxes) > 0:
                for pb in pred_boxes:
                    best_iou = 0
                    for tb in target_boxes:
                        iou = calculate_box_iou(pb, tb)
                        best_iou = max(best_iou, iou)
                    total_box_iou += best_iou
    
    if count > 0:
        metrics['mask_map'] = total_mask_iou / count
        metrics['box_map'] = total_box_iou / count
        metrics['mask_map_50'] = metrics['mask_map']  # Simplified
        metrics['mask_map_75'] = metrics['mask_map'] * 0.8  # Simplified
    
    return metrics


def calculate_box_iou(box1: torch.Tensor, box2: torch.Tensor) -> float:
    """Calculate IoU between two bounding boxes"""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    
    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = box1_area + box2_area - intersection
    
    return intersection / (union + 1e-6)


def visualize_instance_masks(
    image: torch.Tensor,
    predictions: Dict,
    targets: Optional[Dict] = None,
    class_names: Optional[List[str]] = None,
    save_path: Optional[str] = None,
    score_threshold: float = 0.5
):
    """
    Visualize instance segmentation masks
    
    Args:
        image: Image tensor (C, H, W)
        predictions: Prediction dictionary with masks
        targets: Optional target dictionary
        class_names: List of class names
        save_path: Path to save visualization
        score_threshold: Score threshold for displaying
    """
    if class_names is None:
        class_names = [f'Class {i}' for i in range(100)]
    
    # Convert image to numpy
    if isinstance(image, torch.Tensor):
        image = image.permute(1, 2, 0).cpu().numpy()
        if image.max() <= 1:
            image = (image * 255).astype(np.uint8)
    
    fig, axes = plt.subplots(1, 2 if targets else 1, figsize=(15, 8))
    
    if not isinstance(axes, np.ndarray):
        axes = [axes]
    
    # Generate colors for each instance
    num_instances = len(predictions.get('masks', []))
    colors = plt.cm.rainbow(np.linspace(0, 1, max(num_instances, 1)))
    
    # Plot predictions
    ax = axes[0]
    ax.imshow(image)
    ax.set_title('Predictions')
    ax.axis('off')
    
    if 'masks' in predictions and len(predictions['masks']) > 0:
        masks = predictions['masks'].cpu().numpy()
        boxes = predictions.get('boxes', torch.zeros(len(masks), 4)).cpu().numpy()
        labels = predictions.get('labels', torch.zeros(len(masks))).cpu().numpy()
        scores = predictions.get('scores', torch.ones(len(masks))).cpu().numpy()
        
        # Create overlay
        overlay = np.zeros_like(image)
        
        for i, (mask, box, label, score) in enumerate(zip(masks, boxes, labels, scores)):
            if score >= score_threshold:
                # Process mask
                if len(mask.shape) == 3:
                    mask = mask[0]
                mask = mask > 0.5
                
                # Apply color to mask
                color = (colors[i][:3] * 255).astype(np.uint8)
                overlay[mask] = color
                
                # Draw bounding box
                x1, y1, x2, y2 = box.astype(int)
                rect = Rectangle(
                    (x1, y1), x2 - x1, y2 - y1,
                    linewidth=2, edgecolor=colors[i], facecolor='none'
                )
                ax.add_patch(rect)
                
                # Add label
                label_text = f'{class_names[int(label)]}: {score:.2f}'
                ax.text(
                    x1, y1 - 5, label_text,
                    color='white', fontsize=8,
                    bbox=dict(facecolor=colors[i], alpha=0.7)
                )
        
        # Blend overlay with image
        alpha = 0.5
        blended = cv2.addWeighted(image, 1 - alpha, overlay, alpha, 0)
        ax.imshow(blended)
    
    # Plot targets if provided
    if targets:
        ax = axes[1]
        ax.imshow(image)
        ax.set_title('Ground Truth')
        ax.axis('off')
        
        if 'masks' in targets and len(targets['masks']) > 0:
            masks = targets['masks'].cpu().numpy()
            boxes = targets.get('boxes', torch.zeros(len(masks), 4)).cpu().numpy()
            labels = targets.get('labels', torch.zeros(len(masks))).cpu().numpy()
            
            overlay = np.zeros_like(image)
            
            for i, (mask, box, label) in enumerate(zip(masks, boxes, labels)):
                if len(mask.shape) == 3:
                    mask = mask[0]
                mask = mask > 0.5
                
                color = (colors[i % len(colors)][:3] * 255).astype(np.uint8)
                overlay[mask] = color
                
                x1, y1, x2, y2 = box.astype(int)
                rect = Rectangle(
                    (x1, y1), x2 - x1, y2 - y1,
                    linewidth=2, edgecolor=colors[i % len(colors)], facecolor='none'
                )
                ax.add_patch(rect)
                
                ax.text(
                    x1, y1 - 5, class_names[int(label)],
                    color='white', fontsize=8,
                    bbox=dict(facecolor=colors[i % len(colors)], alpha=0.7)
                )
            
            blended = cv2.addWeighted(image, 1 - alpha, overlay, alpha, 0)
            ax.imshow(blended)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=100, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def draw_instance_masks(
    image: np.ndarray,
    masks: np.ndarray,
    boxes: np.ndarray,
    labels: np.ndarray,
    scores: Optional[np.ndarray] = None,
    class_names: Optional[List[str]] = None,
    score_threshold: float = 0.5
) -> np.ndarray:
    """
    Draw instance masks on image
    
    Args:
        image: Image array (H, W, C)
        masks: Instance masks (N, H, W)
        boxes: Bounding boxes (N, 4)
        labels: Class labels (N,)
        scores: Confidence scores (N,)
        class_names: List of class names
        score_threshold: Score threshold
    
    Returns:
        Image with drawn masks
    """
    image = image.copy()
    
    if class_names is None:
        class_names = [f'Class {i}' for i in range(100)]
    
    # Generate colors
    colors = plt.cm.rainbow(np.linspace(0, 1, len(masks)))[:, :3] * 255
    
    # Create overlay
    overlay = np.zeros_like(image)
    
    for i, (mask, box, label) in enumerate(zip(masks, boxes, labels)):
        if scores is not None and scores[i] < score_threshold:
            continue
        
        # Apply mask
        mask = mask > 0.5
        color = colors[i % len(colors)].astype(np.uint8)
        overlay[mask] = color
        
        # Draw box
        x1, y1, x2, y2 = box.astype(int)
        cv2.rectangle(image, (x1, y1), (x2, y2), color.tolist(), 2)
        
        # Draw label
        label_text = class_names[label]
        if scores is not None:
            label_text += f': {scores[i]:.2f}'
        
        cv2.putText(
            image, label_text, (x1, y1 - 5),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1
        )
    
    # Blend overlay
    result = cv2.addWeighted(image, 0.6, overlay, 0.4, 0)
    
    return result


def export_to_onnx(
    model: nn.Module,
    input_shape: Tuple[int, int],
    output_path: str,
    device: torch.device
):
    """Export model to ONNX format"""
    model.eval()
    
    # Create dummy input
    dummy_input = [torch.randn(3, input_shape[0], input_shape[1]).to(device)]
    
    # Export
    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        export_params=True,
        opset_version=11,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['boxes', 'labels', 'scores', 'masks'],
        dynamic_axes={
            'boxes': {0: 'num_detections'},
            'labels': {0: 'num_detections'},
            'scores': {0: 'num_detections'},
            'masks': {0: 'num_detections'}
        }
    )
    
    print(f"Model exported to {output_path}")


if __name__ == "__main__":
    # Test utilities
    mask1 = torch.rand(100, 100) > 0.5
    mask2 = torch.rand(100, 100) > 0.5
    
    iou = calculate_mask_iou(mask1, mask2)
    print(f"Mask IoU: {iou:.4f}")