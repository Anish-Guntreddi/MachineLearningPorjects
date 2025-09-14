"""
Utility functions for object detection
"""
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import Rectangle
import cv2
import os
import shutil
from typing import Dict, List, Tuple, Optional
from collections import defaultdict


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


def calculate_iou(box1: torch.Tensor, box2: torch.Tensor) -> float:
    """
    Calculate IoU between two bounding boxes
    
    Args:
        box1: Tensor of shape (4,) in format [x1, y1, x2, y2]
        box2: Tensor of shape (4,) in format [x1, y1, x2, y2]
    
    Returns:
        IoU value
    """
    # Calculate intersection area
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    
    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    
    # Calculate union area
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = box1_area + box2_area - intersection
    
    # Calculate IoU
    iou = intersection / (union + 1e-6)
    
    return iou


def calculate_map(
    predictions: List[Dict],
    targets: List[Dict],
    iou_threshold: float = 0.5
) -> float:
    """
    Calculate mean Average Precision (mAP)
    
    Args:
        predictions: List of prediction dictionaries
        targets: List of target dictionaries
        iou_threshold: IoU threshold for matching
    
    Returns:
        mAP value
    """
    # Simplified mAP calculation
    # In practice, use COCO evaluation tools for accurate metrics
    
    total_ap = 0
    num_classes = 0
    
    # Group by class
    class_predictions = defaultdict(list)
    class_targets = defaultdict(list)
    
    for pred, target in zip(predictions, targets):
        if 'labels' in pred and 'labels' in target:
            for label, box, score in zip(pred['labels'], pred['boxes'], pred.get('scores', [1.0] * len(pred['labels']))):
                class_predictions[label.item()].append((box, score))
            
            for label, box in zip(target['labels'], target['boxes']):
                class_targets[label.item()].append(box)
    
    # Calculate AP for each class
    for class_id in set(list(class_predictions.keys()) + list(class_targets.keys())):
        if class_id in class_targets and len(class_targets[class_id]) > 0:
            # Sort predictions by score
            preds = sorted(class_predictions[class_id], key=lambda x: x[1], reverse=True)
            targets_boxes = class_targets[class_id]
            
            # Calculate precision and recall
            tp = 0
            fp = 0
            matched = set()
            
            for pred_box, score in preds:
                best_iou = 0
                best_idx = -1
                
                for idx, target_box in enumerate(targets_boxes):
                    if idx not in matched:
                        iou = calculate_iou(pred_box, target_box)
                        if iou > best_iou:
                            best_iou = iou
                            best_idx = idx
                
                if best_iou >= iou_threshold:
                    tp += 1
                    matched.add(best_idx)
                else:
                    fp += 1
            
            # Calculate AP (simplified)
            if tp > 0:
                precision = tp / (tp + fp)
                recall = tp / len(targets_boxes)
                ap = precision * recall  # Simplified AP
            else:
                ap = 0
            
            total_ap += ap
            num_classes += 1
    
    # Calculate mAP
    map_value = total_ap / max(num_classes, 1)
    
    return map_value


def evaluate_detection(
    predictions: List[Dict],
    targets: List[Dict]
) -> Dict:
    """
    Evaluate object detection performance
    
    Args:
        predictions: List of prediction dictionaries
        targets: List of target dictionaries
    
    Returns:
        Dictionary with evaluation metrics
    """
    metrics = {}
    
    # Calculate mAP at different IoU thresholds
    metrics['map_50'] = calculate_map(predictions, targets, iou_threshold=0.5)
    metrics['map_75'] = calculate_map(predictions, targets, iou_threshold=0.75)
    
    # Calculate average mAP from 0.5 to 0.95
    map_values = []
    for iou_thresh in np.arange(0.5, 1.0, 0.05):
        map_values.append(calculate_map(predictions, targets, iou_threshold=iou_thresh))
    metrics['map'] = np.mean(map_values)
    
    return metrics


def non_max_suppression(
    boxes: torch.Tensor,
    scores: torch.Tensor,
    labels: torch.Tensor,
    iou_threshold: float = 0.5
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Apply Non-Maximum Suppression
    
    Args:
        boxes: Bounding boxes, shape (N, 4)
        scores: Confidence scores, shape (N,)
        labels: Class labels, shape (N,)
        iou_threshold: IoU threshold for NMS
    
    Returns:
        Filtered boxes, scores, and labels
    """
    if len(boxes) == 0:
        return boxes, scores, labels
    
    # Apply NMS per class
    keep_masks = []
    
    for class_id in torch.unique(labels):
        class_mask = labels == class_id
        class_boxes = boxes[class_mask]
        class_scores = scores[class_mask]
        
        if len(class_boxes) > 0:
            # Apply NMS
            keep_indices = torchvision.ops.nms(class_boxes, class_scores, iou_threshold)
            
            # Convert back to original indices
            original_indices = torch.where(class_mask)[0]
            keep_masks.extend(original_indices[keep_indices].tolist())
    
    keep_masks = torch.tensor(keep_masks, dtype=torch.long)
    
    return boxes[keep_masks], scores[keep_masks], labels[keep_masks]


def visualize_predictions(
    image: torch.Tensor,
    predictions: Dict,
    targets: Optional[Dict] = None,
    class_names: Optional[List[str]] = None,
    save_path: Optional[str] = None,
    score_threshold: float = 0.5
):
    """
    Visualize detection predictions
    
    Args:
        image: Image tensor (C, H, W)
        predictions: Prediction dictionary
        targets: Optional target dictionary for comparison
        class_names: List of class names
        save_path: Path to save visualization
        score_threshold: Score threshold for displaying predictions
    """
    if class_names is None:
        class_names = [f'Class {i}' for i in range(100)]
    
    # Convert image to numpy
    if isinstance(image, torch.Tensor):
        image = image.permute(1, 2, 0).cpu().numpy()
        # Denormalize if needed
        if image.max() <= 1:
            image = image * 255
        image = image.astype(np.uint8)
    
    fig, axes = plt.subplots(1, 2 if targets else 1, figsize=(15, 8))
    
    if not isinstance(axes, np.ndarray):
        axes = [axes]
    
    # Plot predictions
    ax = axes[0]
    ax.imshow(image)
    ax.set_title('Predictions')
    ax.axis('off')
    
    if 'boxes' in predictions and len(predictions['boxes']) > 0:
        boxes = predictions['boxes'].cpu().numpy()
        labels = predictions['labels'].cpu().numpy()
        scores = predictions.get('scores', torch.ones(len(boxes))).cpu().numpy()
        
        for box, label, score in zip(boxes, labels, scores):
            if score >= score_threshold:
                x1, y1, x2, y2 = box
                rect = Rectangle(
                    (x1, y1), x2 - x1, y2 - y1,
                    linewidth=2, edgecolor='red', facecolor='none'
                )
                ax.add_patch(rect)
                
                # Add label
                label_text = f'{class_names[label]}: {score:.2f}'
                ax.text(
                    x1, y1 - 5, label_text,
                    color='white', fontsize=10,
                    bbox=dict(facecolor='red', alpha=0.5)
                )
    
    # Plot targets if provided
    if targets:
        ax = axes[1]
        ax.imshow(image)
        ax.set_title('Ground Truth')
        ax.axis('off')
        
        if 'boxes' in targets and len(targets['boxes']) > 0:
            boxes = targets['boxes'].cpu().numpy()
            labels = targets['labels'].cpu().numpy()
            
            for box, label in zip(boxes, labels):
                x1, y1, x2, y2 = box
                rect = Rectangle(
                    (x1, y1), x2 - x1, y2 - y1,
                    linewidth=2, edgecolor='green', facecolor='none'
                )
                ax.add_patch(rect)
                
                # Add label
                ax.text(
                    x1, y1 - 5, class_names[label],
                    color='white', fontsize=10,
                    bbox=dict(facecolor='green', alpha=0.5)
                )
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=100, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def draw_boxes(
    image: np.ndarray,
    boxes: np.ndarray,
    labels: np.ndarray,
    scores: Optional[np.ndarray] = None,
    class_names: Optional[List[str]] = None,
    score_threshold: float = 0.5
) -> np.ndarray:
    """
    Draw bounding boxes on image
    
    Args:
        image: Image array (H, W, C)
        boxes: Bounding boxes (N, 4)
        labels: Class labels (N,)
        scores: Confidence scores (N,)
        class_names: List of class names
        score_threshold: Score threshold for displaying boxes
    
    Returns:
        Image with drawn boxes
    """
    image = image.copy()
    
    if class_names is None:
        class_names = [f'Class {i}' for i in range(100)]
    
    # Define colors for each class
    colors = plt.cm.hsv(np.linspace(0, 1, len(class_names)))[:, :3] * 255
    
    for i, (box, label) in enumerate(zip(boxes, labels)):
        if scores is not None and scores[i] < score_threshold:
            continue
        
        x1, y1, x2, y2 = box.astype(int)
        color = colors[label % len(colors)].astype(int).tolist()
        
        # Draw box
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
        
        # Draw label
        label_text = class_names[label]
        if scores is not None:
            label_text += f': {scores[i]:.2f}'
        
        label_size, _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        label_y = max(y1, label_size[1] + 10)
        
        cv2.rectangle(
            image,
            (x1, label_y - label_size[1] - 10),
            (x1 + label_size[0], label_y),
            color, -1
        )
        cv2.putText(
            image, label_text, (x1, label_y - 5),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1
        )
    
    return image


def export_to_onnx(
    model: nn.Module,
    input_shape: Tuple[int, int, int],
    output_path: str,
    device: torch.device
):
    """Export model to ONNX format"""
    model.eval()
    
    # Create dummy input
    dummy_input = torch.randn(1, 3, input_shape[0], input_shape[1]).to(device)
    
    # Export
    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        export_params=True,
        opset_version=11,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['boxes', 'labels', 'scores'],
        dynamic_axes={
            'input': {0: 'batch_size'},
            'boxes': {0: 'num_detections'},
            'labels': {0: 'num_detections'},
            'scores': {0: 'num_detections'}
        }
    )
    
    print(f"Model exported to {output_path}")


if __name__ == "__main__":
    # Test utilities
    boxes1 = torch.tensor([[10, 10, 50, 50], [30, 30, 70, 70]])
    boxes2 = torch.tensor([[20, 20, 60, 60], [70, 70, 100, 100]])
    
    for i in range(len(boxes1)):
        for j in range(len(boxes2)):
            iou = calculate_iou(boxes1[i], boxes2[j])
            print(f"IoU between box {i} and box {j}: {iou:.4f}")