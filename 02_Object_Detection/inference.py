"""
Inference script for object detection
"""
import torch
import torchvision
import numpy as np
from PIL import Image
import cv2
import argparse
import json
import time
import os
from typing import Dict, List, Tuple, Optional

from models import get_model
from utils import load_checkpoint, non_max_suppression, draw_boxes


class ObjectDetector:
    """Object detection inference class"""
    
    def __init__(
        self,
        model_path: str,
        model_name: str = 'faster_rcnn',
        device: str = 'auto',
        num_classes: int = 21,
        class_names: Optional[List[str]] = None
    ):
        """
        Initialize detector
        
        Args:
            model_path: Path to model checkpoint
            model_name: Model architecture name
            device: Device to run on
            num_classes: Number of classes
            class_names: List of class names
        """
        # Set device
        if device == 'auto':
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        
        print(f"Using device: {self.device}")
        
        # Load model
        self.model = get_model(model_name, num_classes=num_classes, pretrained=False)
        
        # Load checkpoint
        if os.path.exists(model_path):
            checkpoint = load_checkpoint(model_path, self.model)
            print(f"Loaded model from {model_path}")
            if 'best_map' in checkpoint:
                print(f"Model mAP: {checkpoint['best_map']:.4f}")
        else:
            print(f"Warning: Model path {model_path} not found. Using random weights.")
        
        self.model.to(self.device)
        self.model.eval()
        
        # Class names
        if class_names is None:
            # VOC classes
            self.class_names = [
                'background', 'aeroplane', 'bicycle', 'bird', 'boat',
                'bottle', 'bus', 'car', 'cat', 'chair', 'cow',
                'diningtable', 'dog', 'horse', 'motorbike', 'person',
                'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor'
            ]
        else:
            self.class_names = class_names
        
        # Image transforms
        self.transform = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor()
        ])
    
    def preprocess_image(self, image_path: str) -> Tuple[torch.Tensor, Tuple[int, int]]:
        """
        Preprocess image for inference
        
        Args:
            image_path: Path to input image
        
        Returns:
            Preprocessed image tensor and original size
        """
        # Load image
        image = Image.open(image_path).convert('RGB')
        original_size = image.size
        
        # Convert to tensor
        image_tensor = self.transform(image)
        
        return image_tensor, original_size
    
    def detect(
        self,
        image_path: str,
        score_threshold: float = 0.5,
        nms_threshold: float = 0.5,
        max_detections: int = 100
    ) -> Dict:
        """
        Detect objects in image
        
        Args:
            image_path: Path to input image
            score_threshold: Score threshold for predictions
            nms_threshold: NMS IoU threshold
            max_detections: Maximum number of detections
        
        Returns:
            Detection results
        """
        # Preprocess image
        image_tensor, original_size = self.preprocess_image(image_path)
        image_tensor = image_tensor.unsqueeze(0).to(self.device)
        
        # Inference
        start_time = time.time()
        with torch.no_grad():
            predictions = self.model(image_tensor)
        inference_time = (time.time() - start_time) * 1000
        
        # Process predictions
        pred = predictions[0]
        
        # Filter by score
        if 'scores' in pred:
            keep = pred['scores'] > score_threshold
            boxes = pred['boxes'][keep]
            labels = pred['labels'][keep]
            scores = pred['scores'][keep]
        else:
            boxes = pred['boxes']
            labels = pred['labels']
            scores = torch.ones(len(boxes))
        
        # Apply NMS
        if len(boxes) > 0:
            keep = torchvision.ops.nms(boxes, scores, nms_threshold)
            boxes = boxes[keep][:max_detections]
            labels = labels[keep][:max_detections]
            scores = scores[keep][:max_detections]
        
        # Format results
        detections = []
        for box, label, score in zip(boxes, labels, scores):
            det = {
                'bbox': box.cpu().numpy().tolist(),
                'class': self.class_names[label.item()],
                'class_id': label.item(),
                'score': score.item()
            }
            detections.append(det)
        
        return {
            'detections': detections,
            'num_detections': len(detections),
            'inference_time_ms': inference_time,
            'image_size': original_size
        }
    
    def detect_batch(
        self,
        image_paths: List[str],
        score_threshold: float = 0.5,
        nms_threshold: float = 0.5
    ) -> List[Dict]:
        """
        Detect objects in multiple images
        
        Args:
            image_paths: List of image paths
            score_threshold: Score threshold
            nms_threshold: NMS threshold
        
        Returns:
            List of detection results
        """
        results = []
        
        for path in image_paths:
            result = self.detect(path, score_threshold, nms_threshold)
            result['image'] = path
            results.append(result)
        
        return results
    
    def detect_video(
        self,
        video_path: str,
        output_path: str,
        score_threshold: float = 0.5,
        nms_threshold: float = 0.5,
        skip_frames: int = 1
    ):
        """
        Detect objects in video
        
        Args:
            video_path: Path to input video
            output_path: Path to output video
            score_threshold: Score threshold
            nms_threshold: NMS threshold
            skip_frames: Process every N frames
        """
        # Open video
        cap = cv2.VideoCapture(video_path)
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # Create video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        frame_count = 0
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            if frame_count % skip_frames == 0:
                # Convert frame to RGB
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                # Save temporary image
                temp_path = 'temp_frame.jpg'
                cv2.imwrite(temp_path, frame)
                
                # Detect objects
                results = self.detect(temp_path, score_threshold, nms_threshold)
                
                # Draw boxes
                if results['detections']:
                    boxes = np.array([d['bbox'] for d in results['detections']])
                    labels = np.array([d['class_id'] for d in results['detections']])
                    scores = np.array([d['score'] for d in results['detections']])
                    
                    frame = draw_boxes(
                        frame, boxes, labels, scores,
                        self.class_names, score_threshold
                    )
                
                # Clean up
                os.remove(temp_path)
            
            # Write frame
            out.write(frame)
            frame_count += 1
        
        cap.release()
        out.release()
        print(f"Video saved to {output_path}")


def visualize_detections(image_path: str, results: Dict, output_path: str):
    """
    Visualize detection results
    
    Args:
        image_path: Path to original image
        results: Detection results
        output_path: Path to save visualization
    """
    # Load image
    image = cv2.imread(image_path)
    
    # Draw detections
    for det in results['detections']:
        x1, y1, x2, y2 = [int(x) for x in det['bbox']]
        label = f"{det['class']}: {det['score']:.2f}"
        
        # Draw box
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        # Draw label
        label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        label_y = max(y1, label_size[1] + 10)
        
        cv2.rectangle(
            image,
            (x1, label_y - label_size[1] - 10),
            (x1 + label_size[0], label_y),
            (0, 255, 0), -1
        )
        cv2.putText(
            image, label, (x1, label_y - 5),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1
        )
    
    # Save image
    cv2.imwrite(output_path, image)
    print(f"Visualization saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(description='Object Detection Inference')
    parser.add_argument('--image', type=str, required=True,
                        help='Path to input image or video')
    parser.add_argument('--model', type=str, default='checkpoints/best_model.pth',
                        help='Path to model checkpoint')
    parser.add_argument('--model-name', type=str, default='faster_rcnn',
                        help='Model architecture')
    parser.add_argument('--device', type=str, default='auto',
                        help='Device to use')
    parser.add_argument('--score-threshold', type=float, default=0.5,
                        help='Score threshold')
    parser.add_argument('--nms-threshold', type=float, default=0.5,
                        help='NMS IoU threshold')
    parser.add_argument('--output', type=str, default=None,
                        help='Output path for visualization')
    parser.add_argument('--video', action='store_true',
                        help='Process video instead of image')
    
    args = parser.parse_args()
    
    # Initialize detector
    detector = ObjectDetector(
        model_path=args.model,
        model_name=args.model_name,
        device=args.device
    )
    
    if args.video:
        # Process video
        output_path = args.output or 'output_video.mp4'
        detector.detect_video(
            args.image,
            output_path,
            args.score_threshold,
            args.nms_threshold
        )
    else:
        # Process image
        results = detector.detect(
            args.image,
            args.score_threshold,
            args.nms_threshold
        )
        
        # Print results
        print(f"\nDetected {results['num_detections']} objects:")
        print("-" * 50)
        for i, det in enumerate(results['detections'], 1):
            print(f"{i}. {det['class']:<15} (score: {det['score']:.4f})")
            print(f"   Bbox: {det['bbox']}")
        
        print(f"\nInference time: {results['inference_time_ms']:.2f} ms")
        
        # Visualize if output specified
        if args.output:
            visualize_detections(args.image, results, args.output)


if __name__ == "__main__":
    main()