"""
Inference script for image classification
"""
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import numpy as np
import argparse
import json
import time
from typing import Dict, List, Tuple
import os

from models import get_model
from utils import load_checkpoint


class ImageClassifier:
    """Image classification inference class"""
    
    def __init__(
        self,
        model_path: str,
        model_name: str = 'resnet18',
        device: str = 'auto',
        num_classes: int = 10
    ):
        """
        Initialize the classifier
        
        Args:
            model_path: Path to model checkpoint
            model_name: Name of the model architecture
            device: Device to run on ('cuda', 'cpu', or 'auto')
            num_classes: Number of output classes
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
            if 'best_acc' in checkpoint:
                print(f"Model accuracy: {checkpoint['best_acc']:.2f}%")
        else:
            print(f"Warning: Model path {model_path} not found. Using random weights.")
        
        self.model.to(self.device)
        self.model.eval()
        
        # Define transforms
        self.transform = transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=(0.4914, 0.4822, 0.4465),
                std=(0.2023, 0.1994, 0.2010)
            )
        ])
        
        # Class names for CIFAR-10
        self.class_names = [
            'airplane', 'automobile', 'bird', 'cat', 'deer',
            'dog', 'frog', 'horse', 'ship', 'truck'
        ]
    
    def preprocess_image(self, image_path: str) -> torch.Tensor:
        """
        Preprocess image for inference
        
        Args:
            image_path: Path to input image
        
        Returns:
            Preprocessed image tensor
        """
        image = Image.open(image_path).convert('RGB')
        image = self.transform(image)
        image = image.unsqueeze(0)  # Add batch dimension
        return image
    
    def predict(
        self,
        image_path: str,
        top_k: int = 5,
        return_probs: bool = True
    ) -> Dict:
        """
        Predict class for a single image
        
        Args:
            image_path: Path to input image
            top_k: Number of top predictions to return
            return_probs: Whether to return probabilities
        
        Returns:
            Dictionary with predictions
        """
        # Preprocess image
        image = self.preprocess_image(image_path)
        image = image.to(self.device)
        
        # Inference
        start_time = time.time()
        with torch.no_grad():
            outputs = self.model(image)
            
            if return_probs:
                probs = torch.softmax(outputs, dim=1)
                top_probs, top_indices = torch.topk(probs, k=min(top_k, len(self.class_names)))
            else:
                top_probs, top_indices = torch.topk(outputs, k=min(top_k, len(self.class_names)))
        
        inference_time = (time.time() - start_time) * 1000  # Convert to ms
        
        # Format results
        predictions = []
        for i in range(top_indices.size(1)):
            pred = {
                'class': self.class_names[top_indices[0, i].item()],
                'class_id': top_indices[0, i].item(),
                'confidence': top_probs[0, i].item()
            }
            predictions.append(pred)
        
        return {
            'predictions': predictions,
            'inference_time_ms': inference_time
        }
    
    def batch_predict(
        self,
        image_paths: List[str],
        batch_size: int = 32
    ) -> List[Dict]:
        """
        Predict classes for multiple images
        
        Args:
            image_paths: List of image paths
            batch_size: Batch size for inference
        
        Returns:
            List of prediction dictionaries
        """
        results = []
        
        for i in range(0, len(image_paths), batch_size):
            batch_paths = image_paths[i:i+batch_size]
            batch_images = []
            
            for path in batch_paths:
                image = self.preprocess_image(path)
                batch_images.append(image)
            
            batch_tensor = torch.cat(batch_images, dim=0).to(self.device)
            
            with torch.no_grad():
                outputs = self.model(batch_tensor)
                probs = torch.softmax(outputs, dim=1)
                top_probs, top_indices = torch.topk(probs, k=1)
            
            for j, path in enumerate(batch_paths):
                result = {
                    'image': path,
                    'prediction': self.class_names[top_indices[j, 0].item()],
                    'confidence': top_probs[j, 0].item()
                }
                results.append(result)
        
        return results
    
    def test_time_augmentation(
        self,
        image_path: str,
        num_augmentations: int = 10
    ) -> Dict:
        """
        Apply test-time augmentation for more robust predictions
        
        Args:
            image_path: Path to input image
            num_augmentations: Number of augmented versions to create
        
        Returns:
            Averaged predictions
        """
        image = Image.open(image_path).convert('RGB')
        
        # Define augmentation transforms
        tta_transform = transforms.Compose([
            transforms.RandomResizedCrop(32, scale=(0.8, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.1, contrast=0.1),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=(0.4914, 0.4822, 0.4465),
                std=(0.2023, 0.1994, 0.2010)
            )
        ])
        
        # Collect predictions
        all_probs = []
        
        for _ in range(num_augmentations):
            augmented = tta_transform(image).unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                outputs = self.model(augmented)
                probs = torch.softmax(outputs, dim=1)
                all_probs.append(probs)
        
        # Average predictions
        avg_probs = torch.stack(all_probs).mean(dim=0)
        top_probs, top_indices = torch.topk(avg_probs, k=5)
        
        # Format results
        predictions = []
        for i in range(top_indices.size(1)):
            pred = {
                'class': self.class_names[top_indices[0, i].item()],
                'class_id': top_indices[0, i].item(),
                'confidence': top_probs[0, i].item()
            }
            predictions.append(pred)
        
        return {
            'predictions': predictions,
            'num_augmentations': num_augmentations
        }


def main():
    parser = argparse.ArgumentParser(description='Image Classification Inference')
    parser.add_argument('--image', type=str, required=True,
                        help='Path to input image')
    parser.add_argument('--model', type=str, default='checkpoints/best_model.pth',
                        help='Path to model checkpoint')
    parser.add_argument('--model-name', type=str, default='resnet18',
                        help='Model architecture name')
    parser.add_argument('--device', type=str, default='auto',
                        help='Device to use (cuda/cpu/auto)')
    parser.add_argument('--top-k', type=int, default=5,
                        help='Number of top predictions to show')
    parser.add_argument('--tta', action='store_true',
                        help='Use test-time augmentation')
    parser.add_argument('--output', type=str, default=None,
                        help='Path to save results JSON')
    
    args = parser.parse_args()
    
    # Initialize classifier
    classifier = ImageClassifier(
        model_path=args.model,
        model_name=args.model_name,
        device=args.device
    )
    
    # Run inference
    if args.tta:
        results = classifier.test_time_augmentation(args.image)
    else:
        results = classifier.predict(args.image, top_k=args.top_k)
    
    # Print results
    print(f"\nPredictions for {args.image}:")
    print("-" * 50)
    for i, pred in enumerate(results['predictions'], 1):
        print(f"{i}. {pred['class']:<15} (confidence: {pred['confidence']:.4f})")
    
    if 'inference_time_ms' in results:
        print(f"\nInference time: {results['inference_time_ms']:.2f} ms")
    
    # Save results if specified
    if args.output:
        with open(args.output, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()