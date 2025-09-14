"""
Training script for instance segmentation
"""
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR, CosineAnnealingLR
import numpy as np
from tqdm import tqdm
import argparse
import os
import json
from datetime import datetime
from typing import Dict, List

from models import get_model
from utils import (
    AverageMeter,
    save_checkpoint,
    load_checkpoint,
    evaluate_segmentation,
    visualize_instance_masks
)


class SegmentationTrainer:
    """Instance segmentation trainer"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        
        # Initialize model
        self.model = get_model(
            config['model_name'],
            num_classes=config['num_classes'],
            pretrained=config.get('pretrained', True)
        ).to(self.device)
        
        # Optimizer
        self.optimizer = self._get_optimizer()
        
        # Scheduler
        self.scheduler = self._get_scheduler()
        
        # Training state
        self.start_epoch = 0
        self.best_map = 0.0
        
        # For demo purposes, we'll use dummy data
        # In practice, use proper data loaders
        self.train_loader = self._get_dummy_loader()
        self.val_loader = self._get_dummy_loader()
    
    def _get_dummy_loader(self):
        """Get dummy data loader for demonstration"""
        dummy_data = []
        for _ in range(10):  # 10 dummy batches
            images = [torch.randn(3, 300, 400) for _ in range(4)]
            targets = []
            for _ in range(4):
                target = {
                    'boxes': torch.rand(5, 4) * 300,
                    'labels': torch.randint(1, self.config['num_classes'], (5,)),
                    'masks': torch.rand(5, 300, 400) > 0.5,
                    'image_id': torch.tensor([0]),
                    'area': torch.rand(5) * 1000,
                    'iscrowd': torch.zeros(5, dtype=torch.int64)
                }
                targets.append(target)
            dummy_data.append((images, targets))
        return dummy_data
    
    def _get_optimizer(self):
        """Get optimizer"""
        params = [p for p in self.model.parameters() if p.requires_grad]
        
        opt_name = self.config.get('optimizer', 'sgd').lower()
        lr = self.config['learning_rate']
        weight_decay = self.config.get('weight_decay', 0.0005)
        
        if opt_name == 'sgd':
            return optim.SGD(
                params,
                lr=lr,
                momentum=0.9,
                weight_decay=weight_decay
            )
        elif opt_name == 'adam':
            return optim.Adam(params, lr=lr, weight_decay=weight_decay)
        elif opt_name == 'adamw':
            return optim.AdamW(params, lr=lr, weight_decay=weight_decay)
        else:
            raise ValueError(f"Unknown optimizer: {opt_name}")
    
    def _get_scheduler(self):
        """Get learning rate scheduler"""
        scheduler_name = self.config.get('scheduler', 'step').lower()
        
        if scheduler_name == 'step':
            return StepLR(
                self.optimizer,
                step_size=self.config.get('step_size', 3),
                gamma=self.config.get('gamma', 0.1)
            )
        elif scheduler_name == 'cosine':
            return CosineAnnealingLR(
                self.optimizer,
                T_max=self.config['epochs'],
                eta_min=1e-6
            )
        else:
            return None
    
    def train_epoch(self, epoch: int) -> Dict:
        """Train for one epoch"""
        self.model.train()
        
        losses = AverageMeter()
        loss_classifier = AverageMeter()
        loss_box_reg = AverageMeter()
        loss_mask = AverageMeter()
        loss_objectness = AverageMeter()
        
        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch}/{self.config['epochs']}")
        
        for images, targets in pbar:
            # Move to device
            images = [img.to(self.device) for img in images]
            targets = [{k: v.to(self.device) for k, v in t.items()} for t in targets]
            
            # Forward pass
            loss_dict = self.model(images, targets)
            
            # Calculate total loss
            loss = sum(loss for loss in loss_dict.values())
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping
            if self.config.get('gradient_clip', 0) > 0:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config['gradient_clip']
                )
            
            self.optimizer.step()
            
            # Update meters
            losses.update(loss.item())
            if 'loss_classifier' in loss_dict:
                loss_classifier.update(loss_dict['loss_classifier'].item())
            if 'loss_box_reg' in loss_dict:
                loss_box_reg.update(loss_dict['loss_box_reg'].item())
            if 'loss_mask' in loss_dict:
                loss_mask.update(loss_dict['loss_mask'].item())
            if 'loss_objectness' in loss_dict:
                loss_objectness.update(loss_dict['loss_objectness'].item())
            
            # Update progress bar
            pbar.set_postfix({
                'Loss': f'{losses.avg:.4f}',
                'Mask': f'{loss_mask.avg:.4f}'
            })
        
        return {
            'loss': losses.avg,
            'loss_classifier': loss_classifier.avg,
            'loss_box_reg': loss_box_reg.avg,
            'loss_mask': loss_mask.avg,
            'loss_objectness': loss_objectness.avg
        }
    
    def validate(self) -> Dict:
        """Validate the model"""
        self.model.eval()
        
        all_predictions = []
        all_targets = []
        
        with torch.no_grad():
            for images, targets in tqdm(self.val_loader, desc="Validating"):
                images = [img.to(self.device) for img in images]
                targets = [{k: v.to(self.device) for k, v in t.items()} for t in targets]
                
                # Get predictions
                predictions = self.model(images)
                
                all_predictions.extend(predictions)
                all_targets.extend(targets)
        
        # Calculate metrics
        metrics = evaluate_segmentation(all_predictions, all_targets)
        
        return metrics
    
    def train(self):
        """Main training loop"""
        for epoch in range(self.start_epoch, self.config['epochs']):
            # Training
            train_metrics = self.train_epoch(epoch)
            
            # Validation
            if epoch % self.config.get('val_frequency', 1) == 0:
                val_metrics = self.validate()
                
                # Log metrics
                print(f"\nEpoch {epoch}/{self.config['epochs']}:")
                print(f"Train Loss: {train_metrics['loss']:.4f}")
                print(f"Train Mask Loss: {train_metrics['loss_mask']:.4f}")
                print(f"Val Mask mAP: {val_metrics.get('mask_map', 0):.4f}")
                
                # Save checkpoint
                is_best = val_metrics.get('mask_map', 0) > self.best_map
                if is_best:
                    self.best_map = val_metrics.get('mask_map', 0)
                
                save_checkpoint({
                    'epoch': epoch + 1,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
                    'best_map': self.best_map,
                    'config': self.config
                }, is_best, checkpoint_dir=self.config['checkpoint_dir'])
            
            # Update scheduler
            if self.scheduler:
                self.scheduler.step()
            
            # Visualize predictions
            if epoch % self.config.get('vis_frequency', 10) == 0:
                self.visualize_batch()
    
    def visualize_batch(self):
        """Visualize predictions on a batch"""
        self.model.eval()
        
        images, targets = self.train_loader[0]
        images = [img.to(self.device) for img in images[:2]]
        
        with torch.no_grad():
            predictions = self.model(images)
        
        # Save visualizations
        save_dir = os.path.join(self.config['checkpoint_dir'], 'visualizations')
        os.makedirs(save_dir, exist_ok=True)
        
        for i in range(len(images)):
            visualize_instance_masks(
                images[i].cpu(),
                predictions[i],
                targets[i] if i < len(targets) else None,
                save_path=os.path.join(save_dir, f'mask_{i}.png')
            )


def main():
    parser = argparse.ArgumentParser(description='Instance Segmentation Training')
    parser.add_argument('--config', type=str, default='configs/default.json',
                        help='Path to config file')
    parser.add_argument('--model', type=str, default='mask_rcnn',
                        help='Model architecture')
    parser.add_argument('--epochs', type=int, default=50,
                        help='Number of epochs')
    parser.add_argument('--batch-size', type=int, default=4,
                        help='Batch size')
    parser.add_argument('--lr', type=float, default=0.005,
                        help='Learning rate')
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to checkpoint to resume')
    args = parser.parse_args()
    
    # Load config
    if os.path.exists(args.config):
        with open(args.config, 'r') as f:
            config = json.load(f)
    else:
        config = {}
    
    # Override config
    config['model_name'] = args.model
    config['epochs'] = args.epochs
    config['batch_size'] = args.batch_size
    config['learning_rate'] = args.lr
    config['resume'] = args.resume
    
    # Set defaults
    config.setdefault('num_classes', 21)
    config.setdefault('checkpoint_dir', './checkpoints')
    config.setdefault('optimizer', 'sgd')
    config.setdefault('weight_decay', 0.0005)
    config.setdefault('gradient_clip', 10.0)
    config.setdefault('val_frequency', 1)
    config.setdefault('vis_frequency', 10)
    
    # Create directories
    os.makedirs(config['checkpoint_dir'], exist_ok=True)
    
    # Train
    trainer = SegmentationTrainer(config)
    trainer.train()


if __name__ == "__main__":
    main()