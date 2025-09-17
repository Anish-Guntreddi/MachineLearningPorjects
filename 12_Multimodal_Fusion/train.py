"""
Training script for Multimodal Fusion models
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm
import argparse
import json
import os
from pathlib import Path
from typing import Dict, Optional
import wandb

from data_loader import MultimodalDataModule, MultimodalDataset, VQADataset, ImageTextDataset
from models import get_fusion_model, CLIPModel, VQAModel
from utils import (
    calculate_metrics, plot_confusion_matrix, plot_training_curves,
    visualize_attention_weights, EarlyStopping, save_checkpoint,
    load_checkpoint
)


class MultimodalTrainer:
    """Trainer for multimodal models"""
    
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        test_loader: DataLoader,
        config: Dict,
        device: torch.device
    ):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.config = config
        self.device = device
        
        # Move model to device
        self.model.to(device)
        
        # Setup optimizer
        self.optimizer = self._get_optimizer()
        
        # Setup scheduler
        self.scheduler = self._get_scheduler()
        
        # Setup loss function
        self.criterion = self._get_criterion()
        
        # Early stopping
        self.early_stopping = EarlyStopping(
            patience=config.get('patience', 10),
            mode='max' if 'accuracy' in config.get('monitor_metric', 'loss') else 'min'
        )
        
        # Mixed precision training
        self.use_amp = config.get('use_amp', False)
        if self.use_amp:
            self.scaler = torch.cuda.amp.GradScaler()
        
        # Tracking
        self.train_losses = []
        self.val_losses = []
        self.val_metrics = []
        self.best_metric = float('-inf') if 'accuracy' in config.get('monitor_metric', 'loss') else float('inf')
    
    def _get_optimizer(self):
        """Get optimizer"""
        opt_name = self.config.get('optimizer', 'adam').lower()
        lr = self.config.get('learning_rate', 1e-4)
        weight_decay = self.config.get('weight_decay', 1e-5)
        
        if opt_name == 'adam':
            return optim.Adam(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        elif opt_name == 'adamw':
            return optim.AdamW(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        elif opt_name == 'sgd':
            return optim.SGD(self.model.parameters(), lr=lr, momentum=0.9, weight_decay=weight_decay)
        else:
            raise ValueError(f"Unknown optimizer: {opt_name}")
    
    def _get_scheduler(self):
        """Get learning rate scheduler"""
        scheduler_name = self.config.get('scheduler', 'none').lower()
        
        if scheduler_name == 'cosine':
            return optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=self.config.get('epochs', 100)
            )
        elif scheduler_name == 'step':
            return optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=self.config.get('step_size', 30),
                gamma=self.config.get('gamma', 0.1)
            )
        elif scheduler_name == 'reduce':
            return optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode='max' if 'accuracy' in self.config.get('monitor_metric', 'loss') else 'min',
                patience=5,
                factor=0.5
            )
        else:
            return None
    
    def _get_criterion(self):
        """Get loss function"""
        task = self.config.get('task', 'classification')
        
        if task == 'classification':
            return nn.CrossEntropyLoss()
        elif task == 'vqa':
            return nn.CrossEntropyLoss()
        elif task == 'matching':
            return nn.BCEWithLogitsLoss()
        elif task == 'clip':
            return CLIPLoss()
        else:
            return nn.MSELoss()
    
    def train_epoch(self, epoch: int) -> float:
        """Train for one epoch"""
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        
        pbar = tqdm(self.train_loader, desc=f'Epoch {epoch+1} [Train]')
        
        for batch in pbar:
            # Move data to device
            batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                    for k, v in batch.items()}
            
            # Forward pass with mixed precision
            if self.use_amp:
                with torch.cuda.amp.autocast():
                    loss = self._forward_pass(batch)
            else:
                loss = self._forward_pass(batch)
            
            # Backward pass
            self.optimizer.zero_grad()
            
            if self.use_amp:
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss.backward()
                self.optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
            
            # Update progress bar
            pbar.set_postfix({'loss': loss.item()})
        
        avg_loss = total_loss / num_batches
        return avg_loss
    
    def _forward_pass(self, batch: Dict) -> torch.Tensor:
        """Forward pass for different model types"""
        model_type = self.config.get('model_type', 'fusion')
        
        if model_type == 'vqa':
            # VQA task
            output = self.model(
                batch['image'],
                batch['question_ids'],
                batch.get('question_mask')
            )
            loss = self.criterion(output, batch['answer'])
        
        elif model_type == 'clip':
            # CLIP-style contrastive learning
            logits_i2t, logits_t2i = self.model(
                batch['image'],
                batch['text_ids']
            )
            labels = torch.arange(len(batch['image'])).to(self.device)
            loss_i2t = self.criterion(logits_i2t, labels)
            loss_t2i = self.criterion(logits_t2i, labels)
            loss = (loss_i2t + loss_t2i) / 2
        
        elif model_type == 'matching':
            # Image-text matching
            inputs = {
                'image': batch['image'],
                'text': batch['text_ids']
            }
            output = self.model(inputs)
            loss = self.criterion(output.squeeze(), batch['label'].float())
        
        else:
            # General multimodal fusion
            inputs = {}
            if 'image' in batch:
                inputs['image'] = batch['image']
            if 'text_ids' in batch:
                inputs['text'] = batch['text_ids']
            if 'audio' in batch:
                inputs['audio'] = batch['audio']
            
            output = self.model(inputs)
            loss = self.criterion(output, batch['label'])
        
        return loss
    
    def validate(self, epoch: int) -> Dict[str, float]:
        """Validate the model"""
        self.model.eval()
        total_loss = 0.0
        all_predictions = []
        all_labels = []
        num_batches = 0
        
        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc=f'Epoch {epoch+1} [Val]'):
                # Move data to device
                batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                        for k, v in batch.items()}
                
                # Forward pass
                if self.use_amp:
                    with torch.cuda.amp.autocast():
                        loss = self._forward_pass(batch)
                        predictions = self._get_predictions(batch)
                else:
                    loss = self._forward_pass(batch)
                    predictions = self._get_predictions(batch)
                
                total_loss += loss.item()
                num_batches += 1
                
                # Collect predictions and labels
                all_predictions.extend(predictions.cpu().numpy())
                all_labels.extend(batch['label'].cpu().numpy())
        
        # Calculate metrics
        avg_loss = total_loss / num_batches
        metrics = calculate_metrics(
            np.array(all_predictions),
            np.array(all_labels),
            task=self.config.get('task', 'classification')
        )
        metrics['loss'] = avg_loss
        
        return metrics
    
    def _get_predictions(self, batch: Dict) -> torch.Tensor:
        """Get predictions for different model types"""
        model_type = self.config.get('model_type', 'fusion')
        
        if model_type == 'vqa':
            output = self.model(
                batch['image'],
                batch['question_ids'],
                batch.get('question_mask')
            )
            predictions = torch.argmax(output, dim=-1)
        
        elif model_type == 'clip':
            image_features = self.model.encode_image(batch['image'])
            text_features = self.model.encode_text(batch['text_ids'])
            similarity = image_features @ text_features.t()
            predictions = torch.argmax(similarity, dim=-1)
        
        elif model_type == 'matching':
            inputs = {
                'image': batch['image'],
                'text': batch['text_ids']
            }
            output = self.model(inputs)
            predictions = (torch.sigmoid(output.squeeze()) > 0.5).long()
        
        else:
            inputs = {}
            if 'image' in batch:
                inputs['image'] = batch['image']
            if 'text_ids' in batch:
                inputs['text'] = batch['text_ids']
            if 'audio' in batch:
                inputs['audio'] = batch['audio']
            
            output = self.model(inputs)
            predictions = torch.argmax(output, dim=-1)
        
        return predictions
    
    def train(self):
        """Main training loop"""
        print(f"Starting training on {self.device}")
        print(f"Model: {self.config.get('model_name', 'unknown')}")
        print(f"Task: {self.config.get('task', 'classification')}")
        
        for epoch in range(self.config.get('epochs', 100)):
            # Train
            train_loss = self.train_epoch(epoch)
            self.train_losses.append(train_loss)
            
            # Validate
            val_metrics = self.validate(epoch)
            self.val_losses.append(val_metrics['loss'])
            self.val_metrics.append(val_metrics)
            
            # Print metrics
            print(f"\nEpoch {epoch+1}/{self.config.get('epochs', 100)}")
            print(f"Train Loss: {train_loss:.4f}")
            print(f"Val Loss: {val_metrics['loss']:.4f}")
            
            if 'accuracy' in val_metrics:
                print(f"Val Accuracy: {val_metrics['accuracy']:.4f}")
            if 'f1' in val_metrics:
                print(f"Val F1: {val_metrics['f1']:.4f}")
            
            # Learning rate scheduling
            if self.scheduler is not None:
                if isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                    monitor_value = val_metrics.get(
                        self.config.get('monitor_metric', 'loss'),
                        val_metrics['loss']
                    )
                    self.scheduler.step(monitor_value)
                else:
                    self.scheduler.step()
            
            # Save best model
            monitor_metric = self.config.get('monitor_metric', 'loss')
            current_metric = val_metrics.get(monitor_metric, val_metrics['loss'])
            
            is_best = False
            if 'accuracy' in monitor_metric or 'f1' in monitor_metric:
                if current_metric > self.best_metric:
                    self.best_metric = current_metric
                    is_best = True
            else:
                if current_metric < self.best_metric:
                    self.best_metric = current_metric
                    is_best = True
            
            if is_best:
                save_checkpoint(
                    self.model,
                    self.optimizer,
                    epoch,
                    val_metrics,
                    self.config['checkpoint_dir'] / 'best_model.pth'
                )
                print(f"Saved best model with {monitor_metric}: {current_metric:.4f}")
            
            # Early stopping
            if self.early_stopping(current_metric):
                print(f"Early stopping triggered at epoch {epoch+1}")
                break
            
            # Log to wandb
            if wandb.run is not None:
                log_dict = {
                    'train_loss': train_loss,
                    'val_loss': val_metrics['loss'],
                    'epoch': epoch
                }
                log_dict.update({f'val_{k}': v for k, v in val_metrics.items() if k != 'loss'})
                wandb.log(log_dict)
        
        print("\nTraining completed!")
        return self.train_losses, self.val_losses, self.val_metrics
    
    def test(self):
        """Test the model"""
        print("\nTesting model...")
        
        # Load best model
        checkpoint_path = self.config['checkpoint_dir'] / 'best_model.pth'
        if checkpoint_path.exists():
            load_checkpoint(self.model, self.optimizer, checkpoint_path)
        
        self.model.eval()
        all_predictions = []
        all_labels = []
        
        with torch.no_grad():
            for batch in tqdm(self.test_loader, desc='Testing'):
                # Move data to device
                batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                        for k, v in batch.items()}
                
                # Get predictions
                predictions = self._get_predictions(batch)
                
                all_predictions.extend(predictions.cpu().numpy())
                all_labels.extend(batch['label'].cpu().numpy())
        
        # Calculate test metrics
        test_metrics = calculate_metrics(
            np.array(all_predictions),
            np.array(all_labels),
            task=self.config.get('task', 'classification')
        )
        
        print("\nTest Results:")
        for metric, value in test_metrics.items():
            print(f"{metric}: {value:.4f}")
        
        # Plot confusion matrix for classification
        if self.config.get('task', 'classification') == 'classification':
            plot_confusion_matrix(
                all_labels,
                all_predictions,
                save_path=self.config['output_dir'] / 'confusion_matrix.png'
            )
        
        return test_metrics


class CLIPLoss(nn.Module):
    """Contrastive loss for CLIP-style training"""
    
    def __init__(self):
        super().__init__()
        self.ce_loss = nn.CrossEntropyLoss()
    
    def forward(self, logits, labels):
        return self.ce_loss(logits, labels)


def main(args):
    """Main training function"""
    # Set random seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Create directories
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = output_dir / 'checkpoints'
    checkpoint_dir.mkdir(exist_ok=True)
    
    # Initialize wandb
    if args.use_wandb:
        wandb.init(
            project='multimodal-fusion',
            name=args.exp_name,
            config=vars(args)
        )
    
    # Setup data
    print("Loading data...")
    
    if args.dataset == 'multimodal':
        dm = MultimodalDataModule(
            dataset_name='multimodal',
            data_path=args.data_path,
            modalities=args.modalities.split(','),
            batch_size=args.batch_size,
            num_workers=args.num_workers
        )
    elif args.dataset == 'vqa':
        dm = MultimodalDataModule(
            dataset_name='vqa',
            data_path=args.data_path,
            batch_size=args.batch_size,
            num_workers=args.num_workers
        )
    elif args.dataset == 'image_text':
        dm = MultimodalDataModule(
            dataset_name='image_text',
            data_path=args.data_path,
            batch_size=args.batch_size,
            num_workers=args.num_workers
        )
    else:
        raise ValueError(f"Unknown dataset: {args.dataset}")
    
    dm.setup()
    
    train_loader = dm.train_dataloader()
    val_loader = dm.val_dataloader()
    test_loader = dm.test_dataloader()
    
    # Setup model
    print(f"Creating model: {args.model}")
    
    if args.model == 'vqa':
        model = VQAModel(
            image_encoder=args.image_encoder,
            text_encoder=args.text_encoder,
            hidden_dim=args.hidden_dim,
            num_answers=args.num_answers,
            dropout=args.dropout
        )
    elif args.model == 'clip':
        model = CLIPModel(
            image_dim=args.image_dim,
            text_dim=args.text_dim,
            embed_dim=args.embed_dim,
            temperature=args.temperature
        )
    else:
        model_kwargs = {
            'image_dim': args.image_dim,
            'text_dim': args.text_dim,
            'audio_dim': args.audio_dim,
            'output_dim': args.num_classes,
            'dropout': args.dropout
        }
        
        if args.model == 'transformer':
            model_kwargs.update({
                'd_model': args.hidden_dim,
                'num_heads': args.num_heads,
                'num_layers': args.num_layers
            })
        
        model = get_fusion_model(args.model, **model_kwargs)
    
    # Print model info
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # Training config
    config = {
        'model_name': args.model,
        'model_type': args.model_type,
        'task': args.task,
        'epochs': args.epochs,
        'learning_rate': args.learning_rate,
        'weight_decay': args.weight_decay,
        'optimizer': args.optimizer,
        'scheduler': args.scheduler,
        'patience': args.patience,
        'monitor_metric': args.monitor_metric,
        'use_amp': args.use_amp,
        'output_dir': output_dir,
        'checkpoint_dir': checkpoint_dir
    }
    
    # Create trainer
    trainer = MultimodalTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        config=config,
        device=device
    )
    
    # Train
    train_losses, val_losses, val_metrics = trainer.train()
    
    # Plot training curves
    plot_training_curves(
        train_losses,
        val_losses,
        save_path=output_dir / 'training_curves.png'
    )
    
    # Test
    test_metrics = trainer.test()
    
    # Save results
    results = {
        'args': vars(args),
        'train_losses': train_losses,
        'val_losses': val_losses,
        'val_metrics': val_metrics,
        'test_metrics': test_metrics
    }
    
    with open(output_dir / 'results.json', 'w') as f:
        json.dump(results, f, indent=2, default=lambda x: float(x) if isinstance(x, np.floating) else x)
    
    print(f"\nResults saved to {output_dir}")
    
    if args.use_wandb:
        wandb.finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train Multimodal Fusion models')
    
    # Data arguments
    parser.add_argument('--data_path', type=str, default='./data',
                       help='Path to data directory')
    parser.add_argument('--dataset', type=str, default='multimodal',
                       choices=['multimodal', 'vqa', 'image_text'],
                       help='Dataset to use')
    parser.add_argument('--modalities', type=str, default='image,text,audio',
                       help='Comma-separated list of modalities')
    
    # Model arguments
    parser.add_argument('--model', type=str, default='early_fusion',
                       choices=['early_fusion', 'late_fusion', 'hierarchical',
                               'vqa', 'clip', 'transformer', 'gated', 'tensor'],
                       help='Model architecture')
    parser.add_argument('--model_type', type=str, default='fusion',
                       choices=['fusion', 'vqa', 'clip', 'matching'],
                       help='Model type for training')
    parser.add_argument('--task', type=str, default='classification',
                       choices=['classification', 'vqa', 'matching', 'clip'],
                       help='Task type')
    parser.add_argument('--image_dim', type=int, default=2048,
                       help='Image feature dimension')
    parser.add_argument('--text_dim', type=int, default=768,
                       help='Text feature dimension')
    parser.add_argument('--audio_dim', type=int, default=128,
                       help='Audio feature dimension')
    parser.add_argument('--hidden_dim', type=int, default=512,
                       help='Hidden dimension')
    parser.add_argument('--embed_dim', type=int, default=512,
                       help='Embedding dimension for CLIP')
    parser.add_argument('--num_classes', type=int, default=10,
                       help='Number of output classes')
    parser.add_argument('--num_answers', type=int, default=1000,
                       help='Number of possible answers for VQA')
    parser.add_argument('--dropout', type=float, default=0.2,
                       help='Dropout rate')
    parser.add_argument('--temperature', type=float, default=0.07,
                       help='Temperature for CLIP')
    
    # Transformer arguments
    parser.add_argument('--num_heads', type=int, default=8,
                       help='Number of attention heads')
    parser.add_argument('--num_layers', type=int, default=6,
                       help='Number of transformer layers')
    
    # Encoder arguments
    parser.add_argument('--image_encoder', type=str, default='resnet',
                       help='Image encoder type')
    parser.add_argument('--text_encoder', type=str, default='bert',
                       help='Text encoder type')
    
    # Training arguments
    parser.add_argument('--epochs', type=int, default=100,
                       help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=32,
                       help='Batch size')
    parser.add_argument('--learning_rate', type=float, default=1e-4,
                       help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-5,
                       help='Weight decay')
    parser.add_argument('--optimizer', type=str, default='adamw',
                       choices=['adam', 'adamw', 'sgd'],
                       help='Optimizer')
    parser.add_argument('--scheduler', type=str, default='cosine',
                       choices=['none', 'cosine', 'step', 'reduce'],
                       help='Learning rate scheduler')
    parser.add_argument('--patience', type=int, default=10,
                       help='Early stopping patience')
    parser.add_argument('--monitor_metric', type=str, default='accuracy',
                       help='Metric to monitor for early stopping')
    parser.add_argument('--use_amp', action='store_true',
                       help='Use mixed precision training')
    
    # Other arguments
    parser.add_argument('--num_workers', type=int, default=4,
                       help='Number of data loading workers')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    parser.add_argument('--output_dir', type=str, default='./outputs/multimodal',
                       help='Output directory')
    parser.add_argument('--exp_name', type=str, default='multimodal_fusion',
                       help='Experiment name')
    parser.add_argument('--use_wandb', action='store_true',
                       help='Use Weights & Biases for logging')
    
    args = parser.parse_args()
    main(args)