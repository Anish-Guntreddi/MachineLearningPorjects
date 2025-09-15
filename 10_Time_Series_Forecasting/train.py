"""
Training script for Time Series Forecasting
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau
from torch.cuda.amp import GradScaler, autocast
import numpy as np
from tqdm import tqdm
import argparse
import os
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple
import wandb

from data_loader import TimeSeriesDataModule
from models import get_model
from utils import (
    calculate_metrics, plot_predictions, plot_training_curves,
    EarlyStopping, inverse_transform_predictions
)


class TimeSeriesTrainer:
    """Trainer for time series forecasting models"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        
        # Initialize data module
        self.data_module = TimeSeriesDataModule(
            dataset_name=config['dataset'],
            data_path=config.get('data_path'),
            sequence_length=config['sequence_length'],
            prediction_length=config['prediction_length'],
            batch_size=config['batch_size'],
            num_workers=config.get('num_workers', 4)
        )
        self.data_module.setup()
        
        # Get input dimension
        sample_batch = next(iter(self.data_module.train_dataloader()))
        input_dim = sample_batch['input'].shape[-1]
        
        # Initialize model
        self.model = get_model(
            config['model_name'],
            input_dim=input_dim,
            prediction_length=config['prediction_length'],
            hidden_dim=config.get('hidden_dim', 128),
            num_layers=config.get('num_layers', 2),
            dropout=config.get('dropout', 0.2)
        ).to(self.device)
        
        print(f"Model: {config['model_name']}")
        print(f"Parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        print(f"Input dim: {input_dim}, Prediction length: {config['prediction_length']}")
        
        # Loss function
        self.criterion = self._get_loss_function()
        
        # Optimizer
        self.optimizer = self._get_optimizer()
        
        # Scheduler
        self.scheduler = self._get_scheduler()
        
        # Mixed precision training
        self.use_amp = config.get('use_amp', True) and torch.cuda.is_available()
        self.scaler = GradScaler() if self.use_amp else None
        
        # Early stopping
        self.early_stopping = EarlyStopping(
            patience=config.get('early_stopping_patience', 10),
            mode='min',
            delta=0.0001
        )
        
        # Tracking
        self.train_losses = []
        self.val_losses = []
        self.val_maes = []
        self.val_mapes = []
        self.best_val_loss = float('inf')
        
        # Initialize wandb if enabled
        if config.get('use_wandb', False):
            wandb.init(
                project="time-series-forecasting",
                config=config,
                name=f"{config['model_name']}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            )
    
    def _get_loss_function(self):
        """Get loss function"""
        loss_name = self.config.get('loss', 'mse').lower()
        
        if loss_name == 'mse':
            return nn.MSELoss()
        elif loss_name == 'mae':
            return nn.L1Loss()
        elif loss_name == 'huber':
            return nn.HuberLoss()
        elif loss_name == 'quantile':
            return QuantileLoss(quantiles=self.config.get('quantiles', [0.1, 0.5, 0.9]))
        else:
            return nn.MSELoss()
    
    def _get_optimizer(self):
        """Get optimizer"""
        opt_name = self.config.get('optimizer', 'adam').lower()
        lr = self.config['learning_rate']
        weight_decay = self.config.get('weight_decay', 1e-4)
        
        if opt_name == 'adam':
            return optim.Adam(
                self.model.parameters(),
                lr=lr,
                weight_decay=weight_decay
            )
        elif opt_name == 'adamw':
            return optim.AdamW(
                self.model.parameters(),
                lr=lr,
                weight_decay=weight_decay
            )
        elif opt_name == 'sgd':
            return optim.SGD(
                self.model.parameters(),
                lr=lr,
                momentum=0.9,
                weight_decay=weight_decay
            )
        else:
            raise ValueError(f"Unknown optimizer: {opt_name}")
    
    def _get_scheduler(self):
        """Get learning rate scheduler"""
        scheduler_name = self.config.get('scheduler', 'cosine').lower()
        
        if scheduler_name == 'cosine':
            return CosineAnnealingLR(
                self.optimizer,
                T_max=self.config['epochs'],
                eta_min=1e-6
            )
        elif scheduler_name == 'plateau':
            return ReduceLROnPlateau(
                self.optimizer,
                mode='min',
                factor=0.5,
                patience=5,
                min_lr=1e-6
            )
        else:
            return None
    
    def train_epoch(self, epoch: int) -> float:
        """Train for one epoch"""
        self.model.train()
        
        total_loss = 0
        num_batches = 0
        
        train_loader = self.data_module.train_dataloader()
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{self.config['epochs']}")
        
        for batch_idx, batch in enumerate(pbar):
            inputs = batch['input'].to(self.device)
            targets = batch['target'].to(self.device)
            
            # Forward pass
            if self.use_amp:
                with autocast():
                    if self.config['model_name'] == 'informer':
                        outputs = self.model(inputs)
                    else:
                        outputs = self.model(inputs)
                    loss = self.criterion(outputs, targets)
            else:
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)
            
            # Backward pass
            self.optimizer.zero_grad()
            
            if self.use_amp:
                self.scaler.scale(loss).backward()
                
                # Gradient clipping
                if self.config.get('gradient_clip', 0) > 0:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.config['gradient_clip']
                    )
                
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss.backward()
                
                if self.config.get('gradient_clip', 0) > 0:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.config['gradient_clip']
                    )
                
                self.optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
            
            # Update progress bar
            pbar.set_postfix({'Loss': f'{loss.item():.4f}'})
            
            # Log to wandb
            if self.config.get('use_wandb', False) and batch_idx % 100 == 0:
                wandb.log({
                    'train_loss': loss.item(),
                    'learning_rate': self.optimizer.param_groups[0]['lr']
                })
        
        avg_loss = total_loss / num_batches
        return avg_loss
    
    def validate(self) -> Tuple[float, Dict]:
        """Validate the model"""
        self.model.eval()
        
        total_loss = 0
        all_predictions = []
        all_targets = []
        num_batches = 0
        
        val_loader = self.data_module.val_dataloader()
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Validation"):
                inputs = batch['input'].to(self.device)
                targets = batch['target'].to(self.device)
                
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)
                
                total_loss += loss.item()
                num_batches += 1
                
                all_predictions.append(outputs.cpu().numpy())
                all_targets.append(targets.cpu().numpy())
        
        avg_loss = total_loss / num_batches
        
        # Concatenate predictions and targets
        all_predictions = np.concatenate(all_predictions, axis=0)
        all_targets = np.concatenate(all_targets, axis=0)
        
        # Inverse transform if scaler is available
        if self.data_module.scaler:
            all_predictions = inverse_transform_predictions(
                all_predictions,
                self.data_module.scaler
            )
            all_targets = inverse_transform_predictions(
                all_targets,
                self.data_module.scaler
            )
        
        # Calculate metrics
        metrics = calculate_metrics(all_predictions, all_targets)
        
        return avg_loss, metrics
    
    def train(self):
        """Main training loop"""
        print("\nStarting training...")
        
        for epoch in range(1, self.config['epochs'] + 1):
            # Training
            train_loss = self.train_epoch(epoch)
            self.train_losses.append(train_loss)
            
            # Validation
            val_loss, val_metrics = self.validate()
            self.val_losses.append(val_loss)
            self.val_maes.append(val_metrics['mae'])
            self.val_mapes.append(val_metrics['mape'])
            
            # Update scheduler
            if self.scheduler:
                if isinstance(self.scheduler, ReduceLROnPlateau):
                    self.scheduler.step(val_loss)
                else:
                    self.scheduler.step()
            
            # Print epoch summary
            print(f"\nEpoch {epoch}/{self.config['epochs']}:")
            print(f"Train Loss: {train_loss:.4f}")
            print(f"Val Loss: {val_loss:.4f}")
            print(f"Val MAE: {val_metrics['mae']:.4f}")
            print(f"Val MAPE: {val_metrics['mape']:.2f}%")
            print(f"Val RMSE: {val_metrics['rmse']:.4f}")
            print(f"LR: {self.optimizer.param_groups[0]['lr']:.6f}")
            
            # Log to wandb
            if self.config.get('use_wandb', False):
                wandb.log({
                    'epoch': epoch,
                    'train_loss': train_loss,
                    'val_loss': val_loss,
                    'val_mae': val_metrics['mae'],
                    'val_mape': val_metrics['mape'],
                    'val_rmse': val_metrics['rmse'],
                    'learning_rate': self.optimizer.param_groups[0]['lr']
                })
            
            # Save best model
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.save_checkpoint(epoch, is_best=True)
            
            # Early stopping
            if self.early_stopping(val_loss):
                print(f"\nEarly stopping triggered at epoch {epoch}")
                break
        
        # Save training curves
        plot_training_curves(
            self.train_losses,
            self.val_losses,
            self.val_maes,
            self.val_mapes,
            save_path=os.path.join(self.config['checkpoint_dir'], 'training_curves.png')
        )
        
        print(f"\nTraining completed. Best validation loss: {self.best_val_loss:.4f}")
    
    def test(self):
        """Test the model"""
        print("\nTesting model...")
        
        # Load best model
        checkpoint_path = os.path.join(self.config['checkpoint_dir'], 'best_model.pth')
        if os.path.exists(checkpoint_path):
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            print(f"Loaded best model from epoch {checkpoint['epoch']}")
        
        self.model.eval()
        
        all_predictions = []
        all_targets = []
        
        test_loader = self.data_module.test_dataloader()
        
        with torch.no_grad():
            for batch in tqdm(test_loader, desc="Testing"):
                inputs = batch['input'].to(self.device)
                targets = batch['target'].to(self.device)
                
                outputs = self.model(inputs)
                
                all_predictions.append(outputs.cpu().numpy())
                all_targets.append(targets.cpu().numpy())
        
        # Concatenate
        all_predictions = np.concatenate(all_predictions, axis=0)
        all_targets = np.concatenate(all_targets, axis=0)
        
        # Inverse transform
        if self.data_module.scaler:
            all_predictions = inverse_transform_predictions(
                all_predictions,
                self.data_module.scaler
            )
            all_targets = inverse_transform_predictions(
                all_targets,
                self.data_module.scaler
            )
        
        # Calculate metrics
        test_metrics = calculate_metrics(all_predictions, all_targets)
        
        print(f"\nTest Results:")
        print(f"MAE: {test_metrics['mae']:.4f}")
        print(f"MAPE: {test_metrics['mape']:.2f}%")
        print(f"RMSE: {test_metrics['rmse']:.4f}")
        print(f"R2 Score: {test_metrics['r2']:.4f}")
        
        # Plot sample predictions
        plot_predictions(
            all_targets[:5],
            all_predictions[:5],
            save_path=os.path.join(self.config['checkpoint_dir'], 'test_predictions.png')
        )
        
        # Save test results
        results = {
            'test_metrics': test_metrics,
            'config': self.config
        }
        
        with open(os.path.join(self.config['checkpoint_dir'], 'test_results.json'), 'w') as f:
            json.dump(results, f, indent=4)
    
    def forecast(self, input_sequence: torch.Tensor, n_steps: Optional[int] = None) -> np.ndarray:
        """Generate forecast for given input sequence"""
        self.model.eval()
        
        if n_steps is None:
            n_steps = self.config['prediction_length']
        
        with torch.no_grad():
            input_seq = input_sequence.to(self.device)
            if input_seq.dim() == 2:
                input_seq = input_seq.unsqueeze(0)
            
            predictions = self.model(input_seq)
            predictions = predictions.cpu().numpy()
            
            # Inverse transform if needed
            if self.data_module.scaler:
                predictions = inverse_transform_predictions(
                    predictions,
                    self.data_module.scaler
                )
        
        return predictions
    
    def save_checkpoint(self, epoch: int, is_best: bool = False):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_val_loss': self.best_val_loss,
            'config': self.config
        }
        
        os.makedirs(self.config['checkpoint_dir'], exist_ok=True)
        
        # Save regular checkpoint
        checkpoint_path = os.path.join(
            self.config['checkpoint_dir'],
            f'checkpoint_epoch_{epoch}.pth'
        )
        torch.save(checkpoint, checkpoint_path)
        
        # Save best model
        if is_best:
            best_path = os.path.join(self.config['checkpoint_dir'], 'best_model.pth')
            torch.save(checkpoint, best_path)
            print(f"Saved best model with loss: {self.best_val_loss:.4f}")


class QuantileLoss(nn.Module):
    """Quantile loss for probabilistic forecasting"""
    
    def __init__(self, quantiles: List[float] = [0.1, 0.5, 0.9]):
        super().__init__()
        self.quantiles = quantiles
    
    def forward(self, predictions, targets):
        losses = []
        for i, q in enumerate(self.quantiles):
            errors = targets - predictions[..., i]
            losses.append(torch.max((q - 1) * errors, q * errors).mean())
        return torch.stack(losses).mean()


def main():
    parser = argparse.ArgumentParser(description='Time Series Forecasting Training')
    parser.add_argument('--config', type=str, default='configs/default.json',
                        help='Path to config file')
    parser.add_argument('--dataset', type=str, default='synthetic',
                        help='Dataset name')
    parser.add_argument('--model', type=str, default='lstm',
                        help='Model architecture')
    parser.add_argument('--epochs', type=int, default=50,
                        help='Number of epochs')
    parser.add_argument('--batch-size', type=int, default=32,
                        help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-3,
                        help='Learning rate')
    parser.add_argument('--seq-len', type=int, default=24,
                        help='Sequence length')
    parser.add_argument('--pred-len', type=int, default=12,
                        help='Prediction length')
    args = parser.parse_args()
    
    # Load config
    if os.path.exists(args.config):
        with open(args.config, 'r') as f:
            config = json.load(f)
    else:
        config = {}
    
    # Override with command line arguments
    config.update({
        'dataset': args.dataset,
        'model_name': args.model,
        'epochs': args.epochs,
        'batch_size': args.batch_size,
        'learning_rate': args.lr,
        'sequence_length': args.seq_len,
        'prediction_length': args.pred_len,
        'data_path': config.get('data_path'),
        'checkpoint_dir': config.get('checkpoint_dir', './checkpoints'),
        'hidden_dim': config.get('hidden_dim', 128),
        'num_layers': config.get('num_layers', 2),
        'optimizer': config.get('optimizer', 'adam'),
        'scheduler': config.get('scheduler', 'cosine'),
        'loss': config.get('loss', 'mse'),
        'dropout': config.get('dropout', 0.2),
        'weight_decay': config.get('weight_decay', 1e-4),
        'gradient_clip': config.get('gradient_clip', 1.0),
        'use_amp': config.get('use_amp', True),
        'early_stopping_patience': config.get('early_stopping_patience', 10),
        'num_workers': config.get('num_workers', 4),
        'use_wandb': config.get('use_wandb', False)
    })
    
    # Create trainer and train
    trainer = TimeSeriesTrainer(config)
    trainer.train()
    trainer.test()


if __name__ == "__main__":
    main()