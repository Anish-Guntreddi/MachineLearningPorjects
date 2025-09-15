"""
Training script for Anomaly Detection
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
from typing import Dict, List, Tuple, Optional
import wandb
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score

from data_loader import AnomalyDataModule
from models import get_model
from utils import (
    calculate_metrics, plot_roc_curve, plot_training_curves,
    EarlyStopping, plot_anomaly_scores, visualize_reconstruction
)


class AnomalyTrainer:
    """Trainer for anomaly detection models"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        
        # Initialize data module
        self.data_module = AnomalyDataModule(
            dataset_name=config['dataset'],
            data_path=config.get('data_path'),
            batch_size=config['batch_size'],
            window_size=config.get('window_size'),
            num_workers=config.get('num_workers', 4)
        )
        self.data_module.setup()
        
        # Get input dimension
        sample_batch = next(iter(self.data_module.train_dataloader()))
        self.input_dim = sample_batch['data'].shape[-1]
        
        # Initialize model
        self.model = get_model(
            config['model_name'],
            input_dim=self.input_dim,
            encoding_dim=config.get('encoding_dim', 32),
            hidden_dims=config.get('hidden_dims', [128, 64]),
            dropout=config.get('dropout', 0.2)
        ).to(self.device)
        
        print(f"Model: {config['model_name']}")
        print(f"Parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        print(f"Input dim: {self.input_dim}")
        
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
        self.val_aucs = []
        self.best_val_auc = 0.0
        self.anomaly_threshold = None
        
        # Initialize wandb if enabled
        if config.get('use_wandb', False):
            wandb.init(
                project="anomaly-detection",
                config=config,
                name=f"{config['model_name']}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            )
    
    def _get_loss_function(self):
        """Get loss function based on model type"""
        model_name = self.config['model_name'].lower()
        
        if model_name == 'vae':
            return VAELoss()
        elif model_name == 'deep_svdd':
            return DeepSVDDLoss()
        elif model_name == 'ganomaly':
            return GANomalyLoss()
        else:
            # Default reconstruction loss
            return nn.MSELoss()
    
    def _get_optimizer(self):
        """Get optimizer"""
        opt_name = self.config.get('optimizer', 'adam').lower()
        lr = self.config['learning_rate']
        weight_decay = self.config.get('weight_decay', 1e-5)
        
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
            data = batch['data'].to(self.device)
            labels = batch['label'].to(self.device)
            
            # Filter to use only normal data for training (semi-supervised)
            if self.config.get('semi_supervised', True):
                normal_mask = labels == 0
                if normal_mask.sum() == 0:
                    continue
                data = data[normal_mask]
            
            # Forward pass
            if self.use_amp:
                with autocast():
                    loss = self._compute_loss(data)
            else:
                loss = self._compute_loss(data)
            
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
        
        avg_loss = total_loss / num_batches if num_batches > 0 else 0
        return avg_loss
    
    def _compute_loss(self, data):
        """Compute loss based on model type"""
        model_name = self.config['model_name'].lower()
        
        if model_name == 'vae':
            reconstructed, mu, log_var = self.model(data)
            loss = self.criterion(reconstructed, data, mu, log_var)
        elif model_name == 'deep_svdd':
            z, dist = self.model(data)
            loss = torch.mean(dist)
        elif model_name == 'ganomaly':
            x_hat, z, z_hat, real_score, fake_score = self.model(data)
            loss = self.criterion(data, x_hat, z, z_hat, real_score, fake_score)
        elif model_name == 'lstm_ae':
            # Handle sequence data
            if data.dim() == 2:
                # Add time dimension if not present
                data = data.unsqueeze(1)
            reconstructed = self.model(data)
            loss = self.criterion(reconstructed, data)
        else:
            # Standard autoencoder
            reconstructed = self.model(data)
            loss = self.criterion(reconstructed, data)
        
        return loss
    
    def validate(self) -> Tuple[float, float]:
        """Validate the model"""
        self.model.eval()
        
        total_loss = 0
        all_scores = []
        all_labels = []
        num_batches = 0
        
        val_loader = self.data_module.val_dataloader()
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Validation"):
                data = batch['data'].to(self.device)
                labels = batch['label'].to(self.device)
                
                # Compute anomaly scores
                scores = self._compute_anomaly_scores(data)
                
                # Compute loss (on normal data only)
                normal_mask = labels == 0
                if normal_mask.sum() > 0:
                    loss = self._compute_loss(data[normal_mask])
                    total_loss += loss.item()
                    num_batches += 1
                
                all_scores.extend(scores.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        avg_loss = total_loss / num_batches if num_batches > 0 else 0
        
        # Calculate AUC
        all_scores = np.array(all_scores)
        all_labels = np.array(all_labels)
        
        if len(np.unique(all_labels)) > 1:
            auc = roc_auc_score(all_labels, all_scores)
        else:
            auc = 0.5
        
        return avg_loss, auc
    
    def _compute_anomaly_scores(self, data):
        """Compute anomaly scores based on model type"""
        model_name = self.config['model_name'].lower()
        
        if model_name == 'vae':
            reconstructed, mu, log_var = self.model(data)
            scores = torch.mean((data - reconstructed) ** 2, dim=tuple(range(1, data.dim())))
        elif model_name == 'deep_svdd':
            scores = self.model.get_anomaly_score(data)
        elif model_name == 'ganomaly':
            scores = self.model.get_anomaly_score(data)
        elif model_name == 'lstm_ae':
            if data.dim() == 2:
                data = data.unsqueeze(1)
            reconstructed = self.model(data)
            scores = torch.mean((data - reconstructed) ** 2, dim=tuple(range(1, data.dim())))
        else:
            # Standard autoencoder reconstruction error
            reconstructed = self.model(data)
            scores = torch.mean((data - reconstructed) ** 2, dim=tuple(range(1, data.dim())))
        
        return scores
    
    def train(self):
        """Main training loop"""
        print("\nStarting training...")
        
        for epoch in range(1, self.config['epochs'] + 1):
            # Training
            train_loss = self.train_epoch(epoch)
            self.train_losses.append(train_loss)
            
            # Validation
            val_loss, val_auc = self.validate()
            self.val_losses.append(val_loss)
            self.val_aucs.append(val_auc)
            
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
            print(f"Val AUC: {val_auc:.4f}")
            print(f"LR: {self.optimizer.param_groups[0]['lr']:.6f}")
            
            # Log to wandb
            if self.config.get('use_wandb', False):
                wandb.log({
                    'epoch': epoch,
                    'train_loss': train_loss,
                    'val_loss': val_loss,
                    'val_auc': val_auc,
                    'learning_rate': self.optimizer.param_groups[0]['lr']
                })
            
            # Save best model
            if val_auc > self.best_val_auc:
                self.best_val_auc = val_auc
                self.save_checkpoint(epoch, is_best=True)
            
            # Early stopping
            if self.early_stopping(val_loss):
                print(f"\nEarly stopping triggered at epoch {epoch}")
                break
        
        # Set anomaly threshold
        self._set_anomaly_threshold()
        
        # Save training curves
        plot_training_curves(
            self.train_losses,
            self.val_losses,
            self.val_aucs,
            save_path=os.path.join(self.config['checkpoint_dir'], 'training_curves.png')
        )
        
        print(f"\nTraining completed. Best validation AUC: {self.best_val_auc:.4f}")
    
    def _set_anomaly_threshold(self):
        """Set anomaly threshold based on validation data"""
        self.model.eval()
        
        all_scores = []
        all_labels = []
        
        val_loader = self.data_module.val_dataloader()
        
        with torch.no_grad():
            for batch in val_loader:
                data = batch['data'].to(self.device)
                labels = batch['label']
                
                scores = self._compute_anomaly_scores(data)
                
                all_scores.extend(scores.cpu().numpy())
                all_labels.extend(labels.numpy())
        
        all_scores = np.array(all_scores)
        all_labels = np.array(all_labels)
        
        # Find threshold that maximizes F1 score
        best_f1 = 0
        best_threshold = 0
        
        for percentile in range(80, 100):
            threshold = np.percentile(all_scores, percentile)
            predictions = (all_scores > threshold).astype(int)
            
            if len(np.unique(all_labels)) > 1:
                f1 = f1_score(all_labels, predictions)
                if f1 > best_f1:
                    best_f1 = f1
                    best_threshold = threshold
        
        self.anomaly_threshold = best_threshold
        print(f"Anomaly threshold set to: {self.anomaly_threshold:.4f}")
    
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
        
        all_scores = []
        all_labels = []
        all_data = []
        
        test_loader = self.data_module.test_dataloader()
        
        with torch.no_grad():
            for batch in tqdm(test_loader, desc="Testing"):
                data = batch['data'].to(self.device)
                labels = batch['label']
                
                scores = self._compute_anomaly_scores(data)
                
                all_scores.extend(scores.cpu().numpy())
                all_labels.extend(labels.numpy())
                all_data.append(data.cpu().numpy())
        
        all_scores = np.array(all_scores)
        all_labels = np.array(all_labels)
        all_data = np.concatenate(all_data, axis=0)
        
        # Calculate metrics
        metrics = calculate_metrics(all_scores, all_labels, self.anomaly_threshold)
        
        print(f"\nTest Results:")
        print(f"AUC-ROC: {metrics['auc_roc']:.4f}")
        print(f"AUC-PR: {metrics['auc_pr']:.4f}")
        print(f"F1 Score: {metrics['f1']:.4f}")
        print(f"Precision: {metrics['precision']:.4f}")
        print(f"Recall: {metrics['recall']:.4f}")
        
        # Plot ROC curve
        plot_roc_curve(
            all_labels,
            all_scores,
            save_path=os.path.join(self.config['checkpoint_dir'], 'roc_curve.png')
        )
        
        # Plot anomaly score distribution
        plot_anomaly_scores(
            all_scores,
            all_labels,
            threshold=self.anomaly_threshold,
            save_path=os.path.join(self.config['checkpoint_dir'], 'anomaly_scores.png')
        )
        
        # Visualize reconstructions for autoencoders
        if 'autoencoder' in self.config['model_name'].lower() or 'ae' in self.config['model_name'].lower():
            visualize_reconstruction(
                self.model,
                all_data[:5],
                self.device,
                save_path=os.path.join(self.config['checkpoint_dir'], 'reconstructions.png')
            )
        
        # Save test results
        results = {
            'test_metrics': metrics,
            'anomaly_threshold': float(self.anomaly_threshold) if self.anomaly_threshold else None,
            'config': self.config
        }
        
        with open(os.path.join(self.config['checkpoint_dir'], 'test_results.json'), 'w') as f:
            json.dump(results, f, indent=4)
    
    def save_checkpoint(self, epoch: int, is_best: bool = False):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_val_auc': self.best_val_auc,
            'anomaly_threshold': self.anomaly_threshold,
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
            print(f"Saved best model with AUC: {self.best_val_auc:.4f}")


class VAELoss(nn.Module):
    """VAE loss with reconstruction and KL divergence"""
    
    def forward(self, reconstructed, original, mu, log_var):
        reconstruction_loss = F.mse_loss(reconstructed, original, reduction='sum')
        kl_divergence = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
        return (reconstruction_loss + kl_divergence) / original.size(0)


class DeepSVDDLoss(nn.Module):
    """Deep SVDD loss"""
    
    def forward(self, distances):
        return torch.mean(distances)


class GANomalyLoss(nn.Module):
    """GANomaly loss"""
    
    def __init__(self, w_adv: float = 1.0, w_con: float = 50.0, w_lat: float = 1.0):
        super().__init__()
        self.w_adv = w_adv
        self.w_con = w_con
        self.w_lat = w_lat
    
    def forward(self, x, x_hat, z, z_hat, real_score, fake_score):
        # Adversarial loss
        adv_loss = F.binary_cross_entropy(real_score, torch.ones_like(real_score)) + \
                   F.binary_cross_entropy(fake_score, torch.zeros_like(fake_score))
        
        # Contextual loss (reconstruction)
        con_loss = F.mse_loss(x_hat, x)
        
        # Latent loss
        lat_loss = F.mse_loss(z_hat, z)
        
        total_loss = self.w_adv * adv_loss + self.w_con * con_loss + self.w_lat * lat_loss
        
        return total_loss


def main():
    parser = argparse.ArgumentParser(description='Anomaly Detection Training')
    parser.add_argument('--config', type=str, default='configs/default.json',
                        help='Path to config file')
    parser.add_argument('--dataset', type=str, default='synthetic',
                        help='Dataset name')
    parser.add_argument('--model', type=str, default='autoencoder',
                        help='Model architecture')
    parser.add_argument('--epochs', type=int, default=50,
                        help='Number of epochs')
    parser.add_argument('--batch-size', type=int, default=32,
                        help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-3,
                        help='Learning rate')
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
        'data_path': config.get('data_path'),
        'checkpoint_dir': config.get('checkpoint_dir', './checkpoints'),
        'encoding_dim': config.get('encoding_dim', 32),
        'hidden_dims': config.get('hidden_dims', [128, 64]),
        'window_size': config.get('window_size'),
        'semi_supervised': config.get('semi_supervised', True),
        'optimizer': config.get('optimizer', 'adam'),
        'scheduler': config.get('scheduler', 'cosine'),
        'dropout': config.get('dropout', 0.2),
        'weight_decay': config.get('weight_decay', 1e-5),
        'gradient_clip': config.get('gradient_clip', 1.0),
        'use_amp': config.get('use_amp', True),
        'early_stopping_patience': config.get('early_stopping_patience', 10),
        'num_workers': config.get('num_workers', 4),
        'use_wandb': config.get('use_wandb', False)
    })
    
    # Create trainer and train
    trainer = AnomalyTrainer(config)
    trainer.train()
    trainer.test()


if __name__ == "__main__":
    main()