"""
Training script for Recommender Systems
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
from sklearn.metrics import mean_squared_error, mean_absolute_error

from data_loader import RecommenderDataModule
from models import get_model
from utils import evaluate_metrics, EarlyStopping, save_training_curves


class RecommenderTrainer:
    """Trainer for recommender system models"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        
        # Initialize data module
        self.data_module = RecommenderDataModule(
            dataset_name=config['dataset'],
            data_path=config['data_path'],
            batch_size=config['batch_size'],
            negative_sampling=config.get('negative_sampling', True),
            num_negatives=config.get('num_negatives', 4),
            num_workers=config.get('num_workers', 4)
        )
        self.data_module.setup()
        
        # Initialize model
        self.model = get_model(
            config['model_name'],
            num_users=self.data_module.num_users,
            num_items=self.data_module.num_items,
            embedding_dim=config.get('embedding_dim', 64),
            dropout=config.get('dropout', 0.2)
        ).to(self.device)
        
        print(f"Model: {config['model_name']}")
        print(f"Parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        print(f"Users: {self.data_module.num_users}, Items: {self.data_module.num_items}")
        
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
            delta=0.001
        )
        
        # Tracking
        self.train_losses = []
        self.val_losses = []
        self.val_rmses = []
        self.val_maes = []
        self.best_val_rmse = float('inf')
        
        # Initialize wandb if enabled
        if config.get('use_wandb', False):
            wandb.init(
                project="recommender-systems",
                config=config,
                name=f"{config['model_name']}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            )
    
    def _get_loss_function(self):
        """Get loss function based on task type"""
        task_type = self.config.get('task_type', 'rating')
        
        if task_type == 'rating':
            return nn.MSELoss()
        elif task_type == 'ranking':
            return nn.BCEWithLogitsLoss()
        else:
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
            users = batch['user'].to(self.device)
            items = batch['item'].to(self.device)
            
            if 'rating' in batch:
                targets = batch['rating'].to(self.device)
            else:
                targets = batch['label'].to(self.device)
            
            # Forward pass
            if self.use_amp:
                with autocast():
                    outputs = self.model(users, items)
                    loss = self.criterion(outputs, targets)
            else:
                outputs = self.model(users, items)
                loss = self.criterion(outputs, targets)
            
            # Add regularization if specified
            if self.config.get('l2_reg', 0) > 0:
                l2_reg = self.config['l2_reg']
                l2_loss = 0
                for param in self.model.parameters():
                    l2_loss += torch.norm(param, 2) ** 2
                loss += l2_reg * l2_loss
            
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
    
    def validate(self) -> Tuple[float, float, float, Dict]:
        """Validate the model"""
        self.model.eval()
        
        all_predictions = []
        all_targets = []
        total_loss = 0
        num_batches = 0
        
        val_loader = self.data_module.val_dataloader()
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Validation"):
                users = batch['user'].to(self.device)
                items = batch['item'].to(self.device)
                targets = batch['rating'].to(self.device)
                
                outputs = self.model(users, items)
                loss = self.criterion(outputs, targets)
                
                total_loss += loss.item()
                num_batches += 1
                
                all_predictions.extend(outputs.cpu().numpy())
                all_targets.extend(targets.cpu().numpy())
        
        avg_loss = total_loss / num_batches
        
        # Calculate metrics
        all_predictions = np.array(all_predictions)
        all_targets = np.array(all_targets)
        
        # Clip predictions to valid rating range
        if self.config.get('clip_predictions', True):
            all_predictions = np.clip(all_predictions, 1, 5)
        
        rmse = np.sqrt(mean_squared_error(all_targets, all_predictions))
        mae = mean_absolute_error(all_targets, all_predictions)
        
        return avg_loss, rmse, mae, {
            'predictions': all_predictions[:100],  # Sample for analysis
            'targets': all_targets[:100]
        }
    
    def train(self):
        """Main training loop"""
        print("\nStarting training...")
        
        for epoch in range(1, self.config['epochs'] + 1):
            # Training
            train_loss = self.train_epoch(epoch)
            self.train_losses.append(train_loss)
            
            # Validation
            val_loss, val_rmse, val_mae, val_metrics = self.validate()
            self.val_losses.append(val_loss)
            self.val_rmses.append(val_rmse)
            self.val_maes.append(val_mae)
            
            # Update scheduler
            if self.scheduler:
                if isinstance(self.scheduler, ReduceLROnPlateau):
                    self.scheduler.step(val_rmse)
                else:
                    self.scheduler.step()
            
            # Print epoch summary
            print(f"\nEpoch {epoch}/{self.config['epochs']}:")
            print(f"Train Loss: {train_loss:.4f}")
            print(f"Val Loss: {val_loss:.4f}, Val RMSE: {val_rmse:.4f}, Val MAE: {val_mae:.4f}")
            print(f"LR: {self.optimizer.param_groups[0]['lr']:.6f}")
            
            # Log to wandb
            if self.config.get('use_wandb', False):
                wandb.log({
                    'epoch': epoch,
                    'train_loss': train_loss,
                    'val_loss': val_loss,
                    'val_rmse': val_rmse,
                    'val_mae': val_mae,
                    'learning_rate': self.optimizer.param_groups[0]['lr']
                })
            
            # Save best model
            if val_rmse < self.best_val_rmse:
                self.best_val_rmse = val_rmse
                self.save_checkpoint(epoch, is_best=True)
            
            # Early stopping
            if self.early_stopping(val_rmse):
                print(f"\nEarly stopping triggered at epoch {epoch}")
                break
        
        # Save training curves
        save_training_curves(
            self.train_losses,
            self.val_losses,
            self.val_rmses,
            self.val_maes,
            save_path=os.path.join(self.config['checkpoint_dir'], 'training_curves.png')
        )
        
        print(f"\nTraining completed. Best validation RMSE: {self.best_val_rmse:.4f}")
    
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
        all_users = []
        all_items = []
        
        test_loader = self.data_module.test_dataloader()
        
        with torch.no_grad():
            for batch in tqdm(test_loader, desc="Testing"):
                users = batch['user'].to(self.device)
                items = batch['item'].to(self.device)
                targets = batch['rating'].to(self.device)
                
                outputs = self.model(users, items)
                
                all_predictions.extend(outputs.cpu().numpy())
                all_targets.extend(targets.cpu().numpy())
                all_users.extend(users.cpu().numpy())
                all_items.extend(items.cpu().numpy())
        
        # Calculate metrics
        all_predictions = np.array(all_predictions)
        all_targets = np.array(all_targets)
        
        if self.config.get('clip_predictions', True):
            all_predictions = np.clip(all_predictions, 1, 5)
        
        # Calculate overall metrics
        test_rmse = np.sqrt(mean_squared_error(all_targets, all_predictions))
        test_mae = mean_absolute_error(all_targets, all_predictions)
        
        print(f"\nTest RMSE: {test_rmse:.4f}")
        print(f"Test MAE: {test_mae:.4f}")
        
        # Calculate per-user metrics
        user_metrics = evaluate_metrics(
            all_predictions,
            all_targets,
            all_users,
            all_items
        )
        
        print(f"\nPer-user metrics:")
        print(f"Average user RMSE: {user_metrics['avg_user_rmse']:.4f}")
        print(f"Average user MAE: {user_metrics['avg_user_mae']:.4f}")
        
        # Save test results
        results = {
            'test_rmse': float(test_rmse),
            'test_mae': float(test_mae),
            'user_metrics': user_metrics,
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
            'best_val_rmse': self.best_val_rmse,
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
            print(f"Saved best model with RMSE: {self.best_val_rmse:.4f}")
    
    def generate_recommendations(self, user_id: int, top_k: int = 10) -> List[int]:
        """Generate top-k recommendations for a user"""
        self.model.eval()
        
        # Encode user ID
        user_idx = self.data_module.user_encoder.transform([user_id])[0]
        
        # Get all items
        all_items = torch.arange(self.data_module.num_items).to(self.device)
        user_tensor = torch.full_like(all_items, user_idx)
        
        with torch.no_grad():
            scores = self.model(user_tensor, all_items)
        
        # Get top-k items
        top_scores, top_indices = torch.topk(scores, top_k)
        
        # Decode item IDs
        top_items = self.data_module.item_encoder.inverse_transform(
            top_indices.cpu().numpy()
        )
        
        return top_items.tolist()


def main():
    parser = argparse.ArgumentParser(description='Recommender System Training')
    parser.add_argument('--config', type=str, default='configs/default.json',
                        help='Path to config file')
    parser.add_argument('--dataset', type=str, default='movielens',
                        help='Dataset name')
    parser.add_argument('--model', type=str, default='ncf',
                        help='Model architecture')
    parser.add_argument('--epochs', type=int, default=50,
                        help='Number of epochs')
    parser.add_argument('--batch-size', type=int, default=256,
                        help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-3,
                        help='Learning rate')
    parser.add_argument('--embedding-dim', type=int, default=64,
                        help='Embedding dimension')
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
        'embedding_dim': args.embedding_dim,
        'data_path': config.get('data_path', './data'),
        'checkpoint_dir': config.get('checkpoint_dir', './checkpoints'),
        'task_type': config.get('task_type', 'rating'),
        'negative_sampling': config.get('negative_sampling', True),
        'num_negatives': config.get('num_negatives', 4),
        'optimizer': config.get('optimizer', 'adam'),
        'scheduler': config.get('scheduler', 'cosine'),
        'dropout': config.get('dropout', 0.2),
        'weight_decay': config.get('weight_decay', 1e-5),
        'l2_reg': config.get('l2_reg', 0),
        'gradient_clip': config.get('gradient_clip', 0),
        'clip_predictions': config.get('clip_predictions', True),
        'use_amp': config.get('use_amp', True),
        'early_stopping_patience': config.get('early_stopping_patience', 10),
        'num_workers': config.get('num_workers', 4),
        'use_wandb': config.get('use_wandb', False)
    })
    
    # Create trainer and train
    trainer = RecommenderTrainer(config)
    trainer.train()
    trainer.test()
    
    # Generate sample recommendations
    print("\nGenerating sample recommendations...")
    sample_user = 1
    recommendations = trainer.generate_recommendations(sample_user, top_k=10)
    print(f"Top 10 recommendations for user {sample_user}: {recommendations}")


if __name__ == "__main__":
    main()