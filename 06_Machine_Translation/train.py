"""
Training script for machine translation
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.cuda.amp import GradScaler, autocast
import numpy as np
from tqdm import tqdm
import argparse
import os
import json
from datetime import datetime
import sacrebleu
from typing import Dict, List, Tuple

from models import get_model


class TranslationTrainer:
    """Machine translation trainer"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        
        # Initialize model
        self.model = get_model(
            config['model_name'],
            src_vocab_size=config.get('src_vocab_size', 32000),
            tgt_vocab_size=config.get('tgt_vocab_size', 32000),
            d_model=config.get('d_model', 512),
            nhead=config.get('nhead', 8),
            num_encoder_layers=config.get('num_encoder_layers', 6),
            num_decoder_layers=config.get('num_decoder_layers', 6)
        ).to(self.device)
        
        # Loss function (ignore padding token)
        self.criterion = nn.CrossEntropyLoss(
            ignore_index=config.get('pad_token_id', 0),
            label_smoothing=config.get('label_smoothing', 0.1)
        )
        
        # Optimizer
        self.optimizer = self._get_optimizer()
        
        # Scheduler
        self.scheduler = ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=0.5,
            patience=5,
            min_lr=1e-6
        )
        
        # Mixed precision
        self.use_amp = config.get('use_amp', True) and torch.cuda.is_available()
        self.scaler = GradScaler() if self.use_amp else None
        
        # Training state
        self.start_epoch = 0
        self.best_bleu = 0.0
    
    def _get_optimizer(self):
        """Get optimizer"""
        opt_name = self.config.get('optimizer', 'adam').lower()
        lr = self.config['learning_rate']
        weight_decay = self.config.get('weight_decay', 1e-4)
        
        if opt_name == 'adam':
            return optim.Adam(
                self.model.parameters(),
                lr=lr,
                betas=(0.9, 0.98),
                eps=1e-9,
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
    
    def train_epoch(self, epoch: int) -> Dict:
        """Train for one epoch"""
        self.model.train()
        
        total_loss = 0
        num_batches = 100  # Dummy number for demonstration
        
        pbar = tqdm(range(num_batches), desc=f"Epoch {epoch}/{self.config['epochs']}")
        
        for batch_idx in pbar:
            # Generate dummy data for demonstration
            src = torch.randint(1, 1000, (8, 20)).to(self.device)
            tgt_input = torch.randint(1, 1000, (8, 15)).to(self.device)
            tgt_output = torch.randint(1, 1000, (8, 15)).to(self.device)
            
            # Forward pass
            if self.use_amp:
                with autocast():
                    output = self.model(src, tgt_input)
                    loss = self.criterion(
                        output.reshape(-1, output.size(-1)),
                        tgt_output.reshape(-1)
                    )
            else:
                output = self.model(src, tgt_input)
                loss = self.criterion(
                    output.reshape(-1, output.size(-1)),
                    tgt_output.reshape(-1)
                )
            
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
            
            # Update progress bar
            pbar.set_postfix({'Loss': f'{loss.item():.4f}'})
        
        return {'loss': total_loss / num_batches}
    
    def validate(self) -> Dict:
        """Validate the model"""
        self.model.eval()
        
        total_loss = 0
        num_batches = 20  # Dummy number
        
        with torch.no_grad():
            for _ in range(num_batches):
                # Generate dummy data
                src = torch.randint(1, 1000, (8, 20)).to(self.device)
                tgt_input = torch.randint(1, 1000, (8, 15)).to(self.device)
                tgt_output = torch.randint(1, 1000, (8, 15)).to(self.device)
                
                output = self.model(src, tgt_input)
                loss = self.criterion(
                    output.reshape(-1, output.size(-1)),
                    tgt_output.reshape(-1)
                )
                
                total_loss += loss.item()
        
        # Calculate dummy BLEU score
        bleu_score = np.random.uniform(20, 40)  # Dummy BLEU
        
        return {
            'loss': total_loss / num_batches,
            'bleu': bleu_score
        }
    
    def train(self):
        """Main training loop"""
        for epoch in range(self.start_epoch, self.config['epochs']):
            # Training
            train_metrics = self.train_epoch(epoch)
            
            # Validation
            val_metrics = self.validate()
            
            # Update scheduler
            self.scheduler.step(val_metrics['loss'])
            
            # Log metrics
            print(f"\nEpoch {epoch}/{self.config['epochs']}:")
            print(f"Train Loss: {train_metrics['loss']:.4f}")
            print(f"Val Loss: {val_metrics['loss']:.4f}")
            print(f"Val BLEU: {val_metrics['bleu']:.2f}")
            
            # Save checkpoint
            is_best = val_metrics['bleu'] > self.best_bleu
            if is_best:
                self.best_bleu = val_metrics['bleu']
                self.save_checkpoint(epoch, is_best)
    
    def save_checkpoint(self, epoch: int, is_best: bool):
        """Save checkpoint"""
        checkpoint = {
            'epoch': epoch + 1,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_bleu': self.best_bleu,
            'config': self.config
        }
        
        checkpoint_dir = self.config.get('checkpoint_dir', './checkpoints')
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        path = os.path.join(checkpoint_dir, 'checkpoint.pth')
        torch.save(checkpoint, path)
        
        if is_best:
            best_path = os.path.join(checkpoint_dir, 'best_model.pth')
            torch.save(checkpoint, best_path)


def main():
    parser = argparse.ArgumentParser(description='Machine Translation Training')
    parser.add_argument('--model', type=str, default='transformer',
                        help='Model architecture')
    parser.add_argument('--epochs', type=int, default=30,
                        help='Number of epochs')
    parser.add_argument('--batch-size', type=int, default=32,
                        help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='Learning rate')
    args = parser.parse_args()
    
    config = {
        'model_name': args.model,
        'epochs': args.epochs,
        'batch_size': args.batch_size,
        'learning_rate': args.lr,
        'src_vocab_size': 32000,
        'tgt_vocab_size': 32000,
        'd_model': 512,
        'nhead': 8,
        'num_encoder_layers': 6,
        'num_decoder_layers': 6,
        'optimizer': 'adam',
        'weight_decay': 1e-4,
        'gradient_clip': 1.0,
        'label_smoothing': 0.1,
        'use_amp': True,
        'checkpoint_dir': './checkpoints'
    }
    
    trainer = TranslationTrainer(config)
    trainer.train()


if __name__ == "__main__":
    main()