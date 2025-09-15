"""
Training script for Automatic Speech Recognition
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import OneCycleLR, ReduceLROnPlateau
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
from jiwer import wer, cer

from data_loader import ASRDataModule, create_char_tokenizer, decode_predictions
from models import get_model
from utils import EarlyStopping, save_training_curves, plot_attention_weights


class ASRTrainer:
    """Trainer for ASR models"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        
        # Initialize data module
        self.data_module = ASRDataModule(
            dataset_name=config['dataset'],
            data_path=config['data_path'],
            batch_size=config['batch_size'],
            num_workers=config.get('num_workers', 4)
        )
        self.data_module.setup()
        
        # Create tokenizer
        self._create_tokenizer()
        
        # Get input dimension
        input_dim = self._get_input_dim()
        
        # Initialize model
        self.model = get_model(
            config['model_name'],
            input_dim=input_dim,
            num_classes=self.tokenizer['vocab_size'],
            hidden_dim=config.get('hidden_dim', 256),
            num_layers=config.get('num_layers', 6),
            dropout=config.get('dropout', 0.1)
        ).to(self.device)
        
        print(f"Model: {config['model_name']}")
        print(f"Parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        
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
            delta=0.01
        )
        
        # Tracking
        self.train_losses = []
        self.val_losses = []
        self.train_wers = []
        self.val_wers = []
        self.best_val_wer = float('inf')
        
        # Initialize wandb if enabled
        if config.get('use_wandb', False):
            wandb.init(
                project="automatic-speech-recognition",
                config=config,
                name=f"{config['model_name']}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            )
    
    def _create_tokenizer(self):
        """Create character tokenizer from training data"""
        # Collect all transcripts
        train_loader = self.data_module.train_dataloader()
        transcripts = []
        
        for batch in train_loader:
            transcripts.extend(batch['transcripts'])
            if len(transcripts) > 1000:  # Sample for efficiency
                break
        
        self.tokenizer = create_char_tokenizer(transcripts)
        print(f"Vocabulary size: {self.tokenizer['vocab_size']}")
    
    def _get_input_dim(self) -> int:
        """Get input dimension based on feature type"""
        feature_type = self.config.get('feature_type', 'melspec')
        
        if feature_type == 'raw':
            return 1
        elif feature_type == 'melspec':
            return 80  # 80 mel bands
        elif feature_type == 'mfcc':
            return 40  # 40 MFCC coefficients
        elif feature_type == 'spectrogram':
            return 161  # Spectrogram bins
        else:
            return 80  # Default mel spectrogram
    
    def _get_loss_function(self):
        """Get loss function based on model type"""
        if self.config['model_name'] in ['deepspeech2', 'conformer', 'transformer']:
            # CTC Loss
            return nn.CTCLoss(blank=self.tokenizer['char_to_idx']['<pad>'], zero_infinity=True)
        else:
            # Cross entropy for sequence-to-sequence models
            return nn.CrossEntropyLoss(ignore_index=self.tokenizer['char_to_idx']['<pad>'])
    
    def _get_optimizer(self):
        """Get optimizer"""
        opt_name = self.config.get('optimizer', 'adamw').lower()
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
                weight_decay=weight_decay,
                nesterov=True
            )
        else:
            raise ValueError(f"Unknown optimizer: {opt_name}")
    
    def _get_scheduler(self):
        """Get learning rate scheduler"""
        scheduler_name = self.config.get('scheduler', 'onecycle').lower()
        
        if scheduler_name == 'onecycle':
            steps_per_epoch = len(self.data_module.train_dataloader())
            return OneCycleLR(
                self.optimizer,
                max_lr=self.config['learning_rate'],
                total_steps=steps_per_epoch * self.config['epochs'],
                pct_start=0.1
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
    
    def train_epoch(self, epoch: int) -> Tuple[float, float]:
        """Train for one epoch"""
        self.model.train()
        
        total_loss = 0
        all_predictions = []
        all_references = []
        
        train_loader = self.data_module.train_dataloader()
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{self.config['epochs']}")
        
        for batch_idx, batch in enumerate(pbar):
            waveforms = batch['waveforms'].to(self.device)
            transcripts = batch['transcripts']
            
            # Convert transcripts to tokens
            targets = []
            target_lengths = []
            for transcript in transcripts:
                tokens = [self.tokenizer['char_to_idx'].get(c, self.tokenizer['char_to_idx']['<pad>']) 
                         for c in transcript.lower()]
                targets.append(tokens)
                target_lengths.append(len(tokens))
            
            # Pad targets
            max_len = max(target_lengths)
            padded_targets = []
            for target in targets:
                padded = target + [self.tokenizer['char_to_idx']['<pad>']] * (max_len - len(target))
                padded_targets.append(padded)
            
            targets = torch.tensor(padded_targets, dtype=torch.long).to(self.device)
            target_lengths = torch.tensor(target_lengths, dtype=torch.long)
            
            # Forward pass
            if self.use_amp:
                with autocast():
                    if self.config['model_name'] in ['deepspeech2', 'conformer']:
                        # Models with CTC loss
                        outputs = self.model(waveforms)
                        input_lengths = torch.full((waveforms.size(0),), outputs.size(1), dtype=torch.long)
                        loss = self.criterion(
                            outputs.transpose(0, 1),  # (time, batch, classes)
                            targets,
                            input_lengths,
                            target_lengths
                        )
                    else:
                        # Sequence-to-sequence models
                        outputs = self.model(waveforms, targets)
                        loss = self.criterion(
                            outputs.reshape(-1, outputs.size(-1)),
                            targets.reshape(-1)
                        )
            else:
                if self.config['model_name'] in ['deepspeech2', 'conformer']:
                    outputs = self.model(waveforms)
                    input_lengths = torch.full((waveforms.size(0),), outputs.size(1), dtype=torch.long)
                    loss = self.criterion(
                        outputs.transpose(0, 1),
                        targets,
                        input_lengths,
                        target_lengths
                    )
                else:
                    outputs = self.model(waveforms, targets)
                    loss = self.criterion(
                        outputs.reshape(-1, outputs.size(-1)),
                        targets.reshape(-1)
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
            
            # Update scheduler if using OneCycle
            if self.scheduler and isinstance(self.scheduler, OneCycleLR):
                self.scheduler.step()
            
            # Statistics
            total_loss += loss.item()
            
            # Decode predictions for WER calculation (sample)
            if batch_idx % 10 == 0:
                with torch.no_grad():
                    if self.config['model_name'] in ['deepspeech2', 'conformer']:
                        # Greedy decoding for CTC
                        predictions = outputs.argmax(dim=-1)
                    else:
                        predictions = outputs.argmax(dim=-1)
                    
                    decoded_preds = decode_predictions(predictions, self.tokenizer)
                    all_predictions.extend(decoded_preds[:2])  # Sample for efficiency
                    all_references.extend(transcripts[:2])
            
            # Update progress bar
            pbar.set_postfix({'Loss': f'{loss.item():.4f}'})
            
            # Log to wandb
            if self.config.get('use_wandb', False) and batch_idx % 10 == 0:
                wandb.log({
                    'train_loss': loss.item(),
                    'learning_rate': self.optimizer.param_groups[0]['lr']
                })
        
        avg_loss = total_loss / len(train_loader)
        
        # Calculate WER on samples
        if all_predictions and all_references:
            avg_wer = wer(all_references, all_predictions)
        else:
            avg_wer = 1.0
        
        return avg_loss, avg_wer
    
    def validate(self) -> Tuple[float, float, Dict]:
        """Validate the model"""
        self.model.eval()
        
        total_loss = 0
        all_predictions = []
        all_references = []
        
        val_loader = self.data_module.val_dataloader()
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Validation"):
                waveforms = batch['waveforms'].to(self.device)
                transcripts = batch['transcripts']
                
                # Convert transcripts to tokens
                targets = []
                target_lengths = []
                for transcript in transcripts:
                    tokens = [self.tokenizer['char_to_idx'].get(c, self.tokenizer['char_to_idx']['<pad>']) 
                             for c in transcript.lower()]
                    targets.append(tokens)
                    target_lengths.append(len(tokens))
                
                # Pad targets
                max_len = max(target_lengths)
                padded_targets = []
                for target in targets:
                    padded = target + [self.tokenizer['char_to_idx']['<pad>']] * (max_len - len(target))
                    padded_targets.append(padded)
                
                targets = torch.tensor(padded_targets, dtype=torch.long).to(self.device)
                target_lengths = torch.tensor(target_lengths, dtype=torch.long)
                
                # Forward pass
                if self.config['model_name'] in ['deepspeech2', 'conformer']:
                    outputs = self.model(waveforms)
                    input_lengths = torch.full((waveforms.size(0),), outputs.size(1), dtype=torch.long)
                    loss = self.criterion(
                        outputs.transpose(0, 1),
                        targets,
                        input_lengths,
                        target_lengths
                    )
                    predictions = outputs.argmax(dim=-1)
                else:
                    outputs = self.model(waveforms, None)  # No teacher forcing
                    loss = self.criterion(
                        outputs.reshape(-1, outputs.size(-1)),
                        targets.reshape(-1)
                    )
                    predictions = outputs.argmax(dim=-1)
                
                total_loss += loss.item()
                
                # Decode predictions
                decoded_preds = decode_predictions(predictions, self.tokenizer)
                all_predictions.extend(decoded_preds)
                all_references.extend(transcripts)
        
        avg_loss = total_loss / len(val_loader)
        
        # Calculate metrics
        avg_wer = wer(all_references, all_predictions)
        avg_cer = cer(all_references, all_predictions)
        
        return avg_loss, avg_wer, {
            'wer': avg_wer,
            'cer': avg_cer,
            'predictions': all_predictions[:5],  # Sample predictions
            'references': all_references[:5]
        }
    
    def train(self):
        """Main training loop"""
        print("\nStarting training...")
        
        for epoch in range(1, self.config['epochs'] + 1):
            # Training
            train_loss, train_wer = self.train_epoch(epoch)
            self.train_losses.append(train_loss)
            self.train_wers.append(train_wer)
            
            # Validation
            val_loss, val_wer, val_metrics = self.validate()
            self.val_losses.append(val_loss)
            self.val_wers.append(val_wer)
            
            # Update scheduler
            if self.scheduler and isinstance(self.scheduler, ReduceLROnPlateau):
                self.scheduler.step(val_wer)
            
            # Print epoch summary
            print(f"\nEpoch {epoch}/{self.config['epochs']}:")
            print(f"Train Loss: {train_loss:.4f}, Train WER: {train_wer:.2%}")
            print(f"Val Loss: {val_loss:.4f}, Val WER: {val_wer:.2%}, Val CER: {val_metrics['cer']:.2%}")
            print(f"LR: {self.optimizer.param_groups[0]['lr']:.6f}")
            
            # Print sample predictions
            print("\nSample predictions:")
            for pred, ref in zip(val_metrics['predictions'][:2], val_metrics['references'][:2]):
                print(f"Reference: {ref}")
                print(f"Predicted: {pred}")
                print()
            
            # Log to wandb
            if self.config.get('use_wandb', False):
                wandb.log({
                    'epoch': epoch,
                    'train_loss': train_loss,
                    'train_wer': train_wer,
                    'val_loss': val_loss,
                    'val_wer': val_wer,
                    'val_cer': val_metrics['cer'],
                    'learning_rate': self.optimizer.param_groups[0]['lr']
                })
            
            # Save best model
            if val_wer < self.best_val_wer:
                self.best_val_wer = val_wer
                self.save_checkpoint(epoch, is_best=True)
            
            # Early stopping
            if self.early_stopping(val_wer):
                print(f"\nEarly stopping triggered at epoch {epoch}")
                break
        
        # Save training curves
        save_training_curves(
            self.train_losses,
            self.val_losses,
            self.train_wers,
            self.val_wers,
            save_path=os.path.join(self.config['checkpoint_dir'], 'training_curves.png'),
            metric_name='WER'
        )
        
        print(f"\nTraining completed. Best validation WER: {self.best_val_wer:.2%}")
    
    def save_checkpoint(self, epoch: int, is_best: bool = False):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_val_wer': self.best_val_wer,
            'tokenizer': self.tokenizer,
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
            print(f"Saved best model with WER: {self.best_val_wer:.2%}")


def main():
    parser = argparse.ArgumentParser(description='ASR Training')
    parser.add_argument('--config', type=str, default='configs/default.json',
                        help='Path to config file')
    parser.add_argument('--dataset', type=str, default='librispeech',
                        help='Dataset name')
    parser.add_argument('--model', type=str, default='conformer',
                        help='Model architecture')
    parser.add_argument('--epochs', type=int, default=50,
                        help='Number of epochs')
    parser.add_argument('--batch-size', type=int, default=16,
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
        'data_path': config.get('data_path', './data'),
        'checkpoint_dir': config.get('checkpoint_dir', './checkpoints'),
        'feature_type': config.get('feature_type', 'melspec'),
        'hidden_dim': config.get('hidden_dim', 256),
        'num_layers': config.get('num_layers', 6),
        'optimizer': config.get('optimizer', 'adamw'),
        'scheduler': config.get('scheduler', 'onecycle'),
        'dropout': config.get('dropout', 0.1),
        'gradient_clip': config.get('gradient_clip', 1.0),
        'use_amp': config.get('use_amp', True),
        'early_stopping_patience': config.get('early_stopping_patience', 10),
        'num_workers': config.get('num_workers', 4),
        'use_wandb': config.get('use_wandb', False)
    })
    
    # Create trainer and train
    trainer = ASRTrainer(config)
    trainer.train()


if __name__ == "__main__":
    main()