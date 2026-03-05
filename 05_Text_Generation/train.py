"""
Training script for text generation
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm
import argparse
import os
import json
from datetime import datetime
from typing import Dict

from models import get_model
from data_loader import create_data_loaders
from utils import (
    AverageMeter,
    EarlyStopping,
    save_checkpoint,
    load_checkpoint,
    calculate_perplexity,
    plot_training_curves
)


class TextGenerationTrainer:
    """Text generation trainer"""

    def __init__(self, config: Dict):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")

        # Load data
        self.train_loader, self.val_loader, self.test_loader = create_data_loaders(
            dataset_name=config.get('dataset', 'wikitext'),
            tokenizer_name=config.get('tokenizer_name', 'gpt2'),
            max_length=config.get('max_length', 512),
            stride=config.get('stride', 256),
            batch_size=config['batch_size'],
            num_workers=config.get('num_workers', 4)
        )

        # Initialize model
        self.model = get_model(
            config['model_name'],
            vocab_size=config.get('vocab_size', 50257)
        ).to(self.device)

        # Loss function
        self.criterion = nn.CrossEntropyLoss(
            ignore_index=-100,
            label_smoothing=config.get('label_smoothing', 0.0)
        )

        # Optimizer
        self.optimizer = self._get_optimizer()

        # Scheduler
        self.scheduler = self._get_scheduler()

        # Mixed precision
        self.use_amp = config.get('use_amp', True) and torch.cuda.is_available()
        self.scaler = GradScaler() if self.use_amp else None

        # Training state
        self.start_epoch = 0
        self.best_val_loss = float('inf')
        self.history = {
            'train_loss': [], 'val_loss': [],
            'train_ppl': [], 'val_ppl': [],
            'learning_rate': []
        }

        # Early stopping
        self.early_stopping = EarlyStopping(
            patience=config.get('early_stopping_patience', 10),
            min_delta=config.get('early_stopping_min_delta', 0.001)
        )

        # Resume from checkpoint
        if config.get('resume'):
            self._load_checkpoint(config['resume'])

        # WandB
        self.use_wandb = config.get('use_wandb', False)
        if self.use_wandb:
            import wandb
            wandb.init(project='text-generation', config=config)

    def _get_optimizer(self):
        """Get optimizer"""
        opt_name = self.config.get('optimizer', 'adamw').lower()
        lr = self.config['learning_rate']
        weight_decay = self.config.get('weight_decay', 0.01)

        if opt_name == 'adamw':
            return optim.AdamW(
                self.model.parameters(),
                lr=lr,
                weight_decay=weight_decay
            )
        elif opt_name == 'adam':
            return optim.Adam(
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
            return optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=self.config['epochs'],
                eta_min=1e-6
            )
        elif scheduler_name == 'linear':
            total_steps = len(self.train_loader) * self.config['epochs']
            return optim.lr_scheduler.LinearLR(
                self.optimizer,
                start_factor=1.0,
                end_factor=0.1,
                total_iters=total_steps
            )
        else:
            return None

    def _load_checkpoint(self, path: str):
        """Load checkpoint"""
        checkpoint = load_checkpoint(path, self.model, self.optimizer)
        self.start_epoch = checkpoint.get('epoch', 0)
        self.best_val_loss = checkpoint.get('best_val_loss', float('inf'))
        if 'history' in checkpoint:
            self.history = checkpoint['history']
        print(f"Resumed from epoch {self.start_epoch}")

    def train_epoch(self, epoch: int) -> Dict:
        """Train for one epoch"""
        self.model.train()
        losses = AverageMeter()

        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch}/{self.config['epochs']}")

        for batch in pbar:
            input_ids = batch['input_ids'].to(self.device)
            labels = batch['labels'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)

            if self.use_amp:
                with autocast():
                    outputs = self._forward(input_ids, attention_mask, labels)
                    loss = outputs if isinstance(outputs, torch.Tensor) else outputs.loss

                self.optimizer.zero_grad()
                self.scaler.scale(loss).backward()

                if self.config.get('gradient_clip', 0) > 0:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.config['gradient_clip']
                    )

                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                outputs = self._forward(input_ids, attention_mask, labels)
                loss = outputs if isinstance(outputs, torch.Tensor) else outputs.loss

                self.optimizer.zero_grad()
                loss.backward()

                if self.config.get('gradient_clip', 0) > 0:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.config['gradient_clip']
                    )

                self.optimizer.step()

            losses.update(loss.item(), input_ids.size(0))
            pbar.set_postfix({
                'Loss': f'{losses.avg:.4f}',
                'PPL': f'{calculate_perplexity(losses.avg):.2f}'
            })

        return {'loss': losses.avg, 'perplexity': calculate_perplexity(losses.avg)}

    def _forward(self, input_ids, attention_mask, labels):
        """Forward pass handling different model types"""
        # HuggingFace models (GPT2LMHeadModel, etc.)
        if hasattr(self.model, 'forward') and 'labels' in self.model.forward.__code__.co_varnames:
            return self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )

        # GPT2FineTuner wrapper
        if hasattr(self.model, 'gpt2'):
            return self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )

        # Custom models (TransformerLM, LSTMGenerator)
        if hasattr(self.model, 'init_hidden'):
            # LSTM model
            logits, _ = self.model(input_ids)
        else:
            logits = self.model(input_ids, attention_mask=attention_mask)

        # Shift logits and labels for next-token prediction
        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = labels[:, 1:].contiguous()

        loss = self.criterion(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1)
        )
        return loss

    @torch.no_grad()
    def validate(self) -> Dict:
        """Validate the model"""
        self.model.eval()
        losses = AverageMeter()

        for batch in tqdm(self.val_loader, desc="Validating"):
            input_ids = batch['input_ids'].to(self.device)
            labels = batch['labels'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)

            outputs = self._forward(input_ids, attention_mask, labels)
            loss = outputs if isinstance(outputs, torch.Tensor) else outputs.loss

            losses.update(loss.item(), input_ids.size(0))

        return {'loss': losses.avg, 'perplexity': calculate_perplexity(losses.avg)}

    def train(self):
        """Main training loop"""
        for epoch in range(self.start_epoch, self.config['epochs']):
            # Training
            train_metrics = self.train_epoch(epoch)

            # Validation
            val_metrics = self.validate()

            # Update scheduler
            if self.scheduler:
                self.scheduler.step()

            # Record history
            current_lr = self.optimizer.param_groups[0]['lr']
            self.history['train_loss'].append(train_metrics['loss'])
            self.history['val_loss'].append(val_metrics['loss'])
            self.history['train_ppl'].append(train_metrics['perplexity'])
            self.history['val_ppl'].append(val_metrics['perplexity'])
            self.history['learning_rate'].append(current_lr)

            # Log metrics
            print(f"\nEpoch {epoch}/{self.config['epochs']}:")
            print(f"  Train Loss: {train_metrics['loss']:.4f} | PPL: {train_metrics['perplexity']:.2f}")
            print(f"  Val Loss:   {val_metrics['loss']:.4f} | PPL: {val_metrics['perplexity']:.2f}")
            print(f"  LR: {current_lr:.2e}")

            # WandB logging
            if self.use_wandb:
                import wandb
                wandb.log({
                    'epoch': epoch,
                    'train_loss': train_metrics['loss'],
                    'val_loss': val_metrics['loss'],
                    'train_perplexity': train_metrics['perplexity'],
                    'val_perplexity': val_metrics['perplexity'],
                    'learning_rate': current_lr
                })

            # Save checkpoint
            is_best = val_metrics['loss'] < self.best_val_loss
            if is_best:
                self.best_val_loss = val_metrics['loss']

            save_checkpoint({
                'epoch': epoch + 1,
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
                'best_val_loss': self.best_val_loss,
                'config': self.config,
                'history': self.history
            }, is_best, checkpoint_dir=self.config['checkpoint_dir'])

            # Early stopping
            if self.early_stopping(val_metrics['loss']):
                print(f"Early stopping at epoch {epoch}")
                break

        # Plot training curves
        plot_training_curves(self.history)

        return self.history


def main():
    parser = argparse.ArgumentParser(description='Text Generation Training')
    parser.add_argument('--config', type=str, default='configs/default.json',
                        help='Path to config file')
    parser.add_argument('--model', type=str, default='gpt2',
                        help='Model architecture')
    parser.add_argument('--dataset', type=str, default='wikitext',
                        help='Dataset name')
    parser.add_argument('--epochs', type=int, default=10,
                        help='Number of epochs')
    parser.add_argument('--batch-size', type=int, default=8,
                        help='Batch size')
    parser.add_argument('--lr', type=float, default=5e-5,
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

    # Override config with CLI args
    config['model_name'] = args.model
    config['dataset'] = args.dataset
    config['epochs'] = args.epochs
    config['batch_size'] = args.batch_size
    config['learning_rate'] = args.lr
    config['resume'] = args.resume

    # Set defaults
    config.setdefault('checkpoint_dir', './checkpoints')
    config.setdefault('num_workers', 4)
    config.setdefault('use_amp', True)
    config.setdefault('gradient_clip', 1.0)
    config.setdefault('weight_decay', 0.01)
    config.setdefault('optimizer', 'adamw')
    config.setdefault('scheduler', 'cosine')
    config.setdefault('max_length', 512)
    config.setdefault('stride', 256)
    config.setdefault('tokenizer_name', 'gpt2')
    config.setdefault('vocab_size', 50257)

    # Create directories
    os.makedirs(config['checkpoint_dir'], exist_ok=True)

    # Train
    trainer = TextGenerationTrainer(config)
    trainer.train()


if __name__ == "__main__":
    main()
