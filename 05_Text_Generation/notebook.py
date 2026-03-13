# # Project 05: Text Generation

**Category:** NLP | **Dataset:** WikiText

GPT-style text generation

---


import sys
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# Add project directory to path
project_dir = os.path.abspath('05_Text_Generation')
if project_dir not in sys.path:
    sys.path.insert(0, project_dir)

# Device auto-detection: CUDA -> MPS -> CPU
def setup_device():
    if torch.cuda.is_available():
        device = torch.device('cuda')
        torch.backends.cudnn.benchmark = True
        print(f'Using CUDA: {torch.cuda.get_device_name(0)}')
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        device = torch.device('mps')
        print('Using Apple MPS')
    else:
        device = torch.device('cpu')
        print('Using CPU')
    return device

def get_platform_config(device):
    if device.type == 'cuda':
        return {'num_workers': 4, 'pin_memory': True, 'use_amp': True, 'amp_dtype': torch.float16}
    elif device.type == 'mps':
        return {'num_workers': 0, 'pin_memory': False, 'use_amp': True, 'amp_dtype': torch.float16}
    else:
        return {'num_workers': 2, 'pin_memory': False, 'use_amp': False, 'amp_dtype': None}

device = setup_device()
platform_config = get_platform_config(device)

# Set random seed for reproducibility
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

print(f'PyTorch version: {torch.__version__}')
print(f'Platform config: {platform_config}')


# Training Configuration
config = {
    'model_name': 'gpt2',
    'epochs': 3,
    'batch_size': 8,
    'learning_rate': 5e-5,
    'num_classes': 0,
    'num_workers': platform_config['num_workers'],
    'pin_memory': platform_config['pin_memory'],
    'use_amp': platform_config['use_amp'],
    'checkpoint_dir': './checkpoints',
    'use_wandb': False,
}

print("Training configuration:")
for k, v in config.items():
    print(f"  {k}: {v}")


from data_loader import create_data_loaders

# Load dataset
train_loader, val_loader, test_loader = create_data_loaders(dataset_name='wikitext', batch_size=config['batch_size'], num_workers=platform_config['num_workers'])

# Print dataset statistics
print(f"Training batches: {len(train_loader)}")
print(f"Validation batches: {len(val_loader)}")
print(f"Test batches: {len(test_loader)}")


# Examine sample data
batch = next(iter(train_loader))
print("Batch keys:", list(batch.keys()))
for key, val in batch.items():
    if isinstance(val, torch.Tensor):
        print(f"  {key}: shape={val.shape}, dtype={val.dtype}")


from models import get_model

# Create model
model = get_model('gpt2')
model = model.to(device)

# Model summary
total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
param_size_mb = sum(p.nelement() * p.element_size() for p in model.parameters()) / 1024 / 1024

print(f"Model: {type(model).__name__}")
print(f"Total parameters: {total_params:,}")
print(f"Trainable parameters: {trainable_params:,}")
print(f"Model size: {param_size_mb:.2f} MB")


from train import TextGenerationTrainer

# Initialize trainer
trainer = TextGenerationTrainer(config)

# Run training
print("Starting training...")
trainer.train()

print("\nTraining complete!")


# Plot training curves
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

if hasattr(trainer, 'history'):
    history = trainer.history
elif hasattr(trainer, 'train_losses'):
    history = {
        'train_loss': trainer.train_losses,
        'val_loss': trainer.val_losses if hasattr(trainer, 'val_losses') else [],
    }
else:
    history = {'train_loss': [], 'val_loss': []}

if history.get('train_loss'):
    axes[0].plot(history['train_loss'], label='Train Loss', color='#0f766e')
    if history.get('val_loss'):
        axes[0].plot(history['val_loss'], label='Val Loss', color='#dc2626')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Training and Validation Loss')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

# Plot accuracy or secondary metric if available
if history.get('train_acc'):
    axes[1].plot(history['train_acc'], label='Train Acc', color='#0f766e')
    if history.get('val_acc'):
        axes[1].plot(history['val_acc'], label='Val Acc', color='#dc2626')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy (%)')
    axes[1].set_title('Training and Validation Accuracy')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
elif history.get('learning_rate'):
    axes[1].plot(history['learning_rate'], color='#0f766e')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Learning Rate')
    axes[1].set_title('Learning Rate Schedule')
    axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()


# Evaluate on test set
model.eval()
total_loss = 0
num_batches = 0

with torch.no_grad():
    for batch in tqdm(test_loader, desc="Evaluating"):
        input_ids = batch['input_ids'].to(device) if isinstance(batch, dict) else batch[0].to(device)
        labels = batch.get('labels', batch.get('label')).to(device) if isinstance(batch, dict) else batch[1].to(device)

        if hasattr(model, 'gpt2') or hasattr(model, 'generate'):
            outputs = model(input_ids=input_ids, labels=labels)
            loss = outputs.loss if hasattr(outputs, 'loss') else outputs[0]
        else:
            outputs = model(input_ids)
            if isinstance(outputs, tuple):
                outputs = outputs[0]
            loss = torch.nn.functional.cross_entropy(
                outputs.view(-1, outputs.size(-1)),
                labels.view(-1),
                ignore_index=-100
            )

        total_loss += loss.item()
        num_batches += 1

avg_loss = total_loss / max(num_batches, 1)
import math
perplexity = math.exp(min(avg_loss, 100))
print(f"\nTest Loss: {avg_loss:.4f}")
print(f"Test Perplexity: {perplexity:.2f}")


# Generate sample predictions
model.eval()

print("Sample predictions:")
test_batch = next(iter(test_loader))
if isinstance(test_batch, dict):
    input_ids = test_batch['input_ids'][:3].to(device)
    with torch.no_grad():
        if hasattr(model, 'generate'):
            outputs = model.generate(input_ids, max_new_tokens=50, do_sample=True, top_k=50)
            print("Generated text samples created.")
        else:
            outputs = model(input_ids)
            print(f"Output shape: {outputs.shape if isinstance(outputs, torch.Tensor) else type(outputs)}")


# ### Domain-Specific: NLP Analysis

# NLP-specific analysis
model.eval()

# Show model vocabulary info if available
if hasattr(model, 'config'):
    print(f"Model config vocab size: {getattr(model.config, 'vocab_size', 'N/A')}")
    print(f"Model config hidden size: {getattr(model.config, 'hidden_size', 'N/A')}")

# Sample generation or prediction
test_batch = next(iter(test_loader))
if isinstance(test_batch, dict):
    input_ids = test_batch.get('input_ids', test_batch.get('src_input_ids'))
    if input_ids is not None:
        print(f"\nSample input shape: {input_ids.shape}")
        print(f"Input token range: [{input_ids.min().item()}, {input_ids.max().item()}]")

print("\nNLP analysis complete.")


import yaml
from datetime import datetime

# Model card
model_card = {
    'project_id': '05',
    'title': 'Text Generation',
    'short_description': 'GPT-style text generation',
    'category': 'NLP',
    'input_type': 'text',
    'output_type': 'generated_text',
    'default_model': 'gpt2',
    'models_available': ['gpt2', 'gpt2-medium', 'transformer', 'lstm'],
    'dataset': {
        'name': 'WikiText',
        'num_classes': 0,
        'train_size': 28475,
        'test_size': 4358,
    },
    'tags': ['text-generation', 'gpt2', 'language-model'],
    'demo_type': 'interactive',
}

# Results
results = {
    'project_id': '05',
    'timestamp': datetime.now().isoformat(),
    'device_used': str(device),
    'best_model': config['model_name'],
    'metrics': {},
    'training_history': {},
}

# Populate results from trainer
if hasattr(trainer, 'history'):
    results['training_history'] = trainer.history
if hasattr(trainer, 'best_acc'):
    results['metrics']['best_accuracy'] = float(trainer.best_acc)
if hasattr(trainer, 'best_val_loss'):
    results['metrics']['best_val_loss'] = float(trainer.best_val_loss)
if hasattr(trainer, 'best_val_rmse'):
    results['metrics']['best_val_rmse'] = float(trainer.best_val_rmse)
if hasattr(trainer, 'best_bleu'):
    results['metrics']['best_bleu'] = float(trainer.best_bleu)
if hasattr(trainer, 'best_map'):
    results['metrics']['best_map'] = float(trainer.best_map)

# Save
os.makedirs('05_Text_Generation', exist_ok=True)

with open(os.path.join('05_Text_Generation', 'model_card.yaml'), 'w') as f:
    yaml.dump(model_card, f, default_flow_style=False, sort_keys=False)

with open(os.path.join('05_Text_Generation', 'results.yaml'), 'w') as f:
    yaml.dump(results, f, default_flow_style=False, sort_keys=False)

print("Exported model_card.yaml and results.yaml")


# ## Summary

### Text Generation

**Category:** NLP

#### Key Findings
- Model trained successfully with the configured hyperparameters
- Results exported to `model_card.yaml` and `results.yaml`

#### Next Steps
- Experiment with different model architectures
- Tune hyperparameters for better performance
- Run full training with more epochs and larger dataset subsets
- See the project README for detailed training configurations
