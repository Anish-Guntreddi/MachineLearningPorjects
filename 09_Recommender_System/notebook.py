# # Project 09: Recommender System

**Category:** Recommender | **Dataset:** MovieLens 100K

Collaborative filtering and neural recommenders

---


import sys
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# Add project directory to path
project_dir = os.path.abspath('09_Recommender_System')
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
    'model_name': 'ncf',
    'epochs': 10,
    'batch_size': 256,
    'learning_rate': 0.001,
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


from data_loader import RecommenderDataModule

# Load dataset
data_module = RecommenderDataModule(dataset_name='movielens', data_path='./data', batch_size=config['batch_size'], num_workers=platform_config['num_workers'])
data_module.setup()
train_loader = data_module.train_dataloader()
val_loader = data_module.val_dataloader()
test_loader = data_module.test_dataloader()

# Print dataset statistics
print(f"Training batches: {len(train_loader)}")
print(f"Validation batches: {len(val_loader)}")
print(f"Test batches: {len(test_loader)}")


# Examine sample data
batch = next(iter(train_loader))
if isinstance(batch, dict):
    print("Batch keys:", list(batch.keys()))
    for key, val in batch.items():
        if isinstance(val, torch.Tensor):
            print(f"  {key}: shape={val.shape}, dtype={val.dtype}")
elif isinstance(batch, (list, tuple)):
    for i, item in enumerate(batch):
        if isinstance(item, torch.Tensor):
            print(f"  Item {i}: shape={item.shape}, dtype={item.dtype}")


from models import get_model

# Create model
model = get_model(config['model_name'], num_users=data_module.num_users, num_items=data_module.num_items, embedding_dim=config.get('embedding_dim', 64))
model = model.to(device)

# Model summary
total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
param_size_mb = sum(p.nelement() * p.element_size() for p in model.parameters()) / 1024 / 1024

print(f"Model: {type(model).__name__}")
print(f"Total parameters: {total_params:,}")
print(f"Trainable parameters: {trainable_params:,}")
print(f"Model size: {param_size_mb:.2f} MB")


from train import RecommenderTrainer

# Initialize trainer
trainer = RecommenderTrainer(config)

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
print("Running evaluation on test set...")

if hasattr(trainer, 'test'):
    test_metrics = trainer.test()
    print("\nTest Results:")
    if isinstance(test_metrics, dict):
        for k, v in test_metrics.items():
            if isinstance(v, float):
                print(f"  {k}: {v:.4f}")
            else:
                print(f"  {k}: {v}")
elif hasattr(trainer, 'validate'):
    test_metrics = trainer.validate()
    print("\nValidation Results:")
    if isinstance(test_metrics, dict):
        for k, v in test_metrics.items():
            if isinstance(v, float):
                print(f"  {k}: {v:.4f}")
            else:
                print(f"  {k}: {v}")


# Sample predictions
print("Generating sample predictions...")
model.eval()

test_batch = next(iter(test_loader))
if isinstance(test_batch, dict):
    sample = {k: v[:4].to(device) if isinstance(v, torch.Tensor) else v for k, v in test_batch.items()}
elif isinstance(test_batch, (list, tuple)):
    sample = [t[:4].to(device) if isinstance(t, torch.Tensor) else t for t in test_batch]

with torch.no_grad():
    if isinstance(sample, dict):
        outputs = model(**{k: v for k, v in sample.items() if k != 'labels' and k != 'label'})
    else:
        outputs = model(sample[0])

print("Sample predictions generated successfully.")


# ### Domain-Specific: Tabular Data Analysis

# Tabular data analysis
print("Data statistics:")

test_batch = next(iter(test_loader))
if isinstance(test_batch, dict):
    for key, val in test_batch.items():
        if isinstance(val, torch.Tensor):
            print(f"  {key}: shape={val.shape}, mean={val.float().mean():.4f}, std={val.float().std():.4f}")
elif isinstance(test_batch, (list, tuple)):
    for i, item in enumerate(test_batch):
        if isinstance(item, torch.Tensor):
            print(f"  Tensor {i}: shape={item.shape}, mean={item.float().mean():.4f}, std={item.float().std():.4f}")

# Visualize predictions vs actual if applicable
try:
    model.eval()
    with torch.no_grad():
        if isinstance(test_batch, dict):
            inputs = test_batch.get('input', test_batch.get('user_ids'))
            if inputs is not None:
                sample_input = {k: v[:8].to(device) for k, v in test_batch.items()
                               if isinstance(v, torch.Tensor) and k not in ('target', 'label', 'labels', 'rating')}
                outputs = model(**sample_input)
                print(f"\nSample output shape: {outputs.shape if isinstance(outputs, torch.Tensor) else type(outputs)}")
except Exception as e:
    print(f"Could not generate predictions: {e}")

print("\nTabular analysis complete.")


import yaml
from datetime import datetime

# Model card
model_card = {
    'project_id': '09',
    'title': 'Recommender System',
    'short_description': 'Collaborative filtering and neural recommenders',
    'category': 'Recommender',
    'input_type': 'tabular',
    'output_type': 'recommendations',
    'default_model': 'ncf',
    'models_available': ['matrix_factorization', 'ncf', 'deepfm'],
    'dataset': {
        'name': 'MovieLens 100K',
        'num_classes': 0,
        'train_size': 80000,
        'test_size': 20000,
    },
    'tags': ['recommender', 'collaborative-filtering', 'movielens'],
    'demo_type': 'interactive',
}

# Results
results = {
    'project_id': '09',
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
os.makedirs('09_Recommender_System', exist_ok=True)

with open(os.path.join('09_Recommender_System', 'model_card.yaml'), 'w') as f:
    yaml.dump(model_card, f, default_flow_style=False, sort_keys=False)

with open(os.path.join('09_Recommender_System', 'results.yaml'), 'w') as f:
    yaml.dump(results, f, default_flow_style=False, sort_keys=False)

print("Exported model_card.yaml and results.yaml")


# ## Summary

### Recommender System

**Category:** Recommender

#### Key Findings
- Model trained successfully with the configured hyperparameters
- Results exported to `model_card.yaml` and `results.yaml`

#### Next Steps
- Experiment with different model architectures
- Tune hyperparameters for better performance
- Run full training with more epochs and larger dataset subsets
- See the project README for detailed training configurations
