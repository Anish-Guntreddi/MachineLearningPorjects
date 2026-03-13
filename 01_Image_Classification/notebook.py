# # Project 01: Image Classification

**Category:** Computer Vision | **Dataset:** CIFAR-10

CIFAR-10/ImageNet classification with CNNs and Vision Transformers

---


import sys
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# Add project directory to path
project_dir = os.path.abspath('01_Image_Classification')
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
    'model_name': 'resnet18',       # resnet18 >> simple_cnn for CIFAR-10
    'epochs': 30,                    # 5 epochs = random chance; 30 gets ~90%+
    'batch_size': 128,               # larger batch = more stable gradients on GPU
    'learning_rate': 0.01,           # higher lr works better with SGD on ResNets
    'optimizer': 'sgd',              # SGD+momentum is the standard for CIFAR
    'weight_decay': 5e-4,
    'scheduler': 'cosine',
    'num_classes': 10,
    'data_dir': './data',
    'num_workers': platform_config['num_workers'],
    'pin_memory': platform_config['pin_memory'],
    'use_amp': platform_config['use_amp'],
    'pretrained': False,             # training from scratch on CIFAR-10
    'checkpoint_dir': './checkpoints',
    'use_wandb': False,
}

print("Training configuration:")
for k, v in config.items():
    print(f"  {k}: {v}")

from data_loader import load_cifar10

# Load dataset
train_loader, val_loader, test_loader = load_cifar10(batch_size=config['batch_size'], num_workers=platform_config['num_workers'], pin_memory=platform_config['pin_memory'])

# Print dataset statistics
print(f"Training batches: {len(train_loader)}")
print(f"Validation batches: {len(val_loader)}")
print(f"Test batches: {len(test_loader)}")


# Visualize sample data
images, targets = next(iter(train_loader))
if isinstance(images, (list, tuple)):
    # Detection format
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    for i, ax in enumerate(axes.flatten()):
        if i < len(images):
            img = images[i]
            if img.shape[0] == 3:
                img = img.permute(1, 2, 0).numpy()
                img = np.clip(img, 0, 1)
            ax.imshow(img)
        ax.axis('off')
    plt.suptitle('Sample Training Images')
    plt.tight_layout()
    plt.show()
else:
    # Classification format
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    for i, ax in enumerate(axes.flatten()):
        if i < images.shape[0]:
            img = images[i].permute(1, 2, 0).numpy()
            img = np.clip(img, 0, 1)
            ax.imshow(img)
            if isinstance(targets, torch.Tensor):
                ax.set_title(f'Label: {targets[i].item()}')
        ax.axis('off')
    plt.suptitle('Sample Training Images')
    plt.tight_layout()
    plt.show()


from models import get_model

# Create model
model = get_model(config['model_name'], num_classes=config['num_classes'], pretrained=config['pretrained'])
model = model.to(device)

# Model summary
total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
param_size_mb = sum(p.nelement() * p.element_size() for p in model.parameters()) / 1024 / 1024

print(f"Model: {type(model).__name__}")
print(f"Total parameters: {total_params:,}")
print(f"Trainable parameters: {trainable_params:,}")
print(f"Model size: {param_size_mb:.2f} MB")

from train import Trainer

# Initialize trainer
trainer = Trainer(config)

# Run training epoch-by-epoch to capture history for plotting
history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}

print("Starting training...")
for epoch in range(config['epochs']):
    train_metrics = trainer.train_epoch(epoch + 1)
    val_metrics = trainer.validate()

    history['train_loss'].append(train_metrics['loss'])
    history['val_loss'].append(val_metrics['loss'])
    history['train_acc'].append(train_metrics['acc1'])
    history['val_acc'].append(val_metrics['acc1'])

    # Update best acc and save checkpoint
    from utils import save_checkpoint
    is_best = val_metrics['acc1'] > trainer.best_acc
    if is_best:
        trainer.best_acc = val_metrics['acc1']
    save_checkpoint({
        'epoch': epoch + 1,
        'model_state_dict': trainer.model.state_dict(),
        'optimizer_state_dict': trainer.optimizer.state_dict(),
        'scheduler_state_dict': trainer.scheduler.state_dict() if trainer.scheduler else None,
        'best_acc': trainer.best_acc,
        'config': config,
    }, is_best, checkpoint_dir=config['checkpoint_dir'])

    if trainer.scheduler:
        from torch.optim.lr_scheduler import OneCycleLR
        if not isinstance(trainer.scheduler, OneCycleLR):
            trainer.scheduler.step()

    print(f"Epoch {epoch+1}/{config['epochs']} | "
          f"Train Loss: {train_metrics['loss']:.4f} Acc: {train_metrics['acc1']:.2f}% | "
          f"Val Loss: {val_metrics['loss']:.4f} Acc: {val_metrics['acc1']:.2f}%")

trainer.history = history
print(f"\nTraining complete! Best val acc: {trainer.best_acc:.2f}%")

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


from tqdm import tqdm
from sklearn.metrics import accuracy_score, classification_report

model.eval()
all_preds = []
all_labels = []

with torch.no_grad():
    for batch in tqdm(test_loader, desc="Evaluating"):
        if isinstance(batch, (list, tuple)) and len(batch) == 2:
            images, labels = batch
            if isinstance(images, list):
                images = [img.to(device) for img in images]
                outputs = model(images)
            else:
                images = images.to(device)
                outputs = model(images)
                _, preds = torch.max(outputs, 1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.numpy())

if all_preds:
    acc = accuracy_score(all_labels, all_preds) * 100
    print(f"\nTest Accuracy: {acc:.2f}%")
    print("\nClassification Report:")
    print(classification_report(all_labels, all_preds))

# Visualize predictions
model.eval()

# Get a batch of test data
test_batch = next(iter(test_loader))
if isinstance(test_batch, (list, tuple)):
    images, targets = test_batch
else:
    images, targets = test_batch, None

if isinstance(images, list):
    sample_images = images[:4]
else:
    sample_images = images[:8].to(device)

with torch.no_grad():
    if isinstance(sample_images, list):
        predictions = model([img.to(device) for img in sample_images])
    else:
        predictions = model(sample_images)

print("Sample predictions generated successfully.")


# ### Domain-Specific: Vision Analysis

# Visualize model predictions with confidence
model.eval()

test_batch = next(iter(test_loader))
if isinstance(test_batch, (list, tuple)):
    images, targets = test_batch
else:
    images, targets = test_batch, None

if isinstance(images, list):
    sample = images[:4]
    with torch.no_grad():
        preds = model([img.to(device) for img in sample])

    fig, axes = plt.subplots(1, len(sample), figsize=(4 * len(sample), 4))
    if len(sample) == 1:
        axes = [axes]
    for i, (img, pred) in enumerate(zip(sample, preds)):
        axes[i].imshow(img.permute(1, 2, 0).numpy().clip(0, 1))
        if 'scores' in pred:
            n_det = (pred['scores'] > 0.5).sum().item()
            axes[i].set_title(f'{n_det} detections')
        axes[i].axis('off')
    plt.tight_layout()
    plt.show()
else:
    sample = images[:8].to(device)
    with torch.no_grad():
        outputs = model(sample)
        if isinstance(outputs, torch.Tensor):
            probs = torch.softmax(outputs, dim=1)
            confs, preds = probs.max(dim=1)

    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    for i, ax in enumerate(axes.flatten()):
        if i < len(sample):
            img = sample[i].cpu().permute(1, 2, 0).numpy().clip(0, 1)
            ax.imshow(img)
            if isinstance(outputs, torch.Tensor):
                ax.set_title(f'Pred: {preds[i].item()} ({confs[i].item():.2f})')
        ax.axis('off')
    plt.tight_layout()
    plt.show()

print("Vision analysis complete.")


import yaml
from datetime import datetime

# Model card
model_card = {
    'project_id': '01',
    'title': 'Image Classification',
    'short_description': 'CIFAR-10/ImageNet classification with CNNs and Vision Transformers',
    'category': 'Computer Vision',
    'input_type': 'image',
    'output_type': 'class_label',
    'default_model': 'simple_cnn',
    'models_available': ['simple_cnn', 'resnet18', 'resnet50', 'efficientnet_b0', 'vit_tiny_patch16_224'],
    'dataset': {
        'name': 'CIFAR-10',
        'num_classes': 10,
        'train_size': 45000,
        'test_size': 10000,
    },
    'tags': ['cnn', 'cifar-10', 'image-classification'],
    'demo_type': 'interactive',
}

# Results
results = {
    'project_id': '01',
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
os.makedirs('01_Image_Classification', exist_ok=True)

with open(os.path.join('01_Image_Classification', 'model_card.yaml'), 'w') as f:
    yaml.dump(model_card, f, default_flow_style=False, sort_keys=False)

with open(os.path.join('01_Image_Classification', 'results.yaml'), 'w') as f:
    yaml.dump(results, f, default_flow_style=False, sort_keys=False)

print("Exported model_card.yaml and results.yaml")


# ## Summary

### Image Classification

**Category:** Computer Vision

#### Key Findings
- Model trained successfully with the configured hyperparameters
- Results exported to `model_card.yaml` and `results.yaml`

#### Next Steps
- Experiment with different model architectures
- Tune hyperparameters for better performance
- Run full training with more epochs and larger dataset subsets
- See the project README for detailed training configurations
