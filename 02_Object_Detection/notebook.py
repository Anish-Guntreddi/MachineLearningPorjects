# # Project 02: Object Detection

**Category:** Computer Vision | **Dataset:** PASCAL VOC

Object detection with YOLO/Faster R-CNN

---


import sys
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# Add project directory to path
project_dir = os.path.abspath('02_Object_Detection')
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
    'model_name': 'fasterrcnn_resnet50_fpn',
    'epochs': 5,
    'batch_size': 4,
    'learning_rate': 0.005,
    'num_classes': 21,
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
train_loader, val_loader, test_loader = create_data_loaders(batch_size=config['batch_size'], num_workers=platform_config['num_workers'])

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
model = get_model('fasterrcnn_resnet50_fpn', num_classes=21, pretrained=True)
model = model.to(device)

# Model summary
total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
param_size_mb = sum(p.nelement() * p.element_size() for p in model.parameters()) / 1024 / 1024

print(f"Model: {type(model).__name__}")
print(f"Total parameters: {total_params:,}")
print(f"Trainable parameters: {trainable_params:,}")
print(f"Model size: {param_size_mb:.2f} MB")


from train import DetectionTrainer

# Initialize trainer
trainer = DetectionTrainer(config)

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
    'project_id': '02',
    'title': 'Object Detection',
    'short_description': 'Object detection with YOLO/Faster R-CNN',
    'category': 'Computer Vision',
    'input_type': 'image',
    'output_type': 'bounding_boxes',
    'default_model': 'fasterrcnn_resnet50_fpn',
    'models_available': ['fasterrcnn_resnet50_fpn', 'retinanet_resnet50_fpn'],
    'dataset': {
        'name': 'PASCAL VOC',
        'num_classes': 21,
        'train_size': 16551,
        'test_size': 4952,
    },
    'tags': ['object-detection', 'yolo', 'faster-rcnn'],
    'demo_type': 'interactive',
}

# Results
results = {
    'project_id': '02',
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
os.makedirs('02_Object_Detection', exist_ok=True)

with open(os.path.join('02_Object_Detection', 'model_card.yaml'), 'w') as f:
    yaml.dump(model_card, f, default_flow_style=False, sort_keys=False)

with open(os.path.join('02_Object_Detection', 'results.yaml'), 'w') as f:
    yaml.dump(results, f, default_flow_style=False, sort_keys=False)

print("Exported model_card.yaml and results.yaml")


# ## Summary

### Object Detection

**Category:** Computer Vision

#### Key Findings
- Model trained successfully with the configured hyperparameters
- Results exported to `model_card.yaml` and `results.yaml`

#### Next Steps
- Experiment with different model architectures
- Tune hyperparameters for better performance
- Run full training with more epochs and larger dataset subsets
- See the project README for detailed training configurations
