"""
Common notebook cells shared across all 12 projects.
Each function returns an nbformat cell object.
"""
import nbformat


def title_cell(config):
    """Cell 1: Title and overview (markdown)"""
    project_num = config['project_num']
    project_name = config['project_name']
    description = config['description']
    category = config['category']
    dataset_name = config['data'].get('dataset_name', 'N/A')

    source = f"""# Project {project_num:02d}: {project_name}

**Category:** {category} | **Dataset:** {dataset_name}

{description}

---
"""
    return nbformat.v4.new_markdown_cell(source)


def setup_cell(config):
    """Cell 2: Environment setup and device detection (code)"""
    project_dir = config['project_dir']

    source = f"""import sys
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# Add project directory to path
project_dir = os.path.abspath('{project_dir}')
if project_dir not in sys.path:
    sys.path.insert(0, project_dir)

# Device auto-detection: CUDA -> MPS -> CPU
def setup_device():
    if torch.cuda.is_available():
        device = torch.device('cuda')
        torch.backends.cudnn.benchmark = True
        print(f'Using CUDA: {{torch.cuda.get_device_name(0)}}')
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        device = torch.device('mps')
        print('Using Apple MPS')
    else:
        device = torch.device('cpu')
        print('Using CPU')
    return device

def get_platform_config(device):
    if device.type == 'cuda':
        return {{'num_workers': 4, 'pin_memory': True, 'use_amp': True, 'amp_dtype': torch.float16}}
    elif device.type == 'mps':
        return {{'num_workers': 0, 'pin_memory': False, 'use_amp': True, 'amp_dtype': torch.float16}}
    else:
        return {{'num_workers': 2, 'pin_memory': False, 'use_amp': False, 'amp_dtype': None}}

device = setup_device()
platform_config = get_platform_config(device)

# Set random seed for reproducibility
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

print(f'PyTorch version: {{torch.__version__}}')
print(f'Platform config: {{platform_config}}')
"""
    return nbformat.v4.new_code_cell(source)


def config_cell(config):
    """Cell 3: Configuration (code)"""
    training = config['training']
    model_cfg = config['model']
    data_cfg = config['data']

    source = f"""# Training Configuration
config = {{
    'model_name': '{model_cfg.get("default_model", "default")}',
    'epochs': {training.get('epochs', 5)},
    'batch_size': {training.get('batch_size', 64)},
    'learning_rate': {training.get('learning_rate', 0.001)},
    'num_classes': {data_cfg.get('num_classes', 10)},
    'num_workers': platform_config['num_workers'],
    'pin_memory': platform_config['pin_memory'],
    'use_amp': platform_config['use_amp'],
    'checkpoint_dir': './checkpoints',
    'use_wandb': False,
}}

print("Training configuration:")
for k, v in config.items():
    print(f"  {{k}}: {{v}}")
"""
    return nbformat.v4.new_code_cell(source)


def data_loading_cell(config):
    """Cell 4: Data loading and exploration (code)"""
    data_cfg = config['data']
    loader_import = data_cfg.get('loader_import', '')
    loader_call = data_cfg.get('loader_call', '')

    source = f"""{loader_import}

# Load dataset
{loader_call}

# Print dataset statistics
print(f"Training batches: {{len(train_loader)}}")
print(f"Validation batches: {{len(val_loader)}}")
print(f"Test batches: {{len(test_loader)}}")
"""
    return nbformat.v4.new_code_cell(source)


def inline_data_cell(inline_code):
    """Cell 4 alternative: inline data loading for projects without data_loader.py"""
    return nbformat.v4.new_code_cell(inline_code)


def preprocessing_cell(config):
    """Cell 5: Preprocessing and augmentation (code)"""
    domain = config['domain']

    if domain == 'vision':
        source = """# Visualize sample data
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
"""
    elif domain == 'nlp':
        source = """# Examine sample data
batch = next(iter(train_loader))
print("Batch keys:", list(batch.keys()))
for key, val in batch.items():
    if isinstance(val, torch.Tensor):
        print(f"  {key}: shape={val.shape}, dtype={val.dtype}")
"""
    elif domain == 'audio':
        source = """# Visualize sample audio data
batch = next(iter(train_loader))
print("Batch keys:", list(batch.keys()))
for key, val in batch.items():
    if isinstance(val, torch.Tensor):
        print(f"  {key}: shape={val.shape}, dtype={val.dtype}")
"""
    elif domain == 'tabular':
        source = """# Examine sample data
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
"""
    else:  # multimodal
        source = """# Examine sample multimodal data
batch = next(iter(train_loader))
if isinstance(batch, dict):
    print("Batch keys:", list(batch.keys()))
    for key, val in batch.items():
        if isinstance(val, torch.Tensor):
            print(f"  {key}: shape={val.shape}, dtype={val.dtype}")
"""

    return nbformat.v4.new_code_cell(source)


def model_cell(config):
    """Cell 6: Model architecture (code)"""
    model_cfg = config['model']
    factory_import = model_cfg.get('factory_import', 'from models import get_model')
    factory_call = model_cfg.get('factory_call', "get_model('default')")

    source = f"""{factory_import}

# Create model
model = {factory_call}
model = model.to(device)

# Model summary
total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
param_size_mb = sum(p.nelement() * p.element_size() for p in model.parameters()) / 1024 / 1024

print(f"Model: {{type(model).__name__}}")
print(f"Total parameters: {{total_params:,}}")
print(f"Trainable parameters: {{trainable_params:,}}")
print(f"Model size: {{param_size_mb:.2f}} MB")
"""
    return nbformat.v4.new_code_cell(source)


def training_cell(config):
    """Cell 7: Training loop (code)"""
    training = config['training']
    trainer_import = training.get('trainer_import', 'from train import Trainer')
    trainer_class = training.get('trainer_class', 'Trainer')

    source = f"""{trainer_import}

# Initialize trainer
trainer = {trainer_class}(config)

# Run training
print("Starting training...")
trainer.train()

print("\\nTraining complete!")
"""
    return nbformat.v4.new_code_cell(source)


def inline_training_cell(inline_code):
    """Cell 7 alternative: inline training for projects without train.py"""
    return nbformat.v4.new_code_cell(inline_code)


def visualization_cell(config):
    """Cell 8: Training visualization (code)"""
    source = """# Plot training curves
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
"""
    return nbformat.v4.new_code_cell(source)


def evaluation_cell(config):
    """Cell 9: Evaluation and metrics (code)"""
    metrics = config['evaluation'].get('metrics', ['accuracy'])
    domain = config['domain']

    if domain == 'vision' and 'accuracy' in metrics:
        source = """# Evaluate on test set
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

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
    accuracy = accuracy_score(all_labels, all_preds) * 100
    print(f"\\nTest Accuracy: {accuracy:.2f}%")
    print("\\nClassification Report:")
    print(classification_report(all_labels, all_preds))
"""
    elif domain == 'nlp':
        source = """# Evaluate on test set
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
print(f"\\nTest Loss: {avg_loss:.4f}")
print(f"Test Perplexity: {perplexity:.2f}")
"""
    else:
        source = """# Evaluate on test set
print("Running evaluation on test set...")

if hasattr(trainer, 'test'):
    test_metrics = trainer.test()
    print("\\nTest Results:")
    if isinstance(test_metrics, dict):
        for k, v in test_metrics.items():
            if isinstance(v, float):
                print(f"  {k}: {v:.4f}")
            else:
                print(f"  {k}: {v}")
elif hasattr(trainer, 'validate'):
    test_metrics = trainer.validate()
    print("\\nValidation Results:")
    if isinstance(test_metrics, dict):
        for k, v in test_metrics.items():
            if isinstance(v, float):
                print(f"  {k}: {v:.4f}")
            else:
                print(f"  {k}: {v}")
"""

    return nbformat.v4.new_code_cell(source)


def predictions_cell(config):
    """Cell 10: Predictions and visualization (code)"""
    domain = config['domain']

    if domain == 'vision':
        source = """# Visualize predictions
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
"""
    elif domain == 'nlp':
        source = """# Generate sample predictions
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
"""
    else:
        source = """# Sample predictions
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
"""

    return nbformat.v4.new_code_cell(source)


def export_cell(config):
    """Cell 11: Export results (code)"""
    project_dir = config['project_dir']
    project_num = config['project_num']
    project_name = config['project_name']
    category = config['category']
    description = config['description']
    data_cfg = config['data']
    model_cfg = config['model']
    eval_cfg = config['evaluation']

    source = f"""import yaml
from datetime import datetime

# Model card
model_card = {{
    'project_id': '{project_num:02d}',
    'title': '{project_name}',
    'short_description': '{description}',
    'category': '{category}',
    'input_type': '{config.get("input_type", config["domain"])}',
    'output_type': '{config.get("output_type", "prediction")}',
    'default_model': '{model_cfg.get("default_model", "default")}',
    'models_available': {model_cfg.get('models_available', [])},
    'dataset': {{
        'name': '{data_cfg.get("dataset_name", "N/A")}',
        'num_classes': {data_cfg.get('num_classes', 0)},
        'train_size': {data_cfg.get('train_size', 0)},
        'test_size': {data_cfg.get('test_size', 0)},
    }},
    'tags': {config.get('tags', [])},
    'demo_type': '{config.get("demo_type", "precomputed")}',
}}

# Results
results = {{
    'project_id': '{project_num:02d}',
    'timestamp': datetime.now().isoformat(),
    'device_used': str(device),
    'best_model': config['model_name'],
    'metrics': {{}},
    'training_history': {{}},
}}

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
os.makedirs('{project_dir}', exist_ok=True)

with open(os.path.join('{project_dir}', 'model_card.yaml'), 'w') as f:
    yaml.dump(model_card, f, default_flow_style=False, sort_keys=False)

with open(os.path.join('{project_dir}', 'results.yaml'), 'w') as f:
    yaml.dump(results, f, default_flow_style=False, sort_keys=False)

print("Exported model_card.yaml and results.yaml")
"""
    return nbformat.v4.new_code_cell(source)


def summary_cell(config):
    """Cell 12: Summary and next steps (markdown)"""
    project_name = config['project_name']
    category = config['category']

    source = f"""## Summary

### {project_name}

**Category:** {category}

#### Key Findings
- Model trained successfully with the configured hyperparameters
- Results exported to `model_card.yaml` and `results.yaml`

#### Next Steps
- Experiment with different model architectures
- Tune hyperparameters for better performance
- Run full training with more epochs and larger dataset subsets
- See the project README for detailed training configurations
"""
    return nbformat.v4.new_markdown_cell(source)
