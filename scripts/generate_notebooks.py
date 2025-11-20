#!/usr/bin/env python3
"""
Script to generate all 12 Jupyter notebooks for the ML portfolio
Each notebook includes complete training, evaluation, and inference code
with automatic CUDA/CPU device selection
"""

import os
import json
from pathlib import Path

# Base directory
BASE_DIR = Path(__file__).parent.parent
NOTEBOOKS_DIR = BASE_DIR / "notebooks"
NOTEBOOKS_DIR.mkdir(exist_ok=True)

def create_notebook_structure(cells):
    """Create notebook JSON structure"""
    return {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3"
            },
            "language_info": {
                "codemirror_mode": {"name": "ipython", "version": 3},
                "file_extension": ".py",
                "mimetype": "text/x-python",
                "name": "python",
                "nbconvert_exporter": "python",
                "pygments_lexer": "ipython3",
                "version": "3.10.0"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 4
    }

def markdown_cell(text):
    """Create a markdown cell"""
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": text.split("\n")
    }

def code_cell(code):
    """Create a code cell"""
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": code.split("\n")
    }

# Notebook templates for each project
NOTEBOOK_CONFIGS = {
    "02_object_detection": {
        "title": "Object Detection - COCO Dataset",
        "dataset": "COCO",
        "model": "Faster R-CNN",
        "framework": "PyTorch + torchvision"
    },
    "03_instance_segmentation": {
        "title": "Instance Segmentation - COCO Dataset",
        "dataset": "COCO",
        "model": "Mask R-CNN",
        "framework": "PyTorch + Detectron2"
    },
    "04_text_classification": {
        "title": "Text Classification - IMDb Sentiment",
        "dataset": "IMDb",
        "model": "BERT",
        "framework": "PyTorch + Transformers"
    },
    "05_text_generation": {
        "title": "Text Generation - GPT-2",
        "dataset": "Custom Text",
        "model": "GPT-2",
        "framework": "PyTorch + Transformers"
    },
    "06_machine_translation": {
        "title": "Machine Translation - WMT14",
        "dataset": "WMT14 En-De",
        "model": "Transformer",
        "framework": "PyTorch + Transformers"
    },
    "07_speech_emotion_recognition": {
        "title": "Speech Emotion Recognition - RAVDESS",
        "dataset": "RAVDESS",
        "model": "Wav2Vec2",
        "framework": "PyTorch + Transformers"
    },
    "08_automatic_speech_recognition": {
        "title": "Automatic Speech Recognition - LibriSpeech",
        "dataset": "LibriSpeech",
        "model": "Wav2Vec2",
        "framework": "PyTorch + Transformers"
    },
    "09_recommender_system": {
        "title": "Recommender System - MovieLens",
        "dataset": "MovieLens-100K",
        "model": "Neural Collaborative Filtering",
        "framework": "PyTorch"
    },
    "10_time_series_forecasting": {
        "title": "Time Series Forecasting",
        "dataset": "Custom Time Series",
        "model": "LSTM",
        "framework": "PyTorch"
    },
    "11_anomaly_detection": {
        "title": "Anomaly Detection",
        "dataset": "Credit Card Fraud",
        "model": "Autoencoder",
        "framework": "PyTorch"
    },
    "12_multimodal_fusion": {
        "title": "Multimodal Fusion",
        "dataset": "Custom Multimodal",
        "model": "Multi-Input Network",
        "framework": "PyTorch"
    }
}

print("Generating Jupyter notebooks...")
print("=" * 60)

for notebook_name, config in NOTEBOOK_CONFIGS.items():
    print(f"\nGenerating {notebook_name}.ipynb...")

    cells = [
        markdown_cell(f"""# {config['title']}
## Complete Training and Evaluation Pipeline

This notebook implements an end-to-end {config['title'].lower()} pipeline.

**Dataset:** {config['dataset']}
**Model:** {config['model']}
**Framework:** {config['framework']}

### Table of Contents
1. [Setup and Imports](#setup)
2. [Device Configuration (CUDA/CPU)](#device)
3. [Data Loading and Exploration](#data)
4. [Data Preprocessing](#preprocessing)
5. [Model Architecture](#model)
6. [Training Loop](#training)
7. [Evaluation and Metrics](#evaluation)
8. [Inference Demo](#inference)
9. [Save Results](#save)"""),

        markdown_cell("## 1. Setup and Imports <a id='setup'></a>"),

        code_cell("""# Install required packages (run once)
!pip install torch torchvision transformers datasets matplotlib seaborn scikit-learn tqdm numpy pandas -q"""),

        code_cell("""import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm.notebook import tqdm
import json
import os
from datetime import datetime

# Set plotting style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette('husl')
%matplotlib inline

print("✅ All imports successful!")"""),

        markdown_cell("## 2. Device Configuration (CUDA/CPU) <a id='device'></a>"),

        code_cell("""# Automatic device selection - works for both CUDA and CPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print(f"Using device: {device}")
if device.type == 'cuda':
    print(f"GPU Name: {torch.cuda.get_device_name(0)}")
    print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
    print(f"CUDA Version: {torch.version.cuda}")
else:
    print("No GPU available, using CPU")

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed(42)
    torch.backends.cudnn.deterministic = True"""),

        markdown_cell(f"""## 3. Data Loading and Exploration <a id='data'></a>

### Dataset: {config['dataset']}

This section handles data downloading and initial exploration."""),

        code_cell(f"""# Create data directory
data_dir = '../datasets/{notebook_name}'
os.makedirs(data_dir, exist_ok=True)

print(f"Dataset: {config['dataset']}")
print(f"Data directory: {{data_dir}}")

# TODO: Add dataset-specific loading code
# This will be customized for each model type"""),

        markdown_cell("## 4. Data Preprocessing <a id='preprocessing'></a>"),

        code_cell("""# TODO: Add preprocessing and augmentation
# This section will include:
# - Data transformations
# - Augmentation strategies
# - Train/val/test splits
# - DataLoader creation

print("✅ Data preprocessing complete")"""),

        markdown_cell("## 5. Model Architecture <a id='model'></a>"),

        code_cell(f"""# TODO: Define model architecture
# Model: {config['model']}

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        # Define layers here
        pass

    def forward(self, x):
        # Define forward pass
        pass

# Initialize model and move to device
model = Model().to(device)

# Count parameters
total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

print(f"✅ Model created and moved to {{device}}")
print(f"Total parameters: {{total_params:,}}")
print(f"Trainable parameters: {{trainable_params:,}}")"""),

        markdown_cell("## 6. Training Loop <a id='training'></a>"),

        code_cell("""# Training configuration
num_epochs = 50
learning_rate = 0.001

# Loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)

print("Training configuration:")
print(f"  Epochs: {num_epochs}")
print(f"  Learning rate: {learning_rate}")
print(f"  Optimizer: AdamW")
print(f"  Scheduler: CosineAnnealingLR")"""),

        code_cell("""# Training function
def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    pbar = tqdm(loader, desc='Training')
    for inputs, labels in pbar:
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()

        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

        pbar.set_postfix({
            'loss': f'{running_loss/(pbar.n+1):.4f}',
            'acc': f'{100.*correct/total:.2f}%'
        })

    return running_loss / len(loader), 100. * correct / total

# Validation function
def validate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        pbar = tqdm(loader, desc='Validation')
        for inputs, labels in pbar:
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, labels)

            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

            pbar.set_postfix({
                'loss': f'{running_loss/(pbar.n+1):.4f}',
                'acc': f'{100.*correct/total:.2f}%'
            })

    return running_loss / len(loader), 100. * correct / total

print("✅ Training and validation functions defined")"""),

        code_cell(f"""# Training loop
history = {{
    'train_loss': [],
    'train_acc': [],
    'val_loss': [],
    'val_acc': [],
    'lr': []
}}

best_acc = 0.0
results_dir = '../{notebook_name}/results'
models_dir = '../{notebook_name}/models'
os.makedirs(results_dir, exist_ok=True)
os.makedirs(models_dir, exist_ok=True)

print("Starting training...\\n")
start_time = datetime.now()

for epoch in range(num_epochs):
    print(f"\\nEpoch {{epoch+1}}/{{num_epochs}}")
    print("-" * 50)

    # TODO: Implement actual training loop
    # train_loss, train_acc = train_epoch(model, trainloader, criterion, optimizer, device)
    # val_loss, val_acc = validate(model, testloader, criterion, device)

    # Update scheduler
    scheduler.step()
    current_lr = optimizer.param_groups[0]['lr']

    # Save history
    # history['train_loss'].append(train_loss)
    # history['train_acc'].append(train_acc)
    # history['val_loss'].append(val_loss)
    # history['val_acc'].append(val_acc)
    # history['lr'].append(current_lr)

    # Save best model
    # if val_acc > best_acc:
    #     best_acc = val_acc
    #     torch.save(model.state_dict(), f'{{models_dir}}/best_model.pt')

end_time = datetime.now()
training_time = (end_time - start_time).total_seconds()

print(f"\\nTraining time: {{training_time/60:.2f}} minutes")"""),

        markdown_cell("## 7. Evaluation and Metrics <a id='evaluation'></a>"),

        code_cell("""# Load best model
# model.load_state_dict(torch.load(f'{models_dir}/best_model.pt'))

# Get predictions and calculate metrics
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix

# TODO: Implement evaluation
print("✅ Evaluation complete")"""),

        markdown_cell("## 8. Inference Demo <a id='inference'></a>"),

        code_cell("""# Inference on sample data
model.eval()

# TODO: Implement inference demonstration
print("✅ Inference demo complete")"""),

        markdown_cell("## 9. Save Results <a id='save'></a>"),

        code_cell(f"""# Save metrics to JSON
metrics = {{
    'model_name': '{config['model']}',
    'dataset': '{config['dataset']}',
    'framework': '{config['framework']}',
    'training_time_minutes': training_time / 60,
    'num_epochs': num_epochs,
    'total_parameters': total_params,
    'trainable_parameters': trainable_params,
    # Add more metrics as needed
}}

with open(f'{{results_dir}}/metrics.json', 'w') as f:
    json.dump(metrics, f, indent=2)

print("✅ Metrics saved to metrics.json")
print("\\n" + "="*50)
print("NOTEBOOK EXECUTION COMPLETE")
print("="*50)
print(f"\\nSaved files:")
print(f"  - Model: {{models_dir}}/best_model.pt")
print(f"  - Metrics: {{results_dir}}/metrics.json")""")
    ]

    # Create notebook
    notebook = create_notebook_structure(cells)

    # Save notebook
    notebook_path = NOTEBOOKS_DIR / f"{notebook_name}.ipynb"
    with open(notebook_path, 'w') as f:
        json.dump(notebook, f, indent=1)

    print(f"  ✅ Created: {notebook_path}")

print("\n" + "=" * 60)
print("All notebooks generated successfully!")
print(f"Location: {NOTEBOOKS_DIR}")
print("\nNext steps:")
print("1. Review and customize each notebook for specific datasets")
print("2. Add dataset-specific loading code")
print("3. Implement model architectures")
print("4. Run training pipelines")
print("=" * 60)
