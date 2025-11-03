# Implementation Plan: ML Portfolio Enhancement

## Overview
This document outlines the implementation plan for adding evaluation notebooks and creating a unified web application to showcase all 12 machine learning projects in the portfolio.

## Phase 1: Jupyter Notebooks for Model Evaluation (Week 1-2)

### 1.1 Notebook Structure Template
Create a standardized notebook structure for all projects:

```
XX_Project_Name/
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_model_training.ipynb
│   ├── 03_evaluation_metrics.ipynb
│   └── 04_inference_demo.ipynb
```

### 1.2 Notebook Components

#### For Each Project, Create:

**01_data_exploration.ipynb**
- Dataset loading and visualization
- Data distribution analysis
- Sample visualization (images/text/audio/etc.)
- Statistical summaries
- Class balance analysis

**02_model_training.ipynb**
- Import from existing train.py
- Training loop visualization
- Loss curves plotting
- Learning rate schedules
- Live training progress with tqdm

**03_evaluation_metrics.ipynb**
- Load trained models from checkpoints
- Comprehensive evaluation metrics:
  - Classification: Confusion matrix, ROC curves, PR curves, per-class metrics
  - Regression: Residual plots, prediction vs actual
  - Recommender: Precision@K, Recall@K, NDCG curves
  - Multimodal: Ablation studies, modality contributions
- Error analysis with worst/best predictions
- Feature importance/attention visualizations

**04_inference_demo.ipynb**
- Interactive widgets for testing
- Single sample predictions
- Batch inference examples
- Prediction confidence visualization
- Grad-CAM or attention maps (where applicable)

### 1.3 Project-Specific Notebooks

#### Computer Vision Projects (01-03)
- **Additional cells for:**
  - Image augmentation visualization
  - Feature map visualization
  - Model architecture diagrams
  - Transfer learning comparison

#### NLP Projects (04-06)
- **Additional cells for:**
  - Token analysis and vocabulary stats
  - Attention weight heatmaps
  - Word embeddings visualization (t-SNE/UMAP)
  - Generation sampling strategies

#### Audio Projects (07-08)
- **Additional cells for:**
  - Waveform and spectrogram plots
  - MFCC feature visualization
  - Audio playback widgets
  - Real-time inference demo

#### Recommender System (09)
- **Additional cells for:**
  - User-item interaction matrix visualization
  - Embedding space visualization
  - Cold start analysis
  - A/B test simulation

#### Time Series (10)
- **Additional cells for:**
  - Temporal pattern analysis
  - Forecast vs actual plots
  - Seasonality decomposition
  - Rolling window predictions

#### Anomaly Detection (11)
- **Additional cells for:**
  - Anomaly score distributions
  - Threshold selection analysis
  - False positive/negative analysis
  - Reconstruction error visualization

#### Multimodal Fusion (12)
- **Additional cells for:**
  - Cross-modal attention maps
  - Modality dropout experiments
  - Feature fusion visualization
  - Synchronized data display

## Phase 2: Unified Web Application (Week 3-4)

### 2.1 Technology Stack

**Frontend:**
- **Framework:** Gradio or Streamlit (recommended: Gradio for better customization)
- **Alternative:** React + FastAPI (for more control)

**Backend:**
- **Framework:** FastAPI
- **Model Serving:** PyTorch with model caching
- **Async Processing:** For heavy inference tasks

**Deployment:**
- **Option 1:** Hugging Face Spaces (easiest)
- **Option 2:** Docker + Cloud Run/AWS EC2
- **Option 3:** Gradio Cloud

### 2.2 Application Architecture

```
ml_portfolio_app/
├── app.py                      # Main application entry
├── config/
│   └── models_config.yaml     # Model paths and configurations
├── components/
│   ├── image_classification.py
│   ├── object_detection.py
│   ├── instance_segmentation.py
│   ├── text_classification.py
│   ├── text_generation.py
│   ├── machine_translation.py
│   ├── speech_emotion.py
│   ├── speech_recognition.py
│   ├── recommender.py
│   ├── time_series.py
│   ├── anomaly_detection.py
│   └── multimodal_fusion.py
├── utils/
│   ├── model_loader.py        # Centralized model loading
│   ├── preprocessing.py       # Input preprocessing
│   └── visualization.py       # Output visualization
├── static/
│   ├── css/
│   └── assets/
├── templates/
│   └── descriptions/          # Model descriptions from READMEs
└── requirements.txt
```

### 2.3 Web App Features

#### Main Interface Design

**Layout Structure:**
```python
# Pseudo-code structure
with gr.Blocks(theme="soft") as app:
    # Header
    gr.Markdown("# ML Portfolio Showcase")

    # Tab-based navigation
    with gr.Tabs():
        with gr.Tab("Image Classification"):
            # Content
        with gr.Tab("Object Detection"):
            # Content
        # ... 12 tabs total
```

#### Each Tab Should Include:

1. **Model Description Section**
   - Brief overview (from README)
   - Key features and capabilities
   - Training dataset information
   - Model architecture details
   - Performance metrics

2. **Interactive Demo Section**
   - Input component (specific to modality)
   - Parameter controls (if applicable)
   - Inference button
   - Output display with visualization

3. **Example Gallery**
   - Pre-loaded examples
   - "Try these examples" functionality
   - Best/worst case demonstrations

4. **Technical Details (Collapsible)**
   - Model size and inference time
   - Hardware requirements
   - API usage example
   - Link to notebook and code

### 2.4 Component-Specific Interfaces

#### Image Classification Tab
```python
# Interface components:
- Image upload (drag & drop)
- Model selection dropdown (ResNet, EfficientNet, ViT, etc.)
- Top-K predictions slider
- Output: Class predictions with confidence bars
- Grad-CAM visualization toggle
```

#### Object Detection Tab
```python
# Interface components:
- Image upload
- Confidence threshold slider
- NMS threshold slider
- Class filter checkboxes
- Output: Annotated image with bounding boxes
- Detection statistics table
```

#### Text Classification Tab
```python
# Interface components:
- Text input box
- Model selection (BERT variants)
- Output: Class probabilities
- Attention visualization
- Token importance highlighting
```

#### Text Generation Tab
```python
# Interface components:
- Prompt text input
- Temperature slider
- Max length slider
- Top-p/Top-k sampling controls
- Output: Generated text with streaming
- Multiple generation samples
```

#### Machine Translation Tab
```python
# Interface components:
- Source text input
- Language pair selector
- Beam search width
- Output: Translated text
- Attention matrix visualization
- Alternative translations
```

#### Speech Emotion Recognition Tab
```python
# Interface components:
- Audio file upload OR microphone recording
- Audio waveform display
- Output: Emotion predictions with confidence
- Spectrogram visualization
- Feature importance plots
```

#### Speech Recognition Tab
```python
# Interface components:
- Audio upload/recording
- Language selector
- Output: Transcribed text
- Word-level confidence scores
- Alternative transcriptions
```

#### Recommender System Tab
```python
# Interface components:
- User ID input OR user preferences
- Number of recommendations slider
- Algorithm selector (CF, SVD, Neural)
- Output: Recommended items
- Explanation of recommendations
- Similar users/items
```

#### Time Series Forecasting Tab
```python
# Interface components:
- CSV upload OR example dataset
- Forecast horizon slider
- Model selector
- Output: Interactive time series plot
- Confidence intervals
- Component decomposition
```

#### Anomaly Detection Tab
```python
# Interface components:
- Data input (file or live)
- Sensitivity threshold slider
- Output: Anomaly scores plot
- Detected anomalies highlighted
- Reconstruction error visualization
```

#### Multimodal Fusion Tab
```python
# Interface components:
- Image upload
- Audio upload
- Text input
- Output: Unified prediction
- Modality contribution bars
- Cross-modal attention maps
```

## Phase 3: Model Preparation (Week 2-3, parallel with Phase 1)

### 3.0 Model Preparation Strategy: Hybrid Approach

**Key Decision:** Models will be trained separately from notebooks, then loaded and analyzed in notebooks.

#### Why This Approach?
- ✅ Notebooks don't need to run 100-epoch training (too slow for interactive work)
- ✅ Can iterate quickly on hyperparameters in notebooks
- ✅ Clear separation between heavy training and analysis/deployment
- ✅ Easy to retrain models without breaking web app

### 3.1 Training Workflow

#### Step 1: Initial Model Training (Command Line)

Train models using existing `train.py` scripts **before** creating notebooks:

```bash
# Example: Image Classification
cd 01_Image_Classification

# Train multiple model architectures
python train.py --model resnet50 --epochs 100 --batch-size 128 --use-wandb
python train.py --model efficientnet_b0 --epochs 100 --batch-size 128 --use-wandb
python train.py --model vit_small --epochs 100 --batch-size 64 --use-wandb

# This saves checkpoints to ./checkpoints/
# - resnet50_best.pth
# - efficientnet_b0_best.pth
# - vit_small_best.pth
```

**Run in Background:** These long training runs can happen while you develop notebooks.

#### Step 2: Notebook Integration

Notebooks **load pre-trained checkpoints** rather than training from scratch:

**In `02_model_training.ipynb`:**

```python
# Cell 1: Load existing checkpoint
import torch
from models import get_model

checkpoint = torch.load('checkpoints/resnet50_best.pth')
model = get_model('resnet50', num_classes=10)
model.load_state_dict(checkpoint['model_state_dict'])

print(f"Loaded model from epoch {checkpoint['epoch']}")
print(f"Best accuracy: {checkpoint['best_acc']:.2f}%")

# Cell 2: OPTIONAL - Continue training (fine-tuning)
# Only if you want to experiment with different hyperparameters
from train import Trainer

fine_tune_config = {
    'model_name': 'resnet50',
    'epochs': 10,  # Just a few more epochs
    'learning_rate': 1e-5,  # Lower learning rate for fine-tuning
    'batch_size': 128
}

trainer = Trainer(fine_tune_config)
trainer.start_epoch = checkpoint['epoch']
trainer.best_acc = checkpoint['best_acc']
trainer.model.load_state_dict(checkpoint['model_state_dict'])
# trainer.train()  # Uncomment to actually fine-tune

# Cell 3: Quick hyperparameter experiments
# Test different settings without full training
for lr in [1e-3, 1e-4, 1e-5]:
    print(f"\nTesting learning rate: {lr}")
    # Run for just 2-3 epochs to see impact
    quick_trainer = Trainer({**fine_tune_config, 'learning_rate': lr, 'epochs': 3})
    results = quick_trainer.train()
    print(f"Quick test accuracy: {results['acc1']:.2f}%")
```

**In `03_evaluation_metrics.ipynb`:**

```python
# Cell 1: Load and compare multiple models
import pandas as pd
import matplotlib.pyplot as plt

models_to_evaluate = {
    'ResNet-50': 'checkpoints/resnet50_best.pth',
    'EfficientNet-B0': 'checkpoints/efficientnet_b0_best.pth',
    'ViT-Small': 'checkpoints/vit_small_best.pth'
}

results = []
for name, checkpoint_path in models_to_evaluate.items():
    checkpoint = torch.load(checkpoint_path)
    model = get_model(checkpoint['config']['model_name'])
    model.load_state_dict(checkpoint['model_state_dict'])

    # Evaluate on test set
    test_metrics = evaluate_model(model, test_loader)

    results.append({
        'Model': name,
        'Accuracy': test_metrics['acc1'],
        'Params (M)': sum(p.numel() for p in model.parameters()) / 1e6,
        'Model Size (MB)': os.path.getsize(checkpoint_path) / 1e6
    })

# Cell 2: Model comparison table
df_results = pd.DataFrame(results)
print(df_results)

# Cell 3: Model selection for production
# Calculate composite score balancing accuracy, size, and speed
df_results['Score'] = (
    0.5 * df_results['Accuracy'] / df_results['Accuracy'].max() +
    0.3 * (1 - df_results['Params (M)'] / df_results['Params (M)'].max()) +
    0.2 * (1 - df_results['Model Size (MB)'] / df_results['Model Size (MB)'].max())
)

best_model = df_results.loc[df_results['Score'].idxmax()]
print(f"\n🏆 Best model for production: {best_model['Model']}")
print(f"   Accuracy: {best_model['Accuracy']:.2f}%")
print(f"   Size: {best_model['Model Size (MB)']:.1f} MB")
```

### 3.2 Model Optimization Pipeline

Create a **separate script** for production optimization:

**`scripts/prepare_production_models.py`:**

```python
"""
Model Optimization Pipeline
This script:
1. Loads trained checkpoints
2. Optimizes for deployment (quantization, fusion, etc.)
3. Measures inference speed
4. Saves to model_zoo/ with metadata
"""

import torch
import torch.quantization
import json
import os
import time
from pathlib import Path

def optimize_for_deployment(checkpoint_path, output_path):
    """Optimize model for production use"""

    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    model = get_model(checkpoint['config']['model_name'])
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    # Step 1: Fuse operations (Conv + BatchNorm + ReLU)
    # This speeds up inference
    try:
        model = torch.quantization.fuse_modules(
            model,
            [['conv', 'bn', 'relu']]
        )
    except:
        print("Could not fuse modules, skipping...")

    # Step 2: Dynamic Quantization (INT8)
    # Reduces model size and speeds up inference
    quantized_model = torch.quantization.quantize_dynamic(
        model,
        {torch.nn.Linear, torch.nn.Conv2d},
        dtype=torch.qint8
    )

    # Step 3: Convert to TorchScript
    # Faster loading and inference
    example_input = torch.randn(1, 3, 224, 224)
    scripted_model = torch.jit.trace(quantized_model, example_input)

    # Step 4: Measure inference time
    inference_times = []
    with torch.no_grad():
        # Warmup
        for _ in range(10):
            _ = scripted_model(example_input)

        # Measure
        for _ in range(100):
            start = time.time()
            _ = scripted_model(example_input)
            inference_times.append((time.time() - start) * 1000)

    avg_inference_time = sum(inference_times) / len(inference_times)

    # Step 5: Save optimized model
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    torch.jit.save(scripted_model, output_path)

    # Step 6: Create metadata file
    metadata = {
        'model_name': checkpoint['config']['model_name'],
        'original_accuracy': float(checkpoint['best_acc']),
        'num_parameters': sum(p.numel() for p in model.parameters()),
        'model_size_mb': os.path.getsize(output_path) / 1e6,
        'input_shape': [3, 224, 224],
        'num_classes': checkpoint['config']['num_classes'],
        'avg_inference_time_ms': avg_inference_time,
        'preprocessing': 'normalize_imagenet',
        'optimization': 'quantized_int8_torchscript'
    }

    metadata_path = output_path.replace('.pt', '_metadata.json')
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)

    print(f"✓ Saved optimized model to {output_path}")
    print(f"  Size: {metadata['model_size_mb']:.1f} MB")
    print(f"  Inference: {avg_inference_time:.2f} ms")

    return metadata

# Run for all projects
if __name__ == "__main__":
    projects = [
        ('01_Image_Classification', 'efficientnet_b0', 'image_classification'),
        ('02_Object_Detection', 'yolov5', 'object_detection'),
        # ... add all 12 projects
    ]

    for project_dir, model_name, output_category in projects:
        checkpoint_path = f"{project_dir}/checkpoints/{model_name}_best.pth"
        output_path = f"model_zoo/{output_category}/{model_name}.pt"

        if os.path.exists(checkpoint_path):
            print(f"\n📦 Processing {project_dir} - {model_name}")
            optimize_for_deployment(checkpoint_path, output_path)
        else:
            print(f"⚠️  Checkpoint not found: {checkpoint_path}")
```

**Usage:**

```bash
# After training models, run optimization
python scripts/prepare_production_models.py

# This populates model_zoo/ with optimized models
```

### 3.3 Model Zoo Structure

Create a centralized repository for production models:

```
model_zoo/
├── config.json                           # Model registry
├── image_classification/
│   ├── efficientnet_b0.pt               # Optimized TorchScript model
│   ├── efficientnet_b0_metadata.json    # Model info
│   ├── resnet50.pt
│   └── resnet50_metadata.json
├── object_detection/
│   ├── yolov5.pt
│   └── yolov5_metadata.json
├── text_classification/
│   ├── bert_base.pt
│   └── bert_base_metadata.json
├── text_generation/
│   └── gpt2_small.pt
├── machine_translation/
│   └── transformer_en_de.pt
├── speech_emotion/
│   └── lstm_emotion.pt
├── speech_recognition/
│   └── wav2vec2.pt
├── recommender/
│   ├── ncf_model.pt
│   └── matrix_factorization.pt
├── time_series/
│   └── lstm_forecaster.pt
├── anomaly_detection/
│   └── autoencoder.pt
└── multimodal_fusion/
    └── cross_attention_model.pt
```

**`model_zoo/config.json`:**

```json
{
  "image_classification": {
    "default": "efficientnet_b0",
    "options": {
      "efficientnet_b0": {
        "display_name": "EfficientNet-B0",
        "description": "Best balance of accuracy and speed",
        "accuracy": 94.1,
        "inference_time_ms": 12
      },
      "resnet50": {
        "display_name": "ResNet-50",
        "description": "Classic architecture, good accuracy",
        "accuracy": 93.2,
        "inference_time_ms": 15
      }
    }
  },
  "text_classification": {
    "default": "bert_base",
    "options": {
      "bert_base": {
        "display_name": "BERT-Base",
        "description": "State-of-the-art text understanding",
        "accuracy": 92.5,
        "inference_time_ms": 45
      }
    }
  }
  // ... etc for all 12 projects
}
```

### 3.4 Model Loading Utility

Create a centralized model loader for the web app:

**`ml_portfolio_app/utils/model_loader.py`:**

```python
"""
Centralized model loading with caching
"""

import torch
import json
from pathlib import Path
from typing import Dict, Optional

class ModelRegistry:
    """Registry for all production models"""

    def __init__(self, model_zoo_path: str = "model_zoo"):
        self.model_zoo_path = Path(model_zoo_path)
        self.config = self._load_config()
        self.loaded_models = {}  # Cache for loaded models

    def _load_config(self) -> Dict:
        """Load model registry configuration"""
        config_path = self.model_zoo_path / "config.json"
        with open(config_path, 'r') as f:
            return json.load(f)

    def get_model(self, category: str, model_name: Optional[str] = None):
        """
        Load a model from the zoo

        Args:
            category: e.g., 'image_classification', 'text_generation'
            model_name: specific model, or None for default

        Returns:
            Loaded PyTorch model
        """
        # Use default if not specified
        if model_name is None:
            model_name = self.config[category]['default']

        # Check cache
        cache_key = f"{category}/{model_name}"
        if cache_key in self.loaded_models:
            return self.loaded_models[cache_key]

        # Load model
        model_path = self.model_zoo_path / category / f"{model_name}.pt"
        print(f"Loading model from {model_path}...")

        model = torch.jit.load(model_path, map_location='cpu')
        model.eval()

        # Cache for future use
        self.loaded_models[cache_key] = model

        return model

    def get_metadata(self, category: str, model_name: Optional[str] = None) -> Dict:
        """Get model metadata"""
        if model_name is None:
            model_name = self.config[category]['default']

        metadata_path = self.model_zoo_path / category / f"{model_name}_metadata.json"
        with open(metadata_path, 'r') as f:
            return json.load(f)

    def list_models(self, category: str) -> Dict:
        """List available models for a category"""
        return self.config[category]['options']

# Global registry instance
registry = ModelRegistry()
```

**Usage in Web App:**

```python
# In ml_portfolio_app/components/image_classification.py
from utils.model_loader import registry

def classify_image(image, model_choice):
    # Load selected model (cached after first load)
    model = registry.get_model('image_classification', model_choice)
    metadata = registry.get_metadata('image_classification', model_choice)

    # Run inference
    outputs = model(preprocess(image))

    return outputs, metadata
```

### 3.5 Updated Timeline

**Week 1: Model Training (Background)**
- Start training all models using existing `train.py` scripts
- Let these run in background/overnight
- Monitor with WandB

**Week 1-2: Notebook Development (Parallel)**
- Create notebooks that **load** pre-trained checkpoints
- Focus on analysis, visualization, and evaluation
- No need to wait for training to complete for notebook structure

**Week 2-3: Model Optimization**
- Once training completes, run optimization script
- Populate model_zoo/ with production models
- Test loading in notebooks

**Week 3-4: Web App Development**
- Use ModelRegistry to load from model_zoo/
- Build interfaces around optimized models
- All models ready and optimized

**Week 5: Deployment**
- Deploy with all models pre-loaded
- No training needed in production

## Phase 4: Integration and Testing (Week 4)

### 4.1 Integration Tasks
1. Connect notebooks to training scripts
2. Ensure model loading works across all components
3. Implement caching for model inference
4. Add progress indicators for long-running tasks

### 4.2 Testing Checklist
- [ ] All notebooks run without errors
- [ ] Model checkpoints load correctly
- [ ] Web app handles edge cases gracefully
- [ ] Inference time is acceptable (<5s per request)
- [ ] Memory usage is optimized
- [ ] Error messages are informative

### 4.3 Performance Optimization
- Implement request queuing for heavy models
- Add model warm-up on startup
- Use batch inference where possible
- Implement result caching

## Phase 5: Documentation and Deployment (Week 5)

### 5.1 Documentation
- Create user guide for web app
- Document API endpoints if applicable
- Add troubleshooting section
- Create video demo

### 5.2 Deployment Options

#### Option A: Hugging Face Spaces (Recommended for Portfolio)
```bash
# Steps:
1. Create HF Space
2. Add requirements.txt
3. Upload app.py and components
4. Configure space settings (GPU if needed)
```

#### Option B: Docker Deployment
```dockerfile
FROM python:3.9
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
CMD ["python", "app.py"]
```

#### Option C: Cloud Deployment (AWS/GCP/Azure)
- Use serverless for light models
- Use GPU instances for heavy models
- Implement auto-scaling

### 5.3 Monitoring
- Add logging for all inference requests
- Track model performance metrics
- Monitor resource usage
- Set up error alerting

## Implementation Timeline

### Week 1-2: Notebooks
- Day 1-3: Create template notebook
- Day 4-10: Implement for all 12 projects
- Day 11-14: Testing and refinement

### Week 3-4: Web Application
- Day 15-17: Setup framework and basic UI
- Day 18-24: Implement all 12 model interfaces
- Day 25-28: Integration and testing

### Week 5: Polish and Deploy
- Day 29-30: Documentation
- Day 31-32: Deployment setup
- Day 33-35: Final testing and launch

## Resource Requirements

### Development Environment
- GPU recommended for model training/testing
- 32GB+ RAM for loading multiple models
- 100GB+ storage for datasets and models

### Production Environment
- **Minimum:** 8GB RAM, 4 CPU cores
- **Recommended:** 16GB RAM, 8 CPU cores, 1 GPU
- **Storage:** 50GB for models and cache

## Success Criteria

### Notebooks
- ✅ All 12 projects have complete notebook sets
- ✅ Evaluation metrics are comprehensive and visualized
- ✅ Notebooks are well-documented and reproducible

### Web Application
- ✅ All 12 models are accessible via web interface
- ✅ Inference time < 5 seconds per request
- ✅ Interface is intuitive and responsive
- ✅ Documentation is complete and accessible
- ✅ Application handles errors gracefully

## Risk Mitigation

### Potential Risks and Solutions

1. **Large model sizes**
   - Solution: Implement model quantization and pruning
   - Alternative: Use cloud storage with on-demand loading

2. **Long inference times**
   - Solution: Add progress indicators and async processing
   - Alternative: Pre-compute results for examples

3. **Memory constraints**
   - Solution: Load models on-demand, not all at once
   - Alternative: Use model serving framework like TorchServe

4. **Dataset availability**
   - Solution: Use smaller, public datasets for demos
   - Alternative: Generate synthetic data for examples

## Next Steps

1. **Immediate Actions:**
   - Set up notebook template
   - Start with 1-2 projects as proof of concept
   - Choose web framework (Gradio vs Streamlit)

2. **First Milestone (End of Week 1):**
   - Complete notebooks for 4 projects
   - Basic web app structure ready

3. **Second Milestone (End of Week 3):**
   - All notebooks complete
   - Web app with 6+ models integrated

4. **Final Milestone (End of Week 5):**
   - Full deployment with all features
   - Documentation complete
   - Public URL available

## Appendix: Useful Resources

### Libraries and Tools
- **Gradio:** https://gradio.app/
- **Streamlit:** https://streamlit.io/
- **FastAPI:** https://fastapi.tiangolo.com/
- **Hugging Face Spaces:** https://huggingface.co/spaces

### Example Projects for Reference
- Gradio Model Zoo: https://github.com/gradio-app/gradio/tree/main/demo
- Streamlit Gallery: https://streamlit.io/gallery
- HF Space Examples: https://huggingface.co/spaces

### Notebook Best Practices
- Jupyter Book: https://jupyterbook.org/
- Papermill for parameterized notebooks: https://papermill.readthedocs.io/