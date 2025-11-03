# Environment Setup Guide

This guide will help you set up your development environment with all necessary libraries and datasets for the 12 ML projects.

## Table of Contents
1. [Prerequisites](#prerequisites)
2. [Python Environment Setup](#python-environment-setup)
3. [Installing Libraries](#installing-libraries)
4. [Dataset Acquisition](#dataset-acquisition)
5. [Verification](#verification)
6. [Troubleshooting](#troubleshooting)

---

## Prerequisites

### System Requirements
- **Operating System:** Windows 10/11, Linux, or macOS
- **Python:** 3.8 or higher (3.9 or 3.10 recommended)
- **GPU (Optional but Recommended):**
  - NVIDIA GPU with CUDA support (for faster training)
  - At least 6GB VRAM (8GB+ recommended)
- **RAM:** 16GB minimum, 32GB recommended
- **Storage:** 100GB+ free space (for datasets and models)

### Check Your System
```bash
# Check Python version
python --version  # Should be 3.8+

# Check if CUDA is available (if you have NVIDIA GPU)
nvidia-smi

# Check available disk space
# Windows:
dir
# Linux/macOS:
df -h
```

---

## Python Environment Setup

### Option 1: Using Conda (Recommended)

```bash
# Create a new conda environment
conda create -n ml_portfolio python=3.10

# Activate the environment
conda activate ml_portfolio

# Install pip in the environment
conda install pip
```

### Option 2: Using venv

```bash
# Create virtual environment
python -m venv ml_portfolio_env

# Activate the environment
# Windows:
ml_portfolio_env\Scripts\activate
# Linux/macOS:
source ml_portfolio_env/bin/activate
```

---

## Installing Libraries

### Complete Requirements File

Create a `requirements.txt` file in your repository root:

```txt
# Core Deep Learning Frameworks
torch==2.1.0
torchvision==0.16.0
torchaudio==2.1.0

# Alternative: For CUDA 11.8 (if you have NVIDIA GPU)
# Install with: pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Hugging Face Ecosystem (for NLP and multimodal)
transformers==4.35.0
datasets==2.14.0
tokenizers==0.15.0
accelerate==0.24.0

# Advanced Vision Models
timm==0.9.12
opencv-python==4.8.1.78
albumentations==1.3.1
Pillow==10.1.0

# Audio Processing
librosa==0.10.1
soundfile==0.12.1
torchaudio==2.1.0
audioread==3.0.1

# NLP Additional Tools
sentencepiece==0.1.99
sacremoses==0.1.1
nltk==3.8.1
spacy==3.7.2

# Recommender Systems
scikit-surprise==1.1.3
implicit==0.7.2
lightfm==1.17

# Time Series & Anomaly Detection
statsmodels==0.14.0
prophet==1.1.5
pyod==1.1.2

# Data Processing & Visualization
numpy==1.24.3
pandas==2.1.3
scipy==1.11.4
scikit-learn==1.3.2
matplotlib==3.8.2
seaborn==0.13.0
plotly==5.18.0

# Web Application Frameworks
gradio==4.8.0
streamlit==1.28.2
fastapi==0.104.1
uvicorn==0.24.0
pydantic==2.5.0

# Experiment Tracking & MLOps
wandb==0.16.0
mlflow==2.8.1
tensorboard==2.15.1

# Jupyter & Notebooks
jupyter==1.0.0
notebook==7.0.6
ipywidgets==8.1.1
ipython==8.18.1

# Utilities
tqdm==4.66.1
pyyaml==6.0.1
python-dotenv==1.0.0
requests==2.31.0
aiohttp==3.9.1

# Testing & Quality
pytest==7.4.3
black==23.11.0
flake8==6.1.0

# Additional Tools
redis==5.0.1  # For caching in web app
python-multipart==0.0.6  # For file uploads in FastAPI
```

### Installation Steps

```bash
# Make sure you're in your virtual environment
# Then install all requirements

pip install -r requirements.txt

# If you have CUDA GPU, install PyTorch with CUDA support:
# For CUDA 11.8:
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# For CUDA 12.1:
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Download spaCy language model
python -m spacy download en_core_web_sm
```

### Installation Time Estimate
- **With fast internet:** 15-30 minutes
- **With slow internet:** 45-90 minutes
- **Total download size:** ~5-8 GB

---

## Dataset Acquisition

### Auto-Downloading Datasets (No Manual Download Needed)

These datasets will automatically download when you run the training scripts:

#### 1. Image Classification (Project 01)
**CIFAR-10 & CIFAR-100**
- **Auto-downloads:** Yes
- **Size:** ~170 MB (CIFAR-10), ~160 MB (CIFAR-100)
- **Location:** `./01_Image_Classification/data/`
- **First run command:**
  ```bash
  cd 01_Image_Classification
  python train.py --model simple_cnn --epochs 1  # Downloads automatically
  ```

#### 2. Text Classification (Project 04)
**AG News, IMDb, SST-2 (via Hugging Face)**
- **Auto-downloads:** Yes
- **Size:** ~50-200 MB depending on dataset
- **Location:** `~/.cache/huggingface/datasets/`
- **First run:**
  ```python
  from datasets import load_dataset
  dataset = load_dataset('ag_news')  # Auto-downloads
  ```

#### 3. Text Generation (Project 05)
**WikiText, OpenWebText (via Hugging Face)**
- **Auto-downloads:** Yes
- **Size:** ~100 MB - 1 GB
- **Location:** `~/.cache/huggingface/datasets/`

#### 4. Machine Translation (Project 06)
**WMT14 EN-DE, Multi30k**
- **Auto-downloads:** Yes via Hugging Face
- **Size:** ~500 MB - 2 GB

#### 5. Speech Emotion Recognition (Project 07)
**RAVDESS, TESS (if using public versions)**
- **Status:** May require manual download
- **See manual section below**

#### 6. Automatic Speech Recognition (Project 08)
**LibriSpeech (via torchaudio)**
- **Auto-downloads:** Yes
- **Size:** ~60 GB for full dataset (can use smaller subsets)
- **Location:** `./08_Automatic_Speech_Recognition/data/`
- **Recommended:** Use dev-clean subset (337 MB) for testing
  ```python
  import torchaudio
  dataset = torchaudio.datasets.LIBRISPEECH("./data", url="dev-clean", download=True)
  ```

#### 7. Recommender System (Project 09)
**MovieLens (100K, 1M, 10M, 25M)**
- **Auto-downloads:** Yes via surprise library
- **Size:** 5 MB (100K) to 250 MB (25M)
- **Location:** `~/.surprise_data/`
- **First run:**
  ```python
  from surprise import Dataset
  data = Dataset.load_builtin('ml-100k')  # Auto-downloads
  ```

#### 8. Time Series Forecasting (Project 10)
**Standard time series datasets**
- **Auto-downloads:** Depends on specific dataset
- **Common sources:**
  - Yahoo Finance (via yfinance): Auto-downloads
  - UCI datasets: Auto-downloads
  - Kaggle: Manual download (see below)

#### 9. Anomaly Detection (Project 11)
**NSL-KDD, Credit Card Fraud**
- **Auto-downloads:** Partial (depends on source)
- **May need manual download from Kaggle**

---

### Manual Download Datasets

These require manual download and setup:

#### Object Detection (Project 02)

**COCO Dataset**
- **Website:** https://cocodataset.org/#download
- **Size:** ~25 GB (images + annotations)
- **What to download:**
  - 2017 Train images [18GB]
  - 2017 Val images [1GB]
  - 2017 Train/Val annotations [241MB]
- **Location:** `./02_Object_Detection/data/coco/`
- **Commands:**
  ```bash
  cd 02_Object_Detection
  mkdir -p data/coco
  cd data/coco

  # Download (Linux/macOS)
  wget http://images.cocodataset.org/zips/train2017.zip
  wget http://images.cocodataset.org/zips/val2017.zip
  wget http://images.cocodataset.org/annotations/annotations_trainval2017.zip

  # Unzip
  unzip train2017.zip
  unzip val2017.zip
  unzip annotations_trainval2017.zip
  ```

**Alternative: Pascal VOC (Smaller)**
- **Website:** http://host.robots.ox.ac.uk/pascal/VOC/
- **Size:** ~2 GB
- **Recommended for testing**

#### Instance Segmentation (Project 03)

**COCO Dataset** (same as Object Detection)
- Use same dataset as Project 02
- Or use **Cityscapes** for urban scenes
  - Website: https://www.cityscapes-dataset.com/
  - Requires registration
  - Size: ~11 GB

#### Speech Emotion Recognition (Project 07)

**RAVDESS Dataset**
- **Website:** https://zenodo.org/record/1188976
- **Size:** ~5 GB
- **Location:** `./07_Speech_Emotion_Recognition/data/ravdess/`
- **Download:**
  ```bash
  # Download from Zenodo
  # Manual download required - register and download from website
  ```

**Alternative: TESS**
- **Website:** https://tspace.library.utoronto.ca/handle/1807/24487
- **Size:** ~500 MB

#### Multimodal Fusion (Project 12)

**MELD (Multimodal EmotionLines Dataset)**
- **Website:** https://affective-meld.github.io/
- **Size:** ~10 GB
- **What to download:**
  - Video files
  - Audio files
  - Text transcripts

**CMU-MOSI**
- **Website:** http://immortal.multicomp.cs.cmu.edu/
- **Size:** ~2 GB
- **Requires registration**

---

### Optional: Large Datasets for Advanced Training

#### ImageNet (for Project 01)
- **Size:** ~150 GB
- **Requires:** Academic registration
- **Website:** https://image-net.org/
- **Note:** Only needed for ImageNet-scale experiments
- **Alternative:** Use Tiny ImageNet (200 classes, ~250 MB)
  ```bash
  wget http://cs231n.stanford.edu/tiny-imagenet-200.zip
  ```

#### Amazon Product Reviews (for Project 09)
- **Website:** https://nijianmo.github.io/amazon/index.html
- **Size:** Varies (1 GB - 20 GB depending on category)
- **What to download:** Choose specific product categories
- **Location:** `./09_Recommender_System/data/amazon/`

---

## Verification

### Test Your Setup

Create and run this test script: `test_environment.py`

```python
"""
Environment verification script
Tests all major dependencies
"""

import sys

def test_imports():
    """Test if all critical packages can be imported"""

    print("Testing Python version...")
    print(f"Python {sys.version}")
    assert sys.version_info >= (3, 8), "Python 3.8+ required"
    print("✓ Python version OK\n")

    print("Testing PyTorch...")
    import torch
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA version: {torch.version.cuda}")
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    print("✓ PyTorch OK\n")

    print("Testing torchvision...")
    import torchvision
    print(f"torchvision version: {torchvision.__version__}")
    print("✓ torchvision OK\n")

    print("Testing Hugging Face...")
    import transformers
    print(f"transformers version: {transformers.__version__}")
    print("✓ Hugging Face OK\n")

    print("Testing timm...")
    import timm
    print(f"timm version: {timm.__version__}")
    print("✓ timm OK\n")

    print("Testing librosa (audio)...")
    import librosa
    print(f"librosa version: {librosa.__version__}")
    print("✓ librosa OK\n")

    print("Testing scikit-learn...")
    import sklearn
    print(f"scikit-learn version: {sklearn.__version__}")
    print("✓ scikit-learn OK\n")

    print("Testing pandas...")
    import pandas as pd
    print(f"pandas version: {pd.__version__}")
    print("✓ pandas OK\n")

    print("Testing matplotlib...")
    import matplotlib
    print(f"matplotlib version: {matplotlib.__version__}")
    print("✓ matplotlib OK\n")

    print("Testing Gradio...")
    import gradio as gr
    print(f"Gradio version: {gr.__version__}")
    print("✓ Gradio OK\n")

    print("Testing surprise (recommender)...")
    import surprise
    print(f"surprise version: {surprise.__version__}")
    print("✓ surprise OK\n")

    print("=" * 50)
    print("✓✓✓ ALL TESTS PASSED! ✓✓✓")
    print("=" * 50)
    print("\nYour environment is ready!")

if __name__ == "__main__":
    try:
        test_imports()
    except ImportError as e:
        print(f"\n❌ ERROR: {e}")
        print("\nPlease install missing package with:")
        print(f"pip install {str(e).split()[-1]}")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        sys.exit(1)
```

**Run the test:**
```bash
python test_environment.py
```

### Test Dataset Download

Create `test_datasets.py`:

```python
"""
Test dataset downloading
"""

def test_cifar10():
    """Test CIFAR-10 download"""
    print("Testing CIFAR-10 download...")
    from torchvision import datasets

    dataset = datasets.CIFAR10(
        root='./test_data',
        train=True,
        download=True
    )
    print(f"✓ CIFAR-10 downloaded: {len(dataset)} samples")

def test_movielens():
    """Test MovieLens download"""
    print("\nTesting MovieLens download...")
    from surprise import Dataset

    data = Dataset.load_builtin('ml-100k')
    print("✓ MovieLens-100K downloaded")

def test_huggingface_dataset():
    """Test Hugging Face dataset"""
    print("\nTesting Hugging Face dataset...")
    from datasets import load_dataset

    dataset = load_dataset('imdb', split='train[:10]')  # Just 10 samples
    print(f"✓ IMDb dataset accessible: {len(dataset)} samples")

if __name__ == "__main__":
    print("Testing dataset downloads...")
    print("This will download small test datasets.\n")

    test_cifar10()
    test_movielens()
    test_huggingface_dataset()

    print("\n" + "=" * 50)
    print("✓✓✓ ALL DATASET TESTS PASSED! ✓✓✓")
    print("=" * 50)
```

**Run the test:**
```bash
python test_datasets.py
```

---

## Quick Start Commands

Once everything is installed, start training your first model:

```bash
# 1. Activate your environment
conda activate ml_portfolio  # or: source ml_portfolio_env/bin/activate

# 2. Start with Image Classification (easiest)
cd 01_Image_Classification
python train.py --model simple_cnn --epochs 5 --batch-size 64

# 3. Monitor with WandB (optional)
wandb login  # Enter your WandB API key
python train.py --model resnet18 --epochs 10 --use-wandb

# 4. Try Recommender System (fast training)
cd ../09_Recommender_System
python train.py --model svd --epochs 20
```

---

## Troubleshooting

### CUDA/GPU Issues

**Problem:** PyTorch not detecting GPU
```bash
# Check CUDA version
nvidia-smi

# Reinstall PyTorch with correct CUDA version
pip uninstall torch torchvision torchaudio
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

**Problem:** CUDA out of memory
```bash
# Reduce batch size
python train.py --batch-size 32  # instead of 128
```

### Import Errors

**Problem:** `ModuleNotFoundError`
```bash
# Make sure you're in the virtual environment
conda activate ml_portfolio

# Reinstall the package
pip install <package-name>
```

**Problem:** Version conflicts
```bash
# Create fresh environment
conda create -n ml_portfolio_fresh python=3.10
conda activate ml_portfolio_fresh
pip install -r requirements.txt
```

### Dataset Download Issues

**Problem:** Slow download speeds
- Use a VPN or different network
- Download during off-peak hours
- For large datasets (COCO, ImageNet), consider using academic networks

**Problem:** Disk space issues
```bash
# Check disk space
df -h  # Linux/macOS
dir    # Windows

# Use smaller dataset versions:
# - Use CIFAR-10 instead of ImageNet
# - Use MovieLens-100K instead of 25M
# - Use LibriSpeech dev-clean instead of full dataset
```

### Memory Issues

**Problem:** Out of RAM during training
```bash
# Reduce batch size
python train.py --batch-size 16

# Reduce number of workers
python train.py --num-workers 2

# Use gradient accumulation
python train.py --batch-size 32 --accumulate-grad-batches 4
```

---

## Storage Requirements Summary

| Component | Size | Required? |
|-----------|------|-----------|
| Python packages | 5-8 GB | ✅ Yes |
| CIFAR-10/100 | 170 MB | ✅ Yes |
| MovieLens-100K | 5 MB | ✅ Yes |
| COCO Dataset | 25 GB | ⚠️ Optional |
| LibriSpeech (dev-clean) | 337 MB | ✅ Yes |
| ImageNet | 150 GB | ❌ Optional |
| Models (checkpoints) | 10-50 GB | ✅ Yes |
| **Total (minimum)** | **~20 GB** | |
| **Total (recommended)** | **~100 GB** | |

---

## Next Steps

After setup is complete:

1. ✅ Run environment verification: `python test_environment.py`
2. ✅ Test dataset downloads: `python test_datasets.py`
3. ✅ Train your first model: `cd 01_Image_Classification && python train.py`
4. ✅ Start creating notebooks (follow IMPLEMENTATION_PLAN.md)
5. ✅ Begin web app development

---

## Additional Resources

- **PyTorch Installation:** https://pytorch.org/get-started/locally/
- **Hugging Face Datasets:** https://huggingface.co/docs/datasets/
- **CUDA Toolkit:** https://developer.nvidia.com/cuda-downloads
- **Gradio Documentation:** https://gradio.app/docs/
- **WandB Setup:** https://docs.wandb.ai/quickstart
