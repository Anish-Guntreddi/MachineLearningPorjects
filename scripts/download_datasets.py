#!/usr/bin/env python3
"""
Dataset Download Script for ML Portfolio
Downloads all required datasets for the 12 ML projects
"""

import os
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

BASE_DIR = Path(__file__).parent.parent
DATASETS_DIR = BASE_DIR / "datasets"
DATASETS_DIR.mkdir(exist_ok=True)

print("=" * 70)
print("ML Portfolio - Dataset Download Script")
print("=" * 70)
print("\nThis script will download datasets for all 12 projects:")
print("1. CIFAR-10 (Image Classification)")
print("2. COCO (Object Detection)")
print("3. COCO (Instance Segmentation)")
print("4. IMDb (Text Classification)")
print("5. Custom Text (Text Generation)")
print("6. WMT14 (Machine Translation)")
print("7. RAVDESS (Speech Emotion Recognition)")
print("8. LibriSpeech (Automatic Speech Recognition)")
print("9. MovieLens-100K (Recommender System)")
print("10. Custom Time Series (Time Series Forecasting)")
print("11. Credit Card Fraud (Anomaly Detection)")
print("12. Custom Multimodal (Multimodal Fusion)")
print("\n" + "=" * 70)

def download_cifar10():
    """Download CIFAR-10 dataset using torchvision"""
    print("\n[1/12] Downloading CIFAR-10...")
    try:
        import torchvision
        data_dir = DATASETS_DIR / "cifar10"
        data_dir.mkdir(exist_ok=True)

        torchvision.datasets.CIFAR10(
            root=str(data_dir),
            train=True,
            download=True
        )
        torchvision.datasets.CIFAR10(
            root=str(data_dir),
            train=False,
            download=True
        )
        print("  ✅ CIFAR-10 downloaded successfully")
        return True
    except Exception as e:
        print(f"  ❌ Error downloading CIFAR-10: {e}")
        return False

def download_imdb():
    """Download IMDb dataset using Hugging Face datasets"""
    print("\n[4/12] Downloading IMDb dataset...")
    try:
        from datasets import load_dataset
        data_dir = DATASETS_DIR / "imdb"
        data_dir.mkdir(exist_ok=True)

        dataset = load_dataset('imdb', cache_dir=str(data_dir))
        print("  ✅ IMDb dataset downloaded successfully")
        print(f"  Train samples: {len(dataset['train'])}")
        print(f"  Test samples: {len(dataset['test'])}")
        return True
    except Exception as e:
        print(f"  ❌ Error downloading IMDb: {e}")
        return False

def download_movielens():
    """Download MovieLens-100K dataset"""
    print("\n[9/12] Downloading MovieLens-100K...")
    try:
        import requests
        import zipfile
        import io

        data_dir = DATASETS_DIR / "movielens"
        data_dir.mkdir(exist_ok=True)

        url = "https://files.grouplens.org/datasets/movielens/ml-100k.zip"
        print("  Downloading from GroupLens...")

        response = requests.get(url)
        z = zipfile.ZipFile(io.BytesIO(response.content))
        z.extractall(str(data_dir))

        print("  ✅ MovieLens-100K downloaded successfully")
        return True
    except Exception as e:
        print(f"  ❌ Error downloading MovieLens: {e}")
        return False

def setup_coco_instructions():
    """Provide instructions for COCO dataset (too large for auto-download)"""
    print("\n[2/12] COCO Dataset (Object Detection)")
    print("[3/12] COCO Dataset (Instance Segmentation)")
    print("  ℹ️  COCO dataset is very large (~25GB)")
    print("  Manual download instructions:")
    print("  1. Visit: https://cocodataset.org/#download")
    print("  2. Download 2017 Train/Val images and annotations")
    print(f"  3. Extract to: {DATASETS_DIR / 'coco'}")
    print("  OR use the following commands:")
    print("  ```bash")
    print(f"  mkdir -p {DATASETS_DIR / 'coco'}")
    print(f"  cd {DATASETS_DIR / 'coco'}")
    print("  wget http://images.cocodataset.org/zips/train2017.zip")
    print("  wget http://images.cocodataset.org/zips/val2017.zip")
    print("  wget http://images.cocodataset.org/annotations/annotations_trainval2017.zip")
    print("  unzip train2017.zip && unzip val2017.zip && unzip annotations_trainval2017.zip")
    print("  ```")

def setup_audio_instructions():
    """Provide instructions for audio datasets"""
    print("\n[7/12] RAVDESS Dataset (Speech Emotion Recognition)")
    print("  ℹ️  Manual download required:")
    print("  1. Visit: https://zenodo.org/record/1188976")
    print("  2. Download the RAVDESS dataset")
    print(f"  3. Extract to: {DATASETS_DIR / 'ravdess'}")

    print("\n[8/12] LibriSpeech Dataset (ASR)")
    print("  ℹ️  Large dataset - download via:")
    print("  ```bash")
    print(f"  mkdir -p {DATASETS_DIR / 'librispeech'}")
    print("  # Download dev-clean subset (346MB)")
    print("  wget https://www.openslr.org/resources/12/dev-clean.tar.gz")
    print(f"  tar -xzf dev-clean.tar.gz -C {DATASETS_DIR / 'librispeech'}")
    print("  ```")

def setup_other_datasets():
    """Setup instructions for remaining datasets"""
    print("\n[5/12] Text Generation Dataset")
    print("  ℹ️  Will use custom text or pre-trained GPT-2")
    print("  No download required - uses Hugging Face models")

    print("\n[6/12] Machine Translation Dataset (WMT14)")
    print("  ℹ️  Will use Hugging Face datasets")
    print("  Auto-downloads during training")

    print("\n[10/12] Time Series Forecasting")
    print("  ℹ️  Will generate synthetic time series data")
    print("  No download required")

    print("\n[11/12] Anomaly Detection (Credit Card Fraud)")
    print("  ℹ️  Download from Kaggle:")
    print("  1. Visit: https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud")
    print("  2. Download creditcard.csv")
    print(f"  3. Place in: {DATASETS_DIR / 'credit_fraud'}")

    print("\n[12/12] Multimodal Fusion")
    print("  ℹ️  Will use combination of existing datasets")
    print("  No additional download required")

def main():
    """Main download function"""
    print("\nStarting automatic downloads...")
    print("Note: Large datasets will show manual download instructions")
    print("-" * 70)

    # Create dataset directories
    for i in range(1, 13):
        project_name = f"{i:02d}_project"
        (DATASETS_DIR / project_name).mkdir(exist_ok=True)

    success_count = 0
    total_auto = 3  # Number of automatic downloads

    # Automatic downloads
    if download_cifar10():
        success_count += 1

    if download_imdb():
        success_count += 1

    if download_movielens():
        success_count += 1

    # Manual download instructions
    setup_coco_instructions()
    setup_audio_instructions()
    setup_other_datasets()

    # Summary
    print("\n" + "=" * 70)
    print("DATASET DOWNLOAD SUMMARY")
    print("=" * 70)
    print(f"Automatic downloads completed: {success_count}/{total_auto}")
    print(f"Datasets directory: {DATASETS_DIR}")
    print("\nReady to use:")
    if success_count > 0:
        print("  ✅ CIFAR-10 (Image Classification)")
        print("  ✅ IMDb (Text Classification)")
        print("  ✅ MovieLens-100K (Recommender System)")

    print("\nRequires manual download:")
    print("  ⏳ COCO (Object Detection & Instance Segmentation)")
    print("  ⏳ RAVDESS (Speech Emotion Recognition)")
    print("  ⏳ LibriSpeech (ASR)")
    print("  ⏳ Credit Card Fraud (Anomaly Detection)")

    print("\nAuto-downloaded during training:")
    print("  🔄 WMT14 (Machine Translation)")
    print("  🔄 Custom text (Text Generation)")

    print("\nGenerated synthetically:")
    print("  🔧 Time Series Data")
    print("  🔧 Multimodal Data")

    print("\n" + "=" * 70)
    print("Next steps:")
    print("1. Follow manual download instructions above for remaining datasets")
    print("2. Run Jupyter notebooks in the notebooks/ directory")
    print("3. Each notebook will handle its specific dataset loading")
    print("=" * 70)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nDownload interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n\nError: {e}")
        sys.exit(1)
