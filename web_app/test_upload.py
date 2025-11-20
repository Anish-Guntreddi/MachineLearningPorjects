#!/usr/bin/env python3
"""
Interactive Demo Script - Test Image Upload and Prediction

This script demonstrates how to:
1. Upload images to the ML Portfolio API
2. Get predictions with confidence scores
3. Test batch predictions
4. View model information

Usage:
    python test_upload.py path/to/image.jpg
    python test_upload.py --batch image1.jpg image2.jpg image3.jpg
    python test_upload.py --info
"""

import sys
import requests
import json
from pathlib import Path
from typing import List


API_BASE_URL = "http://localhost:8000/api"
MODEL_NAME = "image_classification"


def check_api_health():
    """Check if the API is running"""
    try:
        response = requests.get(f"{API_BASE_URL.replace('/api', '')}/api/health", timeout=5)
        return response.status_code == 200
    except requests.exceptions.RequestException:
        return False


def get_model_info():
    """Get model information"""
    try:
        response = requests.get(f"{API_BASE_URL}/models/{MODEL_NAME}")
        if response.status_code == 200:
            return response.json()
        else:
            print(f"❌ Error: {response.status_code}")
            print(response.json())
            return None
    except requests.exceptions.RequestException as e:
        print(f"❌ Connection error: {e}")
        return None


def predict_single_image(image_path: str):
    """
    Upload and predict a single image

    Args:
        image_path: Path to image file
    """
    path = Path(image_path)

    if not path.exists():
        print(f"❌ File not found: {image_path}")
        return None

    if not path.is_file():
        print(f"❌ Not a file: {image_path}")
        return None

    print(f"\n📤 Uploading: {path.name}")
    print(f"   Size: {path.stat().st_size / 1024:.2f} KB")

    try:
        with open(path, 'rb') as f:
            files = {'file': (path.name, f, 'image/jpeg')}

            response = requests.post(
                f"{API_BASE_URL}/models/{MODEL_NAME}/predict",
                files=files,
                timeout=30
            )

        if response.status_code == 200:
            return response.json()
        else:
            print(f"❌ Error: {response.status_code}")
            print(response.json())
            return None

    except requests.exceptions.RequestException as e:
        print(f"❌ Request error: {e}")
        return None


def predict_batch_images(image_paths: List[str]):
    """
    Upload and predict multiple images at once

    Args:
        image_paths: List of paths to image files
    """
    files_to_upload = []

    for image_path in image_paths:
        path = Path(image_path)
        if not path.exists():
            print(f"⚠️  Skipping (not found): {image_path}")
            continue
        files_to_upload.append(path)

    if not files_to_upload:
        print("❌ No valid files to upload")
        return None

    print(f"\n📤 Uploading {len(files_to_upload)} images...")

    try:
        files = [
            ('files', (path.name, open(path, 'rb'), 'image/jpeg'))
            for path in files_to_upload
        ]

        response = requests.post(
            f"{API_BASE_URL}/models/{MODEL_NAME}/batch",
            files=files,
            timeout=60
        )

        # Close file handles
        for _, (_, f, _) in files:
            f.close()

        if response.status_code == 200:
            return response.json()
        else:
            print(f"❌ Error: {response.status_code}")
            print(response.json())
            return None

    except requests.exceptions.RequestException as e:
        print(f"❌ Request error: {e}")
        return None


def print_prediction_result(result: dict, image_name: str = None):
    """Pretty print prediction results"""

    if image_name:
        print(f"\n{'='*60}")
        print(f"📸 Image: {image_name}")
        print('='*60)

    if result.get('status') == 'success':
        pred_result = result['result']

        # Top prediction
        print(f"\n🎯 PREDICTION: {pred_result['prediction'].upper()}")
        print(f"   Confidence: {pred_result['confidence']*100:.2f}%")

        # Top 5 predictions
        print(f"\n📊 TOP 5 PREDICTIONS:")
        for i, pred in enumerate(pred_result.get('top5_predictions', []), 1):
            emoji = '🥇' if i == 1 else '🥈' if i == 2 else '🥉' if i == 3 else '  '
            bar_length = int(pred['confidence'] * 40)
            bar = '█' * bar_length + '░' * (40 - bar_length)
            print(f"{emoji} {pred['class']:12s} {bar} {pred['confidence']*100:6.2f}%")

        # Processing time
        if 'processing_time_ms' in result:
            print(f"\n⏱️  Processing time: {result['processing_time_ms']:.2f}ms")
    else:
        print(f"\n❌ Status: {result.get('status', 'unknown')}")
        if 'detail' in result:
            print(f"   Error: {result['detail']}")


def print_model_info(info: dict):
    """Pretty print model information"""
    print("\n" + "="*60)
    print("📋 MODEL INFORMATION")
    print("="*60)

    if info.get('status') == 'success':
        model = info['model']

        print(f"\n🤖 Name: {model.get('display_name', model['name'])}")
        print(f"📝 Description: {model.get('description', 'N/A')}")
        print(f"🏷️  Category: {model.get('category', 'N/A')}")
        print(f"✅ Status: {model.get('status', 'N/A')}")

        if 'model_info' in model:
            minfo = model['model_info']
            print(f"\n🔧 Architecture: {minfo.get('architecture', 'N/A')}")
            print(f"📊 Dataset: {minfo.get('dataset', 'N/A')}")
            print(f"🎯 Classes: {minfo.get('num_classes', 'N/A')}")

            if 'classes' in minfo:
                print(f"\n📑 Available Classes:")
                for i, cls in enumerate(minfo['classes'], 1):
                    print(f"   {i:2d}. {cls}")
    else:
        print(f"❌ Status: {info.get('status', 'unknown')}")


def main():
    """Main function"""
    print("\n🚀 ML Portfolio - Image Upload Test")
    print("="*60)

    # Check if API is running
    print("\n🔍 Checking API status...")
    if not check_api_health():
        print("❌ API is not running!")
        print("\nPlease start the application first:")
        print("  cd web_app")
        print("  ./start.sh")
        print("\nOr manually:")
        print("  docker-compose up")
        sys.exit(1)

    print("✅ API is running!")

    # Parse arguments
    if len(sys.argv) < 2:
        print("\n📖 Usage:")
        print("  Single image:  python test_upload.py image.jpg")
        print("  Batch images:  python test_upload.py --batch img1.jpg img2.jpg img3.jpg")
        print("  Model info:    python test_upload.py --info")
        print("\nExample:")
        print("  python test_upload.py ~/Pictures/my_dog.jpg")
        sys.exit(0)

    # Handle --info flag
    if '--info' in sys.argv:
        info = get_model_info()
        if info:
            print_model_info(info)
        sys.exit(0)

    # Handle --batch flag
    if '--batch' in sys.argv:
        batch_idx = sys.argv.index('--batch')
        image_paths = sys.argv[batch_idx + 1:]

        if not image_paths:
            print("❌ No images provided for batch processing")
            sys.exit(1)

        result = predict_batch_images(image_paths)

        if result and result.get('status') == 'success':
            print(f"\n✅ Processed {result.get('count', 0)} images")

            for i, pred_result in enumerate(result.get('results', []), 1):
                print_prediction_result(
                    {'status': 'success', 'result': pred_result},
                    image_name=f"Image {i}"
                )

        sys.exit(0)

    # Single image prediction
    image_path = sys.argv[1]
    result = predict_single_image(image_path)

    if result:
        print_prediction_result(result, image_name=Path(image_path).name)
        print("\n✅ Test complete!")
    else:
        print("\n❌ Test failed!")
        sys.exit(1)


if __name__ == "__main__":
    main()
