# Object Detection Project - Pascal VOC/COCO

## 1. Problem Definition & Use Case

**Problem:** Detect and localize multiple objects in images with bounding boxes and class labels.

**Use Case:** Object detection powers:
- Autonomous driving (pedestrian, vehicle detection)
- Video surveillance and security
- Retail analytics (customer tracking, inventory)
- Medical imaging (tumor detection)
- Industrial inspection (defect localization)

**Business Impact:** Enables automated monitoring systems, reduces inspection time by 80%, and improves safety in critical applications.

## 2. Dataset Acquisition & Preprocessing

### Primary Datasets

1. **COCO (Common Objects in Context)**
```python
from pycocotools.coco import COCO
import torchvision.datasets as datasets

coco_train = datasets.CocoDetection(
    root='./data/coco/train2017',
    annFile='./data/coco/annotations/instances_train2017.json'
)
```
- 330K images, 80 object categories
- 1.5M object instances

2. **Pascal VOC 2012**
```python
voc_train = datasets.VOCDetection(
    root='./data',
    year='2012',
    image_set='train',
    download=True
)
```
- 11,530 images, 20 object categories
- Average 2.4 objects per image

### Data Schema
```python
{
    'image': np.ndarray,           # Shape: (H, W, 3)
    'bboxes': List[List[float]],   # [[x1, y1, x2, y2], ...]
    'labels': List[int],           # Class IDs
    'scores': List[float],         # Confidence scores (inference)
    'image_id': int,               # Unique identifier
    'metadata': dict               # Width, height, filename
}
```

### Preprocessing Pipeline
```python
import albumentations as A
from albumentations.pytorch import ToTensorV2

transform_train = A.Compose([
    A.RandomResizedCrop(640, 640, scale=(0.8, 1.0)),
    A.HorizontalFlip(p=0.5),
    A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    A.RandomRotate90(p=0.2),
    A.OneOf([
        A.MotionBlur(p=0.2),
        A.MedianBlur(blur_limit=3, p=0.1),
        A.GaussianBlur(p=0.2),
    ], p=0.3),
    A.Normalize(mean=[0.485, 0.456, 0.406], 
                std=[0.229, 0.224, 0.225]),
    ToTensorV2()
], bbox_params=A.BboxParams(format='pascal_voc', 
                           label_fields=['labels']))
```

### Advanced Data Augmentation
```python
# Mosaic augmentation (YOLO style)
def mosaic_augmentation(images, boxes, labels):
    """Combine 4 images into one mosaic"""
    mosaic_img = np.zeros((640, 640, 3))
    mosaic_boxes = []
    
    for i, (img, box) in enumerate(zip(images, boxes)):
        # Place in quadrant
        x_offset = (i % 2) * 320
        y_offset = (i // 2) * 320
        mosaic_img[y_offset:y_offset+320, 
                  x_offset:x_offset+320] = cv2.resize(img, (320, 320))
        # Adjust boxes
        adjusted_boxes = box.copy()
        adjusted_boxes[:, [0, 2]] = adjusted_boxes[:, [0, 2]] * 0.5 + x_offset
        adjusted_boxes[:, [1, 3]] = adjusted_boxes[:, [1, 3]] * 0.5 + y_offset
        mosaic_boxes.extend(adjusted_boxes)
    
    return mosaic_img, mosaic_boxes
```

## 3. Baseline Models

### Faster R-CNN Baseline
```python
import torchvision.models.detection as detection

model = detection.fasterrcnn_resnet50_fpn(
    pretrained=True,
    num_classes=91  # 90 classes + background
)

# Fine-tune for custom dataset
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
in_features = model.roi_heads.box_predictor.cls_score.in_features
model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
```
**Expected Performance:** mAP@0.5 = 50-55% on COCO

### RetinaNet Baseline
```python
model = detection.retinanet_resnet50_fpn(
    pretrained=True,
    num_classes=91
)
```
**Expected Performance:** mAP@0.5 = 48-53% on COCO

## 4. Advanced/Stretch Models

### 1. YOLOv8 (State-of-the-Art Speed/Accuracy)
```python
from ultralytics import YOLO

# Load pretrained model
model = YOLO('yolov8x.pt')

# Train on custom dataset
model.train(
    data='path/to/dataset.yaml',
    epochs=300,
    imgsz=640,
    batch=16
)
```

### 2. DETR (Transformer-based)
```python
from transformers import DetrForObjectDetection

model = DetrForObjectDetection.from_pretrained(
    'facebook/detr-resnet-101',
    num_labels=91,
    ignore_mismatched_sizes=True
)
```

### 3. Detectron2 (Meta's Framework)
```python
from detectron2.config import get_cfg
from detectron2.engine import DefaultTrainer

cfg = get_cfg()
cfg.merge_from_file("configs/COCO-Detection/faster_rcnn_X_101_32x8d_FPN_3x.yaml")
cfg.DATASETS.TRAIN = ("coco_train",)
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 80

trainer = DefaultTrainer(cfg)
trainer.resume_or_load(resume=False)
trainer.train()
```

### 4. EfficientDet
```python
import timm

model = timm.create_model(
    'tf_efficientdet_d7',
    pretrained=True,
    num_classes=80,
    checkpoint_path='efficientdet_d7-f05bf714.pth'
)
```

**Target Performance:** mAP@0.5 = 65%+ on COCO

## 5. Training Details

### Data Loading Pipeline
```python
from torch.utils.data import DataLoader

def collate_fn(batch):
    return tuple(zip(*batch))

train_loader = DataLoader(
    dataset,
    batch_size=8,
    shuffle=True,
    num_workers=4,
    collate_fn=collate_fn,
    pin_memory=True
)
```

### Training Configuration
```python
config = {
    'base_lr': 0.001,
    'weight_decay': 0.0001,
    'momentum': 0.9,
    'scheduler': 'StepLR',
    'step_size': 3,
    'gamma': 0.1,
    'epochs': 50,
    'warmup_epochs': 3,
    'gradient_clip': 5.0,
    'anchor_sizes': [[32], [64], [128], [256], [512]],
    'aspect_ratios': [[0.5, 1.0, 2.0]] * 5,
    'iou_threshold': 0.5,
    'score_threshold': 0.05,
    'nms_threshold': 0.5,
    'max_detections': 100
}
```

### Loss Functions
```python
class DetectionLoss:
    def __init__(self):
        self.classification_loss = nn.CrossEntropyLoss()
        self.regression_loss = nn.SmoothL1Loss()
        
    def forward(self, predictions, targets):
        cls_loss = self.classification_loss(
            predictions['class_logits'], 
            targets['labels']
        )
        reg_loss = self.regression_loss(
            predictions['box_regression'], 
            targets['boxes']
        )
        return cls_loss + reg_loss
```

## 6. Evaluation Metrics & Validation Strategy

### Core Metrics
```python
from pycocotools.cocoeval import COCOeval

def evaluate_coco(model, data_loader):
    coco_evaluator = COCOeval(coco_gt, coco_dt, 'bbox')
    coco_evaluator.evaluate()
    coco_evaluator.accumulate()
    coco_evaluator.summarize()
    
    metrics = {
        'mAP@0.5:0.95': coco_evaluator.stats[0],
        'mAP@0.5': coco_evaluator.stats[1],
        'mAP@0.75': coco_evaluator.stats[2],
        'mAP_small': coco_evaluator.stats[3],
        'mAP_medium': coco_evaluator.stats[4],
        'mAP_large': coco_evaluator.stats[5],
        'AR@1': coco_evaluator.stats[6],
        'AR@10': coco_evaluator.stats[7],
        'AR@100': coco_evaluator.stats[8]
    }
    return metrics
```

### Custom Metrics
```python
def calculate_metrics(predictions, ground_truth):
    metrics = {
        'precision': [],
        'recall': [],
        'f1_score': [],
        'iou': []
    }
    
    for pred, gt in zip(predictions, ground_truth):
        tp, fp, fn = calculate_tp_fp_fn(pred, gt)
        precision = tp / (tp + fp + 1e-10)
        recall = tp / (tp + fn + 1e-10)
        f1 = 2 * precision * recall / (precision + recall + 1e-10)
        
        metrics['precision'].append(precision)
        metrics['recall'].append(recall)
        metrics['f1_score'].append(f1)
        metrics['iou'].append(calculate_iou(pred['boxes'], gt['boxes']))
    
    return {k: np.mean(v) for k, v in metrics.items()}
```

### Validation Strategy
- **Cross-validation** on smaller datasets (VOC)
- **Single validation set** for large datasets (COCO)
- **Multi-scale evaluation** [480, 640, 800]
- **Test-time augmentation** with horizontal flip

## 7. Experiment Tracking & Reproducibility

### Weights & Biases Integration
```python
import wandb
from wandb import Image as WandbImage

wandb.init(project="object-detection", config=config)

# Log predictions
def log_predictions(images, predictions, ground_truth):
    wandb_images = []
    for img, pred, gt in zip(images, predictions, ground_truth):
        boxes_data = {
            "predictions": {
                "box_data": [{
                    "position": {
                        "minX": box[0],
                        "minY": box[1],
                        "maxX": box[2],
                        "maxY": box[3]
                    },
                    "class_id": int(label),
                    "scores": {"confidence": float(score)}
                } for box, label, score in zip(
                    pred['boxes'], pred['labels'], pred['scores']
                )]
            }
        }
        wandb_images.append(WandbImage(img, boxes=boxes_data))
    
    wandb.log({"predictions": wandb_images})
```

### TensorBoard Logging
```python
from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter('runs/object_detection')

def visualize_predictions(writer, images, outputs, epoch):
    for idx, (img, output) in enumerate(zip(images, outputs)):
        img_with_boxes = draw_boxes(img, output['boxes'], 
                                   output['labels'], output['scores'])
        writer.add_image(f'predictions/image_{idx}', 
                        img_with_boxes, epoch)
```

## 8. Deployment Pathway

### Option 1: Real-time API with FastAPI
```python
from fastapi import FastAPI, File, UploadFile
import torch
import cv2
import numpy as np

app = FastAPI()

@app.post("/detect")
async def detect_objects(file: UploadFile = File(...)):
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    # Preprocess
    input_tensor = transform(img).unsqueeze(0)
    
    # Inference
    with torch.no_grad():
        predictions = model(input_tensor)[0]
    
    # Filter predictions
    keep = predictions['scores'] > 0.5
    boxes = predictions['boxes'][keep].cpu().numpy()
    labels = predictions['labels'][keep].cpu().numpy()
    scores = predictions['scores'][keep].cpu().numpy()
    
    return {
        "boxes": boxes.tolist(),
        "labels": labels.tolist(),
        "scores": scores.tolist()
    }
```

### Option 2: Streamlit Demo
```python
import streamlit as st
import torch
from PIL import Image

st.title("Object Detection Demo")

uploaded_file = st.file_uploader("Choose an image...", type=['jpg', 'jpeg', 'png'])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_column_width=True)
    
    if st.button('Detect Objects'):
        with st.spinner('Detecting...'):
            predictions = detect_objects(image)
            annotated_image = draw_predictions(image, predictions)
            st.image(annotated_image, caption='Detection Results')
            
            # Display statistics
            st.write(f"Found {len(predictions['boxes'])} objects")
            for label, score in zip(predictions['labels'], predictions['scores']):
                st.write(f"- {CLASSES[label]}: {score:.2f}")
```

### Option 3: Edge Deployment with ONNX
```python
# Export to ONNX
dummy_input = torch.randn(1, 3, 640, 640)
torch.onnx.export(
    model,
    dummy_input,
    "detector.onnx",
    opset_version=11,
    input_names=['input'],
    output_names=['boxes', 'labels', 'scores'],
    dynamic_axes={
        'input': {0: 'batch_size'},
        'boxes': {0: 'num_detections'},
        'labels': {0: 'num_detections'},
        'scores': {0: 'num_detections'}
    }
)

# Inference with ONNX Runtime
import onnxruntime as ort

session = ort.InferenceSession("detector.onnx")
outputs = session.run(None, {'input': input_tensor.numpy()})
```

### Video Processing Pipeline
```python
import cv2

def process_video(video_path, output_path):
    cap = cv2.VideoCapture(video_path)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # Detect objects
        detections = detect_objects(frame)
        
        # Draw boxes
        annotated_frame = draw_boxes(frame, detections)
        out.write(annotated_frame)
    
    cap.release()
    out.release()
```

## 9. Extensions & Research Directions

### Advanced Techniques

1. **Few-Shot Object Detection**
```python
# Meta-learning approach for new classes with few examples
from meta_rcnn import MetaRCNN

model = MetaRCNN(n_way=5, k_shot=5)
model.meta_train(support_set, query_set)
```

2. **Semi-Supervised Learning**
```python
# Pseudo-labeling for unlabeled data
def generate_pseudo_labels(model, unlabeled_data):
    pseudo_labels = []
    for img in unlabeled_data:
        with torch.no_grad():
            pred = model(img)
        # Filter high-confidence predictions
        confident = pred['scores'] > 0.9
        pseudo_labels.append({
            'boxes': pred['boxes'][confident],
            'labels': pred['labels'][confident]
        })
    return pseudo_labels
```

3. **Multi-Scale Training**
```python
scales = [480, 512, 544, 576, 608, 640, 672, 704, 736, 768, 800]

for epoch in range(epochs):
    # Random scale each epoch
    scale = random.choice(scales)
    resize = transforms.Resize((scale, scale))
    # Train with resized images
```

4. **Anchor-Free Detection**
- FCOS (Fully Convolutional One-Stage)
- CenterNet
- CornerNet

### Novel Experiments
- **3D Object Detection** from monocular images
- **Panoptic Segmentation** (instance + semantic)
- **Temporal consistency** in video streams
- **Cross-domain adaptation** (synthetic to real)
- **Adversarial robustness** testing

### Industry Applications
- **Retail**: Customer behavior analysis
- **Manufacturing**: Quality control automation
- **Healthcare**: Medical device tracking
- **Agriculture**: Crop monitoring with drones
- **Smart Cities**: Traffic flow optimization

## 10. Portfolio Polish

### Project Structure
```
object_detection/
├── README.md
├── configs/
│   ├── faster_rcnn.yaml
│   ├── yolo.yaml
│   └── detr.yaml
├── data/
│   ├── download.sh
│   └── prepare_coco.py
├── models/
│   ├── __init__.py
│   ├── backbone.py
│   ├── detector.py
│   └── losses.py
├── utils/
│   ├── visualization.py
│   ├── metrics.py
│   └── augmentation.py
├── train.py
├── evaluate.py
├── inference.py
├── notebooks/
│   ├── data_exploration.ipynb
│   ├── error_analysis.ipynb
│   └── results_visualization.ipynb
├── deployment/
│   ├── api.py
│   ├── streamlit_app.py
│   └── Dockerfile
└── tests/
    ├── test_model.py
    └── test_metrics.py
```

### Visualization Components
```python
import matplotlib.pyplot as plt
import matplotlib.patches as patches

def visualize_detections(image, boxes, labels, scores, class_names):
    fig, ax = plt.subplots(1, figsize=(12, 8))
    ax.imshow(image)
    
    colors = plt.cm.hsv(np.linspace(0, 1, len(class_names)))
    
    for box, label, score in zip(boxes, labels, scores):
        x1, y1, x2, y2 = box
        rect = patches.Rectangle(
            (x1, y1), x2-x1, y2-y1,
            linewidth=2,
            edgecolor=colors[label],
            facecolor='none'
        )
        ax.add_patch(rect)
        
        # Add label
        ax.text(x1, y1-5, 
               f'{class_names[label]}: {score:.2f}',
               color='white',
               fontsize=10,
               bbox=dict(facecolor=colors[label], alpha=0.5))
    
    ax.axis('off')
    plt.tight_layout()
    return fig
```

### Performance Analysis Dashboard
```python
def create_performance_dashboard(results):
    fig = plt.figure(figsize=(15, 10))
    
    # mAP by class
    ax1 = plt.subplot(2, 3, 1)
    ax1.bar(range(len(results['ap_per_class'])), 
           results['ap_per_class'])
    ax1.set_title('AP per Class')
    
    # Precision-Recall curve
    ax2 = plt.subplot(2, 3, 2)
    ax2.plot(results['recall'], results['precision'])
    ax2.set_title('Precision-Recall Curve')
    
    # Detection speed
    ax3 = plt.subplot(2, 3, 3)
    ax3.bar(['GPU', 'CPU'], results['inference_time'])
    ax3.set_title('Inference Time (ms)')
    
    # Size analysis
    ax4 = plt.subplot(2, 3, 4)
    sizes = ['small', 'medium', 'large']
    ax4.bar(sizes, [results[f'mAP_{s}'] for s in sizes])
    ax4.set_title('mAP by Object Size')
    
    # Confusion matrix
    ax5 = plt.subplot(2, 3, 5)
    ax5.imshow(results['confusion_matrix'], cmap='Blues')
    ax5.set_title('Class Confusion Matrix')
    
    # Loss curves
    ax6 = plt.subplot(2, 3, 6)
    ax6.plot(results['train_loss'], label='Train')
    ax6.plot(results['val_loss'], label='Val')
    ax6.set_title('Training Progress')
    ax6.legend()
    
    plt.tight_layout()
    return fig
```

### README Showcase
```markdown
# Real-Time Object Detection System

[![Demo](https://img.shields.io/badge/demo-online-green)](https://your-demo.com)
[![Paper](https://img.shields.io/badge/paper-arxiv-red)](https://arxiv.org)

## Key Features
- 🚀 Real-time inference (30+ FPS)
- 📊 State-of-the-art accuracy (65% mAP)
- 🔧 Easy deployment with Docker
- 📱 Mobile-optimized models

## Quick Start
```bash
# Install dependencies
pip install -r requirements.txt

# Download pretrained model
python download_model.py

# Run inference
python detect.py --image path/to/image.jpg --model yolov8

# Start API server
uvicorn api:app --reload
```

## Results

| Model | mAP@0.5 | mAP@0.5:0.95 | FPS (V100) | Parameters |
|-------|---------|--------------|------------|------------|
| YOLOv8-X | 68.2% | 52.3% | 42 | 68.2M |
| Faster R-CNN | 64.5% | 45.1% | 15 | 41.8M |
| DETR | 66.1% | 48.9% | 28 | 41.3M |
```