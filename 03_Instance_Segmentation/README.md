# Instance Segmentation Project - Cityscapes/COCO

## 1. Problem Definition & Use Case

**Problem:** Detect objects and delineate their exact pixel-level boundaries, distinguishing between individual instances of the same class.

**Use Case:** Instance segmentation enables:
- Autonomous driving (precise road understanding)
- Medical imaging (cell/organ segmentation)
- Robotics (object manipulation)
- Agriculture (individual plant analysis)
- Photo editing (automatic background removal)

**Business Impact:** Provides 10x more precise object understanding than bounding boxes, enabling surgical precision in automated systems and reducing manual annotation costs by 70%.

## 2. Dataset Acquisition & Preprocessing

### Primary Datasets

1. **COCO Instance Segmentation**
```python
from pycocotools import mask as maskUtils
import torchvision.datasets as datasets

coco_train = datasets.CocoDetection(
    root='./data/coco/train2017',
    annFile='./data/coco/annotations/instances_train2017.json',
    transforms=get_transform(train=True)
)
```
- 118K training images
- 80 object categories
- 860K segmented instances

2. **Cityscapes**
```python
from torchvision.datasets import Cityscapes

cityscapes_train = Cityscapes(
    './data/cityscapes',
    split='train',
    mode='fine',
    target_type='instance',
    transforms=transform
)
```
- 2,975 training images
- 19 classes for urban scene understanding
- Fine pixel-level annotations

### Data Schema
```python
{
    'image': torch.Tensor,           # Shape: (3, H, W)
    'masks': torch.Tensor,           # Shape: (N, H, W), binary masks
    'boxes': torch.Tensor,           # Shape: (N, 4), bounding boxes
    'labels': torch.Tensor,          # Shape: (N,), class labels
    'image_id': int,                 # Unique identifier
    'area': torch.Tensor,            # Shape: (N,), mask areas
    'iscrowd': torch.Tensor          # Shape: (N,), crowd annotations
}
```

### Preprocessing Pipeline
```python
import albumentations as A
from albumentations.pytorch import ToTensorV2

def get_transform(train=True):
    transforms = []
    
    if train:
        transforms.extend([
            A.RandomResizedCrop(512, 512, scale=(0.5, 1.0)),
            A.HorizontalFlip(p=0.5),
            A.ShiftScaleRotate(
                shift_limit=0.1,
                scale_limit=0.2,
                rotate_limit=30,
                p=0.5
            ),
            A.OneOf([
                A.GaussNoise(p=0.5),
                A.ISONoise(p=0.5),
                A.MultiplicativeNoise(p=0.5),
            ], p=0.2),
            A.OneOf([
                A.ElasticTransform(alpha=120, sigma=120 * 0.05),
                A.GridDistortion(),
                A.OpticalDistortion(distort_limit=2, shift_limit=0.5),
            ], p=0.3),
        ])
    
    transforms.extend([
        A.Normalize(mean=[0.485, 0.456, 0.406], 
                   std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ])
    
    return A.Compose(transforms, 
                    bbox_params=A.BboxParams(format='pascal_voc'),
                    keypoint_params=A.KeypointParams(format='xy'),
                    additional_targets={'masks': 'masks'})
```

### Mask Processing
```python
def process_masks(masks, image_size):
    """Convert polygon annotations to binary masks"""
    processed_masks = []
    
    for mask_data in masks:
        if isinstance(mask_data, dict):
            # RLE format
            mask = maskUtils.decode(mask_data)
        else:
            # Polygon format
            rles = maskUtils.frPyObjects(
                mask_data, image_size[0], image_size[1]
            )
            mask = maskUtils.decode(maskUtils.merge(rles))
        
        processed_masks.append(torch.from_numpy(mask))
    
    return torch.stack(processed_masks) if processed_masks else torch.zeros(0, *image_size)
```

## 3. Baseline Models

### Mask R-CNN Baseline
```python
import torchvision.models.detection as detection

model = detection.maskrcnn_resnet50_fpn(
    pretrained=True,
    num_classes=91
)

# Fine-tune for custom dataset
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor

# Replace mask predictor
mask_predictor_in = model.roi_heads.mask_predictor.conv5_mask.in_channels
hidden_layer = 256
model.roi_heads.mask_predictor = MaskRCNNPredictor(
    mask_predictor_in, hidden_layer, num_classes
)
```
**Expected Performance:** mAP@0.5 = 45-50% on COCO

### Simple Segmentation Network
```python
class SimpleInstanceSeg(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.backbone = models.resnet50(pretrained=True)
        
        # Remove final layers
        self.features = nn.Sequential(*list(self.backbone.children())[:-2])
        
        # FPN
        self.fpn = FPN(
            in_channels_list=[256, 512, 1024, 2048],
            out_channels=256
        )
        
        # Detection head
        self.detection = nn.Conv2d(256, num_classes + 1, 1)
        
        # Mask head
        self.mask = nn.Sequential(
            nn.Conv2d(256, 256, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, num_classes, 1)
        )
```
**Expected Performance:** mAP@0.5 = 35-40%

## 4. Advanced/Stretch Models

### 1. YOLACT++ (Real-time Instance Segmentation)
```python
from yolact import Yolact

model = Yolact()
model.load_weights('yolact_plus_base_54_800000.pth')

# Training configuration
cfg = Config({
    'backbone': 'resnet101',
    'fpn_channels': 256,
    'num_classes': 81,
    'mask_proto_net': [(256, 3, {'padding': 1})] * 3,
    'mask_dim': 32,
    'masks_to_train': 100,
    'mask_alpha': 6.125,
    'mask_proto_bias': False
})
```

### 2. SOLOv2 (Segmenting Objects by Locations)
```python
from mmdet.models import build_detector
from mmcv import Config

cfg = Config.fromfile('configs/solov2/solov2_r101_fpn_8gpu_3x.py')
model = build_detector(cfg.model)
```

### 3. Detectron2 Panoptic Segmentation
```python
from detectron2.config import get_cfg
from detectron2.projects.panoptic_deeplab import add_panoptic_deeplab_config

cfg = get_cfg()
add_panoptic_deeplab_config(cfg)
cfg.merge_from_file("configs/panoptic_fpn_R_101_3x.yaml")
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(
    "COCO-PanopticSegmentation/panoptic_fpn_R_101_3x.yaml"
)
```

### 4. Segment Anything Model (SAM)
```python
from segment_anything import sam_model_registry, SamPredictor

sam = sam_model_registry["vit_h"](checkpoint="sam_vit_h.pth")
predictor = SamPredictor(sam)

# Interactive segmentation
predictor.set_image(image)
masks, scores, logits = predictor.predict(
    point_coords=np.array([[x, y]]),
    point_labels=np.array([1]),  # 1 for foreground
    multimask_output=True
)
```

**Target Performance:** mAP@0.5 = 55%+ on COCO

## 5. Training Details

### Custom Data Loader
```python
class InstanceSegDataset(Dataset):
    def __init__(self, root, transforms=None):
        self.root = root
        self.transforms = transforms
        self.imgs = list(sorted(os.listdir(os.path.join(root, "images"))))
        self.masks = list(sorted(os.listdir(os.path.join(root, "masks"))))
    
    def __getitem__(self, idx):
        img_path = os.path.join(self.root, "images", self.imgs[idx])
        mask_path = os.path.join(self.root, "masks", self.masks[idx])
        
        img = Image.open(img_path).convert("RGB")
        mask = np.array(Image.open(mask_path))
        
        # Extract individual instances
        obj_ids = np.unique(mask)[1:]  # Remove background
        masks = mask == obj_ids[:, None, None]
        
        # Get bounding boxes
        boxes = []
        for i in range(len(obj_ids)):
            pos = np.where(masks[i])
            xmin = np.min(pos[1])
            xmax = np.max(pos[1])
            ymin = np.min(pos[0])
            ymax = np.max(pos[0])
            boxes.append([xmin, ymin, xmax, ymax])
        
        target = {
            'boxes': torch.as_tensor(boxes, dtype=torch.float32),
            'labels': torch.ones((len(obj_ids),), dtype=torch.int64),
            'masks': torch.as_tensor(masks, dtype=torch.uint8),
            'image_id': torch.tensor([idx]),
            'area': (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0]),
            'iscrowd': torch.zeros((len(obj_ids),), dtype=torch.int64)
        }
        
        if self.transforms:
            img, target = self.transforms(img, target)
        
        return img, target
```

### Training Configuration
```python
config = {
    'learning_rate': 0.0025,
    'weight_decay': 0.0001,
    'momentum': 0.9,
    'batch_size': 4,
    'num_epochs': 50,
    'warmup_iters': 1000,
    'warmup_factor': 0.001,
    'clip_gradients': 10.0,
    
    # Loss weights
    'loss_weights': {
        'loss_classifier': 1.0,
        'loss_box_reg': 1.0,
        'loss_mask': 1.0,
        'loss_objectness': 1.0,
        'loss_rpn_box_reg': 1.0
    },
    
    # Model specific
    'roi_batch_size_per_image': 512,
    'rpn_batch_size_per_image': 256,
    'rpn_positive_fraction': 0.5,
    'box_positive_fraction': 0.25,
    'bbox_reg_weights': (10.0, 10.0, 5.0, 5.0)
}
```

### Multi-Scale Training
```python
class MultiScaleTrainer:
    def __init__(self, model, scales=[480, 512, 544, 576, 608, 640]):
        self.model = model
        self.scales = scales
    
    def train_step(self, images, targets):
        # Random scale selection
        scale = random.choice(self.scales)
        
        # Resize images and targets
        resized_images = []
        resized_targets = []
        
        for img, target in zip(images, targets):
            h, w = img.shape[-2:]
            scale_factor = scale / max(h, w)
            
            new_h = int(h * scale_factor)
            new_w = int(w * scale_factor)
            
            resized_img = F.interpolate(
                img.unsqueeze(0), 
                size=(new_h, new_w), 
                mode='bilinear'
            ).squeeze(0)
            
            # Scale boxes and masks
            target['boxes'] *= scale_factor
            target['masks'] = F.interpolate(
                target['masks'].float().unsqueeze(0),
                size=(new_h, new_w),
                mode='nearest'
            ).squeeze(0).byte()
            
            resized_images.append(resized_img)
            resized_targets.append(target)
        
        return self.model(resized_images, resized_targets)
```

## 6. Evaluation Metrics & Validation Strategy

### Core Metrics
```python
from pycocotools.cocoeval import COCOeval

def evaluate_instance_segmentation(model, data_loader, device):
    model.eval()
    coco_evaluator = COCOeval(coco_gt, coco_dt, 'segm')
    
    metrics = {}
    
    with torch.no_grad():
        for images, targets in data_loader:
            images = [img.to(device) for img in images]
            outputs = model(images)
            
            # Convert to COCO format
            for output, target in zip(outputs, targets):
                coco_dt.append({
                    'image_id': target['image_id'].item(),
                    'category_id': output['labels'].cpu().numpy(),
                    'segmentation': maskUtils.encode(
                        np.asfortranarray(output['masks'].cpu().numpy())
                    ),
                    'score': output['scores'].cpu().numpy()
                })
    
    coco_evaluator.evaluate()
    coco_evaluator.accumulate()
    coco_evaluator.summarize()
    
    return {
        'mask_mAP': coco_evaluator.stats[0],
        'mask_mAP_50': coco_evaluator.stats[1],
        'mask_mAP_75': coco_evaluator.stats[2],
        'mask_mAP_small': coco_evaluator.stats[3],
        'mask_mAP_medium': coco_evaluator.stats[4],
        'mask_mAP_large': coco_evaluator.stats[5]
    }
```

### IoU Metrics
```python
def calculate_mask_iou(pred_masks, gt_masks):
    """Calculate IoU for instance masks"""
    ious = []
    
    for pred_mask, gt_mask in zip(pred_masks, gt_masks):
        intersection = (pred_mask & gt_mask).sum()
        union = (pred_mask | gt_mask).sum()
        iou = intersection / (union + 1e-6)
        ious.append(iou)
    
    return np.mean(ious)

def pixel_accuracy(pred_masks, gt_masks):
    """Calculate pixel-level accuracy"""
    correct = (pred_masks == gt_masks).sum()
    total = gt_masks.numel()
    return correct / total
```

### Panoptic Quality (PQ)
```python
def panoptic_quality(pred_segments, gt_segments):
    """Calculate PQ metric for panoptic segmentation"""
    tp, fp, fn = 0, 0, 0
    sum_iou = 0
    
    matched_gt = set()
    
    for pred_id in np.unique(pred_segments):
        if pred_id == 0:  # Skip background
            continue
        
        pred_mask = pred_segments == pred_id
        best_iou = 0
        best_gt_id = None
        
        for gt_id in np.unique(gt_segments):
            if gt_id == 0 or gt_id in matched_gt:
                continue
            
            gt_mask = gt_segments == gt_id
            iou = calculate_iou(pred_mask, gt_mask)
            
            if iou > best_iou:
                best_iou = iou
                best_gt_id = gt_id
        
        if best_iou > 0.5:
            tp += 1
            sum_iou += best_iou
            matched_gt.add(best_gt_id)
        else:
            fp += 1
    
    fn = len(np.unique(gt_segments)) - len(matched_gt) - 1  # -1 for background
    
    sq = sum_iou / (tp + 1e-6)  # Segmentation Quality
    rq = tp / (tp + 0.5 * fp + 0.5 * fn + 1e-6)  # Recognition Quality
    pq = sq * rq  # Panoptic Quality
    
    return {'PQ': pq, 'SQ': sq, 'RQ': rq}
```

## 7. Experiment Tracking & Reproducibility

### MLflow + DVC Pipeline
```python
import mlflow
import dvc.api

# Track experiments
with mlflow.start_run():
    # Log hyperparameters
    mlflow.log_params(config)
    
    # Version data with DVC
    data_url = dvc.api.get_url('data/coco.dvc')
    mlflow.log_param('data_version', data_url)
    
    # Training loop
    for epoch in range(num_epochs):
        train_loss = train_epoch(model, train_loader)
        val_metrics = evaluate(model, val_loader)
        
        mlflow.log_metrics({
            'train_loss': train_loss,
            'val_mask_mAP': val_metrics['mask_mAP'],
            'val_mask_mAP_50': val_metrics['mask_mAP_50']
        }, step=epoch)
        
        # Log sample predictions
        if epoch % 10 == 0:
            fig = visualize_predictions(model, val_loader)
            mlflow.log_figure(fig, f'predictions_epoch_{epoch}.png')
    
    # Save model
    mlflow.pytorch.log_model(model, 'model')
```

### TensorBoard Integration
```python
from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter('runs/instance_segmentation')

def log_segmentation_results(writer, images, outputs, targets, step):
    for idx in range(min(4, len(images))):  # Log 4 samples
        img = images[idx]
        output = outputs[idx]
        target = targets[idx]
        
        # Draw predicted masks
        pred_mask = draw_segmentation_masks(
            img, 
            output['masks'] > 0.5,
            colors=['red', 'blue', 'green', 'yellow'] * 20
        )
        
        # Draw ground truth masks
        gt_mask = draw_segmentation_masks(
            img,
            target['masks'],
            colors=['red', 'blue', 'green', 'yellow'] * 20
        )
        
        writer.add_image(f'predictions/{idx}', pred_mask, step)
        writer.add_image(f'ground_truth/{idx}', gt_mask, step)
```

## 8. Deployment Pathway

### Option 1: Real-time Web Service
```python
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
import torch
import cv2
import base64

app = FastAPI()

model = load_model('best_model.pth')
model.eval()

@app.post("/segment")
async def segment_image(file: UploadFile = File(...)):
    # Read and preprocess image
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    # Run inference
    with torch.no_grad():
        tensor = transform(image).unsqueeze(0)
        outputs = model(tensor)[0]
    
    # Process outputs
    masks = outputs['masks'] > 0.5
    labels = outputs['labels'].cpu().numpy()
    scores = outputs['scores'].cpu().numpy()
    
    # Encode masks as RLE
    encoded_masks = []
    for mask in masks:
        rle = maskUtils.encode(
            np.asfortranarray(mask.cpu().numpy().astype(np.uint8))
        )
        rle['counts'] = rle['counts'].decode('utf-8')
        encoded_masks.append(rle)
    
    return JSONResponse({
        'masks': encoded_masks,
        'labels': labels.tolist(),
        'scores': scores.tolist()
    })

@app.post("/segment_interactive")
async def interactive_segment(
    file: UploadFile = File(...),
    points: List[List[int]] = [],
    labels: List[int] = []
):
    """Interactive segmentation with user clicks"""
    # SAM-based interactive segmentation
    image = load_image(await file.read())
    
    predictor.set_image(image)
    masks, scores, _ = predictor.predict(
        point_coords=np.array(points),
        point_labels=np.array(labels),
        multimask_output=True
    )
    
    return {
        'masks': encode_masks(masks),
        'scores': scores.tolist()
    }
```

### Option 2: Gradio Interactive Demo
```python
import gradio as gr
from segment_anything import SamPredictor

def segment_with_points(image, points):
    predictor.set_image(image)
    
    if len(points) > 0:
        point_coords = np.array([[p[0], p[1]] for p in points])
        point_labels = np.array([1] * len(points))  # All foreground
        
        masks, scores, _ = predictor.predict(
            point_coords=point_coords,
            point_labels=point_labels,
            multimask_output=False
        )
        
        # Overlay mask on image
        mask = masks[0]
        colored_mask = np.zeros_like(image)
        colored_mask[mask] = [0, 255, 0]  # Green mask
        
        result = cv2.addWeighted(image, 0.7, colored_mask, 0.3, 0)
        return result
    
    return image

demo = gr.Interface(
    fn=segment_with_points,
    inputs=[
        gr.Image(label="Upload Image"),
        gr.Point(label="Click on objects to segment")
    ],
    outputs=gr.Image(label="Segmentation Result"),
    title="Interactive Instance Segmentation",
    examples=[
        ["examples/street.jpg"],
        ["examples/office.jpg"]
    ]
)

demo.launch()
```

### Option 3: Mobile Deployment with ONNX
```python
# Export to ONNX
import torch.onnx

dummy_input = torch.randn(1, 3, 512, 512)
torch.onnx.export(
    model,
    dummy_input,
    "instance_seg.onnx",
    opset_version=11,
    input_names=['input'],
    output_names=['masks', 'labels', 'scores', 'boxes'],
    dynamic_axes={
        'input': {0: 'batch'},
        'masks': {0: 'num_detections'},
        'labels': {0: 'num_detections'},
        'scores': {0: 'num_detections'},
        'boxes': {0: 'num_detections'}
    }
)

# Mobile inference with ONNX Runtime
import onnxruntime as ort

class MobileSegmentation:
    def __init__(self, model_path):
        self.session = ort.InferenceSession(
            model_path,
            providers=['CoreMLExecutionProvider']  # For iOS
        )
    
    def segment(self, image):
        input_tensor = preprocess(image)
        outputs = self.session.run(None, {'input': input_tensor})
        return postprocess(outputs)
```

## 9. Extensions & Research Directions

### Advanced Techniques

1. **3D Instance Segmentation**
```python
class PointNet3DInstanceSeg(nn.Module):
    """3D instance segmentation for point clouds"""
    def __init__(self, num_classes):
        super().__init__()
        self.backbone = PointNetBackbone()
        self.instance_head = nn.Sequential(
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, num_classes + 1)  # +1 for background
        )
```

2. **Video Instance Segmentation**
```python
class VideoInstanceSeg(nn.Module):
    """Temporal consistency for video segmentation"""
    def __init__(self):
        super().__init__()
        self.frame_encoder = ResNet50()
        self.temporal_module = nn.LSTM(2048, 1024, bidirectional=True)
        self.mask_decoder = MaskDecoder()
    
    def forward(self, video_frames):
        features = [self.frame_encoder(f) for f in video_frames]
        temporal_features, _ = self.temporal_module(torch.stack(features))
        masks = [self.mask_decoder(f) for f in temporal_features]
        return masks
```

3. **Weakly Supervised Segmentation**
```python
def train_with_weak_labels(model, images, bounding_boxes):
    """Train with only bounding box annotations"""
    # Generate pseudo masks using GrabCut
    pseudo_masks = []
    for img, box in zip(images, bounding_boxes):
        mask = np.zeros(img.shape[:2], np.uint8)
        rect = (box[0], box[1], box[2]-box[0], box[3]-box[1])
        cv2.grabCut(img, mask, rect, None, None, 5, cv2.GC_INIT_WITH_RECT)
        pseudo_masks.append(mask == cv2.GC_FGD)
    
    # Train with pseudo labels
    return train_step(model, images, pseudo_masks)
```

### Novel Experiments
- **Few-shot instance segmentation** with meta-learning
- **Open-vocabulary segmentation** (segment anything)
- **Compositional reasoning** (part-whole relationships)
- **Active learning** for efficient annotation
- **Domain adaptation** for synthetic-to-real transfer

### Industry Applications
- **Medical Imaging**: Tumor/organ instance segmentation
- **Agriculture**: Individual plant/fruit detection
- **Manufacturing**: Defect segmentation in products
- **Retail**: Shelf analysis and inventory tracking
- **Geospatial**: Building/vehicle segmentation in satellite imagery

## 10. Portfolio Polish

### Project Structure
```
instance_segmentation/
├── README.md
├── configs/
│   ├── mask_rcnn.yaml
│   ├── yolact.yaml
│   └── sam.yaml
├── data/
│   ├── prepare_coco.py
│   └── prepare_cityscapes.py
├── models/
│   ├── mask_rcnn.py
│   ├── yolact.py
│   └── sam_adapter.py
├── utils/
│   ├── visualization.py
│   ├── coco_eval.py
│   └── augmentations.py
├── train.py
├── evaluate.py
├── inference.py
├── interactive_demo.py
├── notebooks/
│   ├── data_analysis.ipynb
│   ├── model_comparison.ipynb
│   └── error_analysis.ipynb
└── deployment/
    ├── api.py
    ├── gradio_app.py
    └── export_onnx.py
```

### Visualization Suite
```python
import matplotlib.pyplot as plt
from matplotlib import patches, patheffects

def visualize_instance_segmentation(image, masks, boxes, labels, scores, 
                                   class_names, save_path=None):
    """Comprehensive instance segmentation visualization"""
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # Original image
    axes[0].imshow(image)
    axes[0].set_title('Original Image')
    axes[0].axis('off')
    
    # Instance masks with different colors
    ax = axes[1]
    ax.imshow(image)
    
    # Generate distinct colors for each instance
    colors = plt.cm.rainbow(np.linspace(0, 1, len(masks)))
    
    for mask, color, label, score in zip(masks, colors, labels, scores):
        masked_image = np.zeros_like(image)
        masked_image[mask > 0.5] = (np.array(color[:3]) * 255).astype(np.uint8)
        ax.imshow(masked_image, alpha=0.5)
        
        # Find contour for better visualization
        contours = cv2.findContours(
            (mask > 0.5).astype(np.uint8),
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )[0]
        
        for contour in contours:
            ax.plot(contour[:, 0, 0], contour[:, 0, 1], 
                   color=color, linewidth=2)
    
    ax.set_title('Instance Masks')
    ax.axis('off')
    
    # Semantic segmentation (merged instances)
    ax = axes[2]
    semantic_mask = np.zeros((*image.shape[:2], 3))
    
    for mask, label in zip(masks, labels):
        color = plt.cm.tab20(label % 20)[:3]
        semantic_mask[mask > 0.5] = color
    
    ax.imshow(semantic_mask)
    ax.set_title('Semantic Segmentation')
    ax.axis('off')
    
    # Add legend
    handles = []
    for class_id in np.unique(labels):
        class_name = class_names[class_id]
        color = plt.cm.tab20(class_id % 20)
        handles.append(patches.Patch(color=color, label=class_name))
    
    fig.legend(handles=handles, loc='lower center', ncol=5)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    return fig
```

### Performance Dashboard
```python
def create_segmentation_dashboard(results):
    """Create comprehensive performance analysis dashboard"""
    fig = plt.figure(figsize=(20, 12))
    
    # mAP breakdown
    ax1 = plt.subplot(3, 4, 1)
    metrics = ['mAP', 'mAP@50', 'mAP@75', 'mAP_S', 'mAP_M', 'mAP_L']
    values = [results[m] for m in metrics]
    ax1.bar(metrics, values)
    ax1.set_title('mAP Breakdown')
    ax1.set_ylim([0, 1])
    
    # Per-class performance
    ax2 = plt.subplot(3, 4, 2)
    ax2.barh(results['class_names'], results['per_class_ap'])
    ax2.set_title('Per-Class AP')
    ax2.set_xlabel('Average Precision')
    
    # IoU distribution
    ax3 = plt.subplot(3, 4, 3)
    ax3.hist(results['iou_scores'], bins=50, edgecolor='black')
    ax3.set_title('IoU Distribution')
    ax3.set_xlabel('IoU')
    ax3.set_ylabel('Count')
    
    # Inference time analysis
    ax4 = plt.subplot(3, 4, 4)
    components = ['Backbone', 'RPN', 'RoI', 'Mask Head', 'NMS']
    times = results['component_times']
    ax4.pie(times, labels=components, autopct='%1.1f%%')
    ax4.set_title('Inference Time Breakdown')
    
    # Precision-Recall curves
    ax5 = plt.subplot(3, 4, 5)
    for class_name, pr_curve in results['pr_curves'].items():
        ax5.plot(pr_curve['recall'], pr_curve['precision'], 
                label=class_name)
    ax5.set_title('Precision-Recall Curves')
    ax5.set_xlabel('Recall')
    ax5.set_ylabel('Precision')
    ax5.legend()
    
    # Mask quality analysis
    ax6 = plt.subplot(3, 4, 6)
    ax6.scatter(results['mask_sizes'], results['mask_ious'])
    ax6.set_title('Mask Quality vs Size')
    ax6.set_xlabel('Mask Size (pixels)')
    ax6.set_ylabel('Mask IoU')
    
    # Training curves
    ax7 = plt.subplot(3, 4, 7)
    ax7.plot(results['train_loss'], label='Train Loss')
    ax7.plot(results['val_loss'], label='Val Loss')
    ax7.set_title('Training Progress')
    ax7.set_xlabel('Epoch')
    ax7.set_ylabel('Loss')
    ax7.legend()
    
    # Confusion matrix for classifications
    ax8 = plt.subplot(3, 4, 8)
    im = ax8.imshow(results['confusion_matrix'], cmap='Blues')
    ax8.set_title('Classification Confusion Matrix')
    plt.colorbar(im, ax=ax8)
    
    # Sample predictions grid
    gs = plt.subplot(3, 4, (9, 12))
    sample_grid = create_prediction_grid(results['sample_predictions'])
    gs.imshow(sample_grid)
    gs.set_title('Sample Predictions')
    gs.axis('off')
    
    plt.tight_layout()
    return fig
```

### README Template
```markdown
# Instance Segmentation: Pixel-Perfect Object Detection

[![Demo](https://img.shields.io/badge/demo-live-brightgreen)](https://your-demo.com)
[![Model Zoo](https://img.shields.io/badge/models-download-blue)](https://your-models.com)

## Features
- 🎯 Pixel-level precision for each object instance
- ⚡ Real-time inference (25+ FPS)
- 🔄 Interactive segmentation with point prompts
- 📊 State-of-the-art accuracy on COCO (55% mAP)

## Quick Start
```bash
# Install dependencies
pip install -r requirements.txt

# Download pretrained models
python download_models.py

# Run inference on image
python inference.py --image path/to/image.jpg --model mask_rcnn

# Start interactive demo
python interactive_demo.py
```

## Model Performance

| Model | Backbone | mAP | mAP@50 | mAP@75 | FPS | Params |
|-------|----------|-----|--------|--------|-----|--------|
| Mask R-CNN | ResNet-101 | 38.2 | 58.8 | 41.3 | 11 | 63.2M |
| YOLACT++ | ResNet-101 | 34.1 | 53.3 | 36.8 | 33 | 36.6M |
| SOLOv2 | ResNet-101 | 39.7 | 59.9 | 42.9 | 18 | 46.7M |
| SAM | ViT-H | 46.5 | 65.2 | 50.1 | 8 | 636M |

## Visualization Gallery
![Instance Segmentation Results](assets/results_grid.png)
```