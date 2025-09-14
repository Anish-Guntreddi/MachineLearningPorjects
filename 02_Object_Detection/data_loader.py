"""
Data loading for object detection (COCO/VOC datasets)
"""
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision
from torchvision import transforms
import numpy as np
from pycocotools.coco import COCO
import os
import cv2
from PIL import Image
from typing import Tuple, List, Dict, Optional
import albumentations as A
from albumentations.pytorch import ToTensorV2


class COCODetection(Dataset):
    """COCO dataset for object detection"""
    
    def __init__(
        self,
        root_dir: str,
        annotation_file: str,
        transforms: Optional = None,
        train: bool = True
    ):
        """
        Args:
            root_dir: Root directory of images
            annotation_file: Path to annotation file
            transforms: Albumentations transforms
            train: Whether this is training set
        """
        self.root_dir = root_dir
        self.transforms = transforms
        self.train = train
        
        # Load COCO annotations
        self.coco = COCO(annotation_file)
        self.image_ids = list(self.coco.imgs.keys())
        
        # Filter images with annotations
        self.image_ids = [img_id for img_id in self.image_ids 
                          if len(self.coco.getAnnIds(imgIds=img_id)) > 0]
        
        # Get category mapping
        self.cat_ids = self.coco.getCatIds()
        self.cat2label = {cat_id: i + 1 for i, cat_id in enumerate(self.cat_ids)}
        self.label2cat = {v: k for k, v in self.cat2label.items()}
        
    def __len__(self):
        return len(self.image_ids)
    
    def __getitem__(self, idx):
        img_id = self.image_ids[idx]
        
        # Load image
        img_info = self.coco.loadImgs(img_id)[0]
        img_path = os.path.join(self.root_dir, img_info['file_name'])
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Load annotations
        ann_ids = self.coco.getAnnIds(imgIds=img_id)
        anns = self.coco.loadAnns(ann_ids)
        
        # Extract bboxes and labels
        boxes = []
        labels = []
        areas = []
        iscrowd = []
        
        for ann in anns:
            # Skip crowd annotations
            if ann.get('iscrowd', 0) == 1 and self.train:
                continue
            
            x, y, w, h = ann['bbox']
            # Convert to [x1, y1, x2, y2]
            boxes.append([x, y, x + w, y + h])
            labels.append(self.cat2label[ann['category_id']])
            areas.append(ann['area'])
            iscrowd.append(ann.get('iscrowd', 0))
        
        # Convert to tensors
        boxes = np.array(boxes, dtype=np.float32)
        labels = np.array(labels, dtype=np.int64)
        areas = np.array(areas, dtype=np.float32)
        iscrowd = np.array(iscrowd, dtype=np.int64)
        
        # Apply transforms
        if self.transforms:
            transformed = self.transforms(
                image=img,
                bboxes=boxes,
                labels=labels
            )
            img = transformed['image']
            boxes = np.array(transformed['bboxes'], dtype=np.float32)
            labels = np.array(transformed['labels'], dtype=np.int64)
        
        # Convert to tensors
        if len(boxes) == 0:
            boxes = torch.zeros((0, 4), dtype=torch.float32)
            labels = torch.zeros((0,), dtype=torch.int64)
            areas = torch.zeros((0,), dtype=torch.float32)
            iscrowd = torch.zeros((0,), dtype=torch.int64)
        else:
            boxes = torch.as_tensor(boxes, dtype=torch.float32)
            labels = torch.as_tensor(labels, dtype=torch.int64)
            areas = torch.as_tensor(areas, dtype=torch.float32)
            iscrowd = torch.as_tensor(iscrowd, dtype=torch.int64)
        
        target = {
            'boxes': boxes,
            'labels': labels,
            'image_id': torch.tensor([img_id]),
            'area': areas,
            'iscrowd': iscrowd
        }
        
        img = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
        
        return img, target


class VOCDetection(Dataset):
    """PASCAL VOC dataset for object detection"""
    
    def __init__(
        self,
        root_dir: str,
        year: str = '2012',
        image_set: str = 'train',
        transforms: Optional = None
    ):
        """
        Args:
            root_dir: Root directory of VOC dataset
            year: Dataset year
            image_set: 'train', 'val', or 'trainval'
            transforms: Albumentations transforms
        """
        self.root_dir = root_dir
        self.year = year
        self.image_set = image_set
        self.transforms = transforms
        
        # VOC classes
        self.classes = [
            'background', 'aeroplane', 'bicycle', 'bird', 'boat',
            'bottle', 'bus', 'car', 'cat', 'chair', 'cow',
            'diningtable', 'dog', 'horse', 'motorbike', 'person',
            'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor'
        ]
        self.class_to_idx = {cls: i for i, cls in enumerate(self.classes)}
        
        # Load dataset
        self.dataset = torchvision.datasets.VOCDetection(
            root=root_dir,
            year=year,
            image_set=image_set,
            download=True
        )
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        img, target = self.dataset[idx]
        
        # Convert PIL to numpy
        img = np.array(img)
        
        # Parse annotations
        boxes = []
        labels = []
        
        objects = target['annotation']['object']
        if not isinstance(objects, list):
            objects = [objects]
        
        for obj in objects:
            bbox = obj['bndbox']
            x1 = float(bbox['xmin'])
            y1 = float(bbox['ymin'])
            x2 = float(bbox['xmax'])
            y2 = float(bbox['ymax'])
            
            boxes.append([x1, y1, x2, y2])
            labels.append(self.class_to_idx[obj['name']])
        
        boxes = np.array(boxes, dtype=np.float32)
        labels = np.array(labels, dtype=np.int64)
        
        # Apply transforms
        if self.transforms:
            transformed = self.transforms(
                image=img,
                bboxes=boxes,
                labels=labels
            )
            img = transformed['image']
            boxes = np.array(transformed['bboxes'], dtype=np.float32)
            labels = np.array(transformed['labels'], dtype=np.int64)
        
        # Convert to tensors
        img = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
        
        target = {
            'boxes': torch.as_tensor(boxes, dtype=torch.float32),
            'labels': torch.as_tensor(labels, dtype=torch.int64),
            'image_id': torch.tensor([idx])
        }
        
        return img, target


def get_detection_transforms(train: bool = True, img_size: int = 640):
    """Get transforms for object detection"""
    
    if train:
        transform = A.Compose([
            A.RandomResizedCrop(img_size, img_size, scale=(0.5, 1.0)),
            A.HorizontalFlip(p=0.5),
            A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            A.OneOf([
                A.MotionBlur(p=0.2),
                A.MedianBlur(blur_limit=3, p=0.1),
                A.GaussianBlur(p=0.2),
            ], p=0.3),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2()
        ], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['labels']))
    else:
        transform = A.Compose([
            A.Resize(img_size, img_size),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2()
        ], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['labels']))
    
    return transform


def collate_fn(batch):
    """Custom collate function for object detection"""
    images = []
    targets = []
    
    for img, target in batch:
        images.append(img)
        targets.append(target)
    
    return images, targets


def get_data_loaders(
    dataset_name: str = 'voc',
    data_dir: str = './data',
    batch_size: int = 8,
    num_workers: int = 4,
    img_size: int = 640
) -> Tuple[DataLoader, DataLoader]:
    """
    Get data loaders for object detection
    
    Args:
        dataset_name: 'voc' or 'coco'
        data_dir: Root data directory
        batch_size: Batch size
        num_workers: Number of workers
        img_size: Image size for training
    
    Returns:
        train_loader, val_loader
    """
    
    train_transform = get_detection_transforms(train=True, img_size=img_size)
    val_transform = get_detection_transforms(train=False, img_size=img_size)
    
    if dataset_name == 'voc':
        train_dataset = VOCDetection(
            root_dir=data_dir,
            year='2012',
            image_set='train',
            transforms=train_transform
        )
        
        val_dataset = VOCDetection(
            root_dir=data_dir,
            year='2012',
            image_set='val',
            transforms=val_transform
        )
    else:
        # For COCO, you need to download and set up the dataset first
        raise NotImplementedError("COCO dataset loading not fully implemented. Please download COCO first.")
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True
    )
    
    return train_loader, val_loader


class MosaicDataset(Dataset):
    """Dataset wrapper for Mosaic augmentation (YOLO-style)"""
    
    def __init__(self, dataset, img_size=640):
        self.dataset = dataset
        self.img_size = img_size
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        # Get 4 images for mosaic
        indices = [idx] + np.random.choice(len(self.dataset), 3, replace=False).tolist()
        
        mosaic_img = np.zeros((self.img_size, self.img_size, 3), dtype=np.float32)
        mosaic_boxes = []
        mosaic_labels = []
        
        for i, idx in enumerate(indices):
            img, target = self.dataset[idx]
            h, w = img.shape[1:]
            
            # Place in quadrant
            if i == 0:  # Top-left
                x1, y1, x2, y2 = 0, 0, self.img_size // 2, self.img_size // 2
            elif i == 1:  # Top-right
                x1, y1, x2, y2 = self.img_size // 2, 0, self.img_size, self.img_size // 2
            elif i == 2:  # Bottom-left
                x1, y1, x2, y2 = 0, self.img_size // 2, self.img_size // 2, self.img_size
            else:  # Bottom-right
                x1, y1, x2, y2 = self.img_size // 2, self.img_size // 2, self.img_size, self.img_size
            
            # Resize and place image
            img_resized = cv2.resize(img.permute(1, 2, 0).numpy(), (x2 - x1, y2 - y1))
            mosaic_img[y1:y2, x1:x2] = img_resized
            
            # Adjust boxes
            boxes = target['boxes'].numpy()
            if len(boxes) > 0:
                boxes[:, [0, 2]] = boxes[:, [0, 2]] * (x2 - x1) / w + x1
                boxes[:, [1, 3]] = boxes[:, [1, 3]] * (y2 - y1) / h + y1
                mosaic_boxes.extend(boxes)
                mosaic_labels.extend(target['labels'].numpy())
        
        mosaic_img = torch.from_numpy(mosaic_img).permute(2, 0, 1)
        
        target = {
            'boxes': torch.tensor(mosaic_boxes, dtype=torch.float32),
            'labels': torch.tensor(mosaic_labels, dtype=torch.int64)
        }
        
        return mosaic_img, target


if __name__ == "__main__":
    # Test data loading
    train_loader, val_loader = get_data_loaders(
        dataset_name='voc',
        batch_size=2,
        img_size=416
    )
    
    print(f"Train batches: {len(train_loader)}")
    print(f"Val batches: {len(val_loader)}")
    
    # Test one batch
    images, targets = next(iter(train_loader))
    print(f"Batch size: {len(images)}")
    print(f"Image shape: {images[0].shape}")
    print(f"Targets: {targets[0]}")