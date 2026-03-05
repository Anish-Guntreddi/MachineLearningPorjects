"""
Data loading for instance segmentation
"""
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision
from torchvision import transforms as T
import numpy as np
import os
from typing import Dict, List, Tuple, Optional
from PIL import Image


class InstanceSegmentationDataset(Dataset):
    """Dataset for instance segmentation with COCO-format annotations"""

    def __init__(
        self,
        images: List[torch.Tensor],
        targets: List[Dict],
        transforms=None
    ):
        self.images = images
        self.targets = targets
        self.transforms = transforms

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        target = self.targets[idx]

        if self.transforms:
            image = self.transforms(image)

        return image, target


def get_transforms(train: bool = True):
    """Get transforms for instance segmentation"""
    transform_list = []

    if train:
        transform_list.append(T.RandomHorizontalFlip(0.5))

    transform_list.append(T.ConvertImageDtype(torch.float))

    return T.Compose(transform_list)


def create_synthetic_data(
    num_samples: int = 100,
    num_classes: int = 21,
    img_height: int = 300,
    img_width: int = 400,
    max_objects: int = 5
) -> Tuple[List[torch.Tensor], List[Dict]]:
    """
    Create synthetic data for instance segmentation demonstration

    Args:
        num_samples: Number of samples to generate
        num_classes: Number of object classes (including background)
        img_height: Image height
        img_width: Image width
        max_objects: Maximum number of objects per image

    Returns:
        images: List of image tensors
        targets: List of target dicts with boxes, labels, masks
    """
    images = []
    targets = []

    for i in range(num_samples):
        # Create random image
        img = torch.rand(3, img_height, img_width)

        # Random number of objects
        num_objects = np.random.randint(1, max_objects + 1)

        boxes = []
        labels = []
        masks = []
        areas = []

        for _ in range(num_objects):
            # Random bounding box
            x1 = np.random.randint(0, img_width - 50)
            y1 = np.random.randint(0, img_height - 50)
            x2 = np.random.randint(x1 + 20, min(x1 + 150, img_width))
            y2 = np.random.randint(y1 + 20, min(y1 + 150, img_height))

            boxes.append([x1, y1, x2, y2])
            labels.append(np.random.randint(1, num_classes))

            # Create mask for this object
            mask = torch.zeros(img_height, img_width, dtype=torch.uint8)
            mask[y1:y2, x1:x2] = 1
            masks.append(mask)

            areas.append((x2 - x1) * (y2 - y1))

        target = {
            'boxes': torch.tensor(boxes, dtype=torch.float32),
            'labels': torch.tensor(labels, dtype=torch.int64),
            'masks': torch.stack(masks),
            'image_id': torch.tensor([i]),
            'area': torch.tensor(areas, dtype=torch.float32),
            'iscrowd': torch.zeros(num_objects, dtype=torch.int64)
        }

        images.append(img)
        targets.append(target)

    return images, targets


def collate_fn(batch):
    """Custom collate function for detection models"""
    return tuple(zip(*batch))


def create_data_loaders(
    num_train: int = 100,
    num_val: int = 20,
    num_test: int = 20,
    num_classes: int = 21,
    batch_size: int = 4,
    num_workers: int = 4,
    pin_memory: bool = True
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create data loaders for instance segmentation

    Args:
        num_train: Number of training samples
        num_val: Number of validation samples
        num_test: Number of test samples
        num_classes: Number of classes
        batch_size: Batch size
        num_workers: Number of data loading workers
        pin_memory: Whether to pin memory

    Returns:
        train_loader, val_loader, test_loader
    """
    # Create synthetic data
    train_images, train_targets = create_synthetic_data(num_train, num_classes)
    val_images, val_targets = create_synthetic_data(num_val, num_classes)
    test_images, test_targets = create_synthetic_data(num_test, num_classes)

    # Create datasets
    train_dataset = InstanceSegmentationDataset(
        train_images, train_targets, transforms=get_transforms(train=True)
    )
    val_dataset = InstanceSegmentationDataset(
        val_images, val_targets, transforms=get_transforms(train=False)
    )
    test_dataset = InstanceSegmentationDataset(
        test_images, test_targets, transforms=get_transforms(train=False)
    )

    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=collate_fn
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=collate_fn
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=collate_fn
    )

    return train_loader, val_loader, test_loader


if __name__ == "__main__":
    # Test data loading
    train_loader, val_loader, test_loader = create_data_loaders(
        num_train=20, num_val=5, num_test=5, batch_size=2, num_workers=0
    )

    images, targets = next(iter(train_loader))
    print(f"Batch size: {len(images)}")
    print(f"Image shape: {images[0].shape}")
    print(f"Target keys: {targets[0].keys()}")
    print(f"Boxes shape: {targets[0]['boxes'].shape}")
    print(f"Masks shape: {targets[0]['masks'].shape}")
