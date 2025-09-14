"""
Data loading and preprocessing for image classification
"""
import torch
from torch.utils.data import DataLoader, Dataset
import torchvision
from torchvision import datasets, transforms
import numpy as np
from typing import Tuple, Optional
import os


def get_data_transforms(train: bool = True, img_size: int = 32) -> transforms.Compose:
    """
    Get data transforms for training or validation
    
    Args:
        train: Whether to apply training augmentations
        img_size: Size of the images
    
    Returns:
        Composed transforms
    """
    # CIFAR-10 statistics
    cifar_mean = (0.4914, 0.4822, 0.4465)
    cifar_std = (0.2023, 0.1994, 0.2010)
    
    if train:
        transform = transforms.Compose([
            transforms.RandomCrop(img_size, padding=4),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            transforms.RandomRotation(degrees=15),
            transforms.ToTensor(),
            transforms.Normalize(mean=cifar_mean, std=cifar_std)
        ])
    else:
        transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=cifar_mean, std=cifar_std)
        ])
    
    return transform


def load_cifar10(
    data_dir: str = './data',
    batch_size: int = 128,
    num_workers: int = 4,
    pin_memory: bool = True
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Load CIFAR-10 dataset
    
    Args:
        data_dir: Directory to save/load data
        batch_size: Batch size for training
        num_workers: Number of workers for data loading
        pin_memory: Pin memory for faster GPU transfer
    
    Returns:
        train_loader, val_loader, test_loader
    """
    # Create data directory if it doesn't exist
    os.makedirs(data_dir, exist_ok=True)
    
    # Get transforms
    train_transform = get_data_transforms(train=True)
    test_transform = get_data_transforms(train=False)
    
    # Load datasets
    train_dataset = datasets.CIFAR10(
        root=data_dir,
        train=True,
        download=True,
        transform=train_transform
    )
    
    # Split train into train and validation
    train_size = int(0.9 * len(train_dataset))
    val_size = len(train_dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        train_dataset, [train_size, val_size]
    )
    
    # Update validation dataset transform
    val_dataset.dataset.transform = test_transform
    
    test_dataset = datasets.CIFAR10(
        root=data_dir,
        train=False,
        download=True,
        transform=test_transform
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=True if num_workers > 0 else False
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size * 2,  # Can use larger batch for validation
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=True if num_workers > 0 else False
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size * 2,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory
    )
    
    return train_loader, val_loader, test_loader


class MixupDataset(Dataset):
    """Dataset wrapper for Mixup augmentation"""
    
    def __init__(self, dataset, alpha=0.2):
        self.dataset = dataset
        self.alpha = alpha
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        img1, label1 = self.dataset[idx]
        
        # Get random second sample
        idx2 = np.random.randint(0, len(self.dataset))
        img2, label2 = self.dataset[idx2]
        
        # Mixup
        lam = np.random.beta(self.alpha, self.alpha)
        img = lam * img1 + (1 - lam) * img2
        
        return img, label1, label2, lam


class CutMixDataset(Dataset):
    """Dataset wrapper for CutMix augmentation"""
    
    def __init__(self, dataset, alpha=1.0):
        self.dataset = dataset
        self.alpha = alpha
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        img1, label1 = self.dataset[idx]
        
        # Get random second sample
        idx2 = np.random.randint(0, len(self.dataset))
        img2, label2 = self.dataset[idx2]
        
        # CutMix
        lam = np.random.beta(self.alpha, self.alpha)
        
        # Get random box
        H, W = img1.shape[1:]
        cut_rat = np.sqrt(1. - lam)
        cut_w = int(W * cut_rat)
        cut_h = int(H * cut_rat)
        
        # Uniform sampling
        cx = np.random.randint(W)
        cy = np.random.randint(H)
        
        bbx1 = np.clip(cx - cut_w // 2, 0, W)
        bby1 = np.clip(cy - cut_h // 2, 0, H)
        bbx2 = np.clip(cx + cut_w // 2, 0, W)
        bby2 = np.clip(cy + cut_h // 2, 0, H)
        
        # Apply CutMix
        img = img1.clone()
        img[:, bby1:bby2, bbx1:bbx2] = img2[:, bby1:bby2, bbx1:bbx2]
        
        # Adjust lambda
        lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (W * H))
        
        return img, label1, label2, lam


if __name__ == "__main__":
    # Test data loading
    train_loader, val_loader, test_loader = load_cifar10(batch_size=32)
    
    print(f"Train batches: {len(train_loader)}")
    print(f"Val batches: {len(val_loader)}")
    print(f"Test batches: {len(test_loader)}")
    
    # Test one batch
    images, labels = next(iter(train_loader))
    print(f"Batch shape: {images.shape}")
    print(f"Labels shape: {labels.shape}")