"""
Model architectures for image classification
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
import timm
from typing import Optional


class SimpleCNN(nn.Module):
    """Simple CNN baseline model"""
    
    def __init__(self, num_classes: int = 10):
        super(SimpleCNN, self).__init__()
        
        self.features = nn.Sequential(
            # Block 1
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout2d(0.25),
            
            # Block 2
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout2d(0.25),
            
            # Block 3
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1)
        )
        
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(128, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )
    
    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


class ResNetModel(nn.Module):
    """ResNet-based model with customizable architecture"""
    
    def __init__(
        self,
        model_name: str = 'resnet18',
        num_classes: int = 10,
        pretrained: bool = True,
        freeze_backbone: bool = False
    ):
        super(ResNetModel, self).__init__()
        
        # Load pretrained model
        if model_name == 'resnet18':
            self.model = models.resnet18(pretrained=pretrained)
            in_features = 512
        elif model_name == 'resnet34':
            self.model = models.resnet34(pretrained=pretrained)
            in_features = 512
        elif model_name == 'resnet50':
            self.model = models.resnet50(pretrained=pretrained)
            in_features = 2048
        elif model_name == 'resnet101':
            self.model = models.resnet101(pretrained=pretrained)
            in_features = 2048
        else:
            raise ValueError(f"Unknown model: {model_name}")
        
        # Freeze backbone if specified
        if freeze_backbone:
            for param in self.model.parameters():
                param.requires_grad = False
        
        # Replace final layer
        self.model.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(in_features, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )
    
    def forward(self, x):
        return self.model(x)


class EfficientNetModel(nn.Module):
    """EfficientNet model using timm"""
    
    def __init__(
        self,
        model_name: str = 'efficientnet_b0',
        num_classes: int = 10,
        pretrained: bool = True
    ):
        super(EfficientNetModel, self).__init__()
        
        self.model = timm.create_model(
            model_name,
            pretrained=pretrained,
            num_classes=num_classes
        )
    
    def forward(self, x):
        return self.model(x)


class VisionTransformer(nn.Module):
    """Vision Transformer model using timm"""
    
    def __init__(
        self,
        model_name: str = 'vit_tiny_patch16_224',
        num_classes: int = 10,
        pretrained: bool = True,
        img_size: int = 32
    ):
        super(VisionTransformer, self).__init__()
        
        self.model = timm.create_model(
            model_name,
            pretrained=pretrained,
            num_classes=num_classes,
            img_size=img_size
        )
    
    def forward(self, x):
        return self.model(x)


class ConvNeXtModel(nn.Module):
    """ConvNeXt model using timm"""
    
    def __init__(
        self,
        model_name: str = 'convnext_tiny',
        num_classes: int = 10,
        pretrained: bool = True
    ):
        super(ConvNeXtModel, self).__init__()
        
        self.model = timm.create_model(
            model_name,
            pretrained=pretrained,
            num_classes=num_classes
        )
    
    def forward(self, x):
        return self.model(x)


class EnsembleModel(nn.Module):
    """Ensemble of multiple models"""
    
    def __init__(self, models: list, weights: Optional[list] = None):
        super(EnsembleModel, self).__init__()
        self.models = nn.ModuleList(models)
        
        if weights is None:
            self.weights = [1.0 / len(models)] * len(models)
        else:
            self.weights = weights
    
    def forward(self, x):
        outputs = []
        for model in self.models:
            outputs.append(model(x))
        
        # Weighted average of predictions
        weighted_output = torch.zeros_like(outputs[0])
        for output, weight in zip(outputs, self.weights):
            weighted_output += weight * output
        
        return weighted_output


def get_model(
    model_name: str,
    num_classes: int = 10,
    pretrained: bool = True,
    **kwargs
) -> nn.Module:
    """
    Factory function to get model by name
    
    Args:
        model_name: Name of the model
        num_classes: Number of output classes
        pretrained: Whether to use pretrained weights
        **kwargs: Additional model-specific arguments
    
    Returns:
        Model instance
    """
    model_name = model_name.lower()
    
    if model_name == 'simple_cnn':
        return SimpleCNN(num_classes=num_classes)
    elif model_name.startswith('resnet'):
        return ResNetModel(model_name, num_classes, pretrained)
    elif model_name.startswith('efficientnet'):
        return EfficientNetModel(model_name, num_classes, pretrained)
    elif model_name.startswith('vit'):
        return VisionTransformer(model_name, num_classes, pretrained, **kwargs)
    elif model_name.startswith('convnext'):
        return ConvNeXtModel(model_name, num_classes, pretrained)
    else:
        raise ValueError(f"Unknown model: {model_name}")


if __name__ == "__main__":
    # Test models
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Test SimpleCNN
    model = SimpleCNN(num_classes=10).to(device)
    x = torch.randn(2, 3, 32, 32).to(device)
    output = model(x)
    print(f"SimpleCNN output shape: {output.shape}")
    
    # Test ResNet
    model = ResNetModel('resnet18', num_classes=10).to(device)
    output = model(x)
    print(f"ResNet18 output shape: {output.shape}")
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")