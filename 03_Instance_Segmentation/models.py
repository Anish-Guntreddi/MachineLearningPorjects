"""
Instance segmentation models
"""
import torch
import torch.nn as nn
import torchvision
from torchvision.models.detection import MaskRCNN
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from typing import Dict, List, Optional


def get_mask_rcnn(
    num_classes: int,
    pretrained: bool = True,
    backbone_name: str = 'resnet50',
    trainable_backbone_layers: int = 3,
    hidden_layer: int = 256
) -> MaskRCNN:
    """
    Get Mask R-CNN model for instance segmentation
    
    Args:
        num_classes: Number of classes (including background)
        pretrained: Use pretrained backbone
        backbone_name: Backbone architecture
        trainable_backbone_layers: Number of trainable backbone layers
        hidden_layer: Hidden layer size for mask predictor
    
    Returns:
        Mask R-CNN model
    """
    if pretrained:
        # Load pretrained model
        model = torchvision.models.detection.maskrcnn_resnet50_fpn(
            pretrained=True,
            trainable_backbone_layers=trainable_backbone_layers
        )
        
        # Replace the box predictor
        in_features = model.roi_heads.box_predictor.cls_score.in_features
        model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
        
        # Replace the mask predictor
        in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
        model.roi_heads.mask_predictor = MaskRCNNPredictor(
            in_features_mask,
            hidden_layer,
            num_classes
        )
    else:
        # Create model from scratch
        model = torchvision.models.detection.maskrcnn_resnet50_fpn(
            pretrained=False,
            num_classes=num_classes,
            trainable_backbone_layers=trainable_backbone_layers
        )
    
    return model


class SimpleInstanceSegmentation(nn.Module):
    """Simple instance segmentation model for learning"""
    
    def __init__(self, num_classes: int = 21):
        super(SimpleInstanceSegmentation, self).__init__()
        self.num_classes = num_classes
        
        # Backbone (simplified ResNet-like)
        self.backbone = nn.ModuleDict({
            'conv1': nn.Sequential(
                nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
            ),
            'layer1': self._make_layer(64, 64, 2),
            'layer2': self._make_layer(64, 128, 2, stride=2),
            'layer3': self._make_layer(128, 256, 2, stride=2),
            'layer4': self._make_layer(256, 512, 2, stride=2)
        })
        
        # FPN (Feature Pyramid Network)
        self.fpn = nn.ModuleDict({
            'inner4': nn.Conv2d(512, 256, 1),
            'inner3': nn.Conv2d(256, 256, 1),
            'inner2': nn.Conv2d(128, 256, 1),
            'inner1': nn.Conv2d(64, 256, 1),
            
            'layer4': nn.Conv2d(256, 256, 3, padding=1),
            'layer3': nn.Conv2d(256, 256, 3, padding=1),
            'layer2': nn.Conv2d(256, 256, 3, padding=1),
            'layer1': nn.Conv2d(256, 256, 3, padding=1)
        })
        
        # Detection head
        self.box_head = nn.Sequential(
            nn.Linear(256 * 7 * 7, 1024),
            nn.ReLU(),
            nn.Linear(1024, 1024),
            nn.ReLU()
        )
        
        self.cls_score = nn.Linear(1024, num_classes)
        self.bbox_pred = nn.Linear(1024, num_classes * 4)
        
        # Mask head
        self.mask_head = nn.Sequential(
            nn.Conv2d(256, 256, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(256, 256, 2, stride=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, num_classes, 1)
        )
    
    def _make_layer(self, in_channels, out_channels, blocks, stride=1):
        """Create a residual layer"""
        layers = []
        layers.append(nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, stride, 1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        ))
        
        for _ in range(1, blocks):
            layers.append(nn.Sequential(
                nn.Conv2d(out_channels, out_channels, 3, 1, 1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)
            ))
        
        return nn.Sequential(*layers)
    
    def forward(self, images, targets=None):
        """
        Forward pass
        
        Args:
            images: List of tensors or tensor batch
            targets: List of target dictionaries (training only)
        
        Returns:
            Losses during training, predictions during inference
        """
        # For simplicity, return dummy outputs
        # In practice, implement full forward pass with RPN, RoI pooling, etc.
        
        if self.training and targets is not None:
            # Training mode - return losses
            losses = {
                'loss_classifier': torch.tensor(0.5, requires_grad=True),
                'loss_box_reg': torch.tensor(0.3, requires_grad=True),
                'loss_mask': torch.tensor(0.4, requires_grad=True),
                'loss_objectness': torch.tensor(0.2, requires_grad=True),
                'loss_rpn_box_reg': torch.tensor(0.1, requires_grad=True)
            }
            return losses
        else:
            # Inference mode - return predictions
            predictions = []
            for img in images if isinstance(images, list) else [images]:
                pred = {
                    'boxes': torch.rand(5, 4) * 100,
                    'labels': torch.randint(0, self.num_classes, (5,)),
                    'scores': torch.rand(5),
                    'masks': torch.rand(5, 1, 28, 28)
                }
                predictions.append(pred)
            return predictions


def get_model(
    model_name: str,
    num_classes: int,
    pretrained: bool = True,
    **kwargs
) -> nn.Module:
    """
    Factory function to get instance segmentation model
    
    Args:
        model_name: Name of the model
        num_classes: Number of classes
        pretrained: Use pretrained weights
        **kwargs: Additional model-specific arguments
    
    Returns:
        Instance segmentation model
    """
    model_name = model_name.lower()
    
    if model_name == 'mask_rcnn':
        return get_mask_rcnn(num_classes, pretrained, **kwargs)
    elif model_name == 'simple':
        return SimpleInstanceSegmentation(num_classes)
    else:
        raise ValueError(f"Unknown model: {model_name}")


if __name__ == "__main__":
    # Test models
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Test Mask R-CNN
    model = get_model('mask_rcnn', num_classes=21)
    model.eval()
    
    # Test input
    x = [torch.randn(3, 300, 400)]
    
    with torch.no_grad():
        predictions = model(x)
    
    print(f"Mask R-CNN predictions: {len(predictions)} images")
    if predictions:
        print(f"First prediction keys: {predictions[0].keys()}")
        if 'masks' in predictions[0]:
            print(f"Mask shape: {predictions[0]['masks'].shape}")