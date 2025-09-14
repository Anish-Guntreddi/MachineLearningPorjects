"""
Object detection models
"""
import torch
import torch.nn as nn
import torchvision
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection import retinanet_resnet50_fpn
from typing import Dict, List, Optional
import warnings
warnings.filterwarnings('ignore')


def get_faster_rcnn(
    num_classes: int,
    pretrained: bool = True,
    backbone_name: str = 'resnet50',
    trainable_backbone_layers: int = 3
) -> FasterRCNN:
    """
    Get Faster R-CNN model
    
    Args:
        num_classes: Number of classes (including background)
        pretrained: Use pretrained backbone
        backbone_name: Backbone architecture
        trainable_backbone_layers: Number of trainable backbone layers
    
    Returns:
        Faster R-CNN model
    """
    if pretrained:
        # Load pretrained model
        model = torchvision.models.detection.fasterrcnn_resnet50_fpn(
            pretrained=True,
            trainable_backbone_layers=trainable_backbone_layers
        )
        
        # Replace the classifier head
        in_features = model.roi_heads.box_predictor.cls_score.in_features
        model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    else:
        # Create model from scratch
        model = torchvision.models.detection.fasterrcnn_resnet50_fpn(
            pretrained=False,
            num_classes=num_classes,
            trainable_backbone_layers=trainable_backbone_layers
        )
    
    return model


def get_retinanet(
    num_classes: int,
    pretrained: bool = True,
    trainable_backbone_layers: int = 3
) -> nn.Module:
    """
    Get RetinaNet model
    
    Args:
        num_classes: Number of classes
        pretrained: Use pretrained model
        trainable_backbone_layers: Number of trainable backbone layers
    
    Returns:
        RetinaNet model
    """
    if pretrained:
        model = retinanet_resnet50_fpn(
            pretrained=True,
            num_classes=91,  # COCO classes
            trainable_backbone_layers=trainable_backbone_layers
        )
        
        # Replace classification head
        num_anchors = model.head.classification_head.num_anchors
        model.head.classification_head.num_classes = num_classes
        
        cls_logits = nn.Conv2d(256, num_anchors * num_classes, kernel_size=3, stride=1, padding=1)
        torch.nn.init.normal_(cls_logits.weight, std=0.01)
        torch.nn.init.constant_(cls_logits.bias, -4.0)
        model.head.classification_head.cls_logits = cls_logits
    else:
        model = retinanet_resnet50_fpn(
            pretrained=False,
            num_classes=num_classes,
            trainable_backbone_layers=trainable_backbone_layers
        )
    
    return model


class SimpleDetector(nn.Module):
    """Simple object detection model for learning purposes"""
    
    def __init__(self, num_classes: int, img_size: int = 416):
        super(SimpleDetector, self).__init__()
        self.num_classes = num_classes
        self.img_size = img_size
        
        # Backbone
        self.backbone = nn.Sequential(
            # Block 1
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # Block 2
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # Block 3
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # Block 4
            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # Block 5
            nn.Conv2d(512, 1024, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(1024),
            nn.ReLU(inplace=True),
        )
        
        # Detection head
        self.detection_head = nn.Sequential(
            nn.Conv2d(1024, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )
        
        # Output layers
        self.num_anchors = 3
        self.bbox_pred = nn.Conv2d(256, self.num_anchors * 4, kernel_size=1)
        self.cls_pred = nn.Conv2d(256, self.num_anchors * num_classes, kernel_size=1)
        self.obj_pred = nn.Conv2d(256, self.num_anchors * 1, kernel_size=1)
    
    def forward(self, x):
        # Extract features
        features = self.backbone(x)
        detection_features = self.detection_head(features)
        
        # Predictions
        bbox_pred = self.bbox_pred(detection_features)
        cls_pred = self.cls_pred(detection_features)
        obj_pred = self.obj_pred(detection_features)
        
        batch_size = x.size(0)
        h, w = detection_features.size(2), detection_features.size(3)
        
        # Reshape predictions
        bbox_pred = bbox_pred.view(batch_size, self.num_anchors, 4, h, w)
        cls_pred = cls_pred.view(batch_size, self.num_anchors, self.num_classes, h, w)
        obj_pred = obj_pred.view(batch_size, self.num_anchors, 1, h, w)
        
        return {
            'bbox': bbox_pred,
            'class': cls_pred,
            'objectness': obj_pred
        }


class YOLOv5(nn.Module):
    """Simplified YOLOv5-like architecture"""
    
    def __init__(self, num_classes: int = 80, anchors: Optional[List] = None):
        super(YOLOv5, self).__init__()
        self.num_classes = num_classes
        
        if anchors is None:
            # Default anchors for 3 detection layers
            self.anchors = [
                [[10, 13], [16, 30], [33, 23]],      # Small objects
                [[30, 61], [62, 45], [59, 119]],     # Medium objects
                [[116, 90], [156, 198], [373, 326]]  # Large objects
            ]
        else:
            self.anchors = anchors
        
        self.num_anchors = len(self.anchors[0])
        
        # Backbone (CSPDarknet-like)
        self.backbone = self._make_backbone()
        
        # Neck (FPN + PAN)
        self.neck = self._make_neck()
        
        # Detection heads
        self.detect = self._make_detect_layers()
    
    def _make_backbone(self):
        """Create backbone network"""
        return nn.ModuleList([
            # P1
            self._conv_block(3, 32, kernel_size=6, stride=2, padding=2),
            # P2
            self._conv_block(32, 64, kernel_size=3, stride=2, padding=1),
            self._csp_block(64, 64, n=1),
            # P3
            self._conv_block(64, 128, kernel_size=3, stride=2, padding=1),
            self._csp_block(128, 128, n=2),
            # P4
            self._conv_block(128, 256, kernel_size=3, stride=2, padding=1),
            self._csp_block(256, 256, n=8),
            # P5
            self._conv_block(256, 512, kernel_size=3, stride=2, padding=1),
            self._csp_block(512, 512, n=8),
            # P6
            self._conv_block(512, 1024, kernel_size=3, stride=2, padding=1),
            self._csp_block(1024, 1024, n=4),
        ])
    
    def _make_neck(self):
        """Create FPN + PAN neck"""
        return nn.ModuleList([
            # FPN
            nn.Sequential(
                nn.Conv2d(1024, 512, kernel_size=1),
                nn.Upsample(scale_factor=2, mode='nearest'),
            ),
            nn.Sequential(
                nn.Conv2d(512, 256, kernel_size=1),
                nn.Upsample(scale_factor=2, mode='nearest'),
            ),
            # PAN
            self._conv_block(256, 256, kernel_size=3, stride=2, padding=1),
            self._conv_block(512, 512, kernel_size=3, stride=2, padding=1),
        ])
    
    def _make_detect_layers(self):
        """Create detection layers"""
        out_channels = self.num_anchors * (5 + self.num_classes)
        return nn.ModuleList([
            nn.Conv2d(256, out_channels, kernel_size=1),   # Small
            nn.Conv2d(512, out_channels, kernel_size=1),   # Medium
            nn.Conv2d(1024, out_channels, kernel_size=1),  # Large
        ])
    
    def _conv_block(self, in_channels, out_channels, kernel_size=1, stride=1, padding=0):
        """Convolution block with BatchNorm and SiLU"""
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.SiLU(inplace=True)
        )
    
    def _csp_block(self, in_channels, out_channels, n=1):
        """Cross Stage Partial block"""
        hidden_channels = out_channels // 2
        return nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, kernel_size=1),
            nn.Sequential(*[
                self._bottleneck(hidden_channels, hidden_channels) for _ in range(n)
            ]),
            nn.Conv2d(hidden_channels, out_channels, kernel_size=1)
        )
    
    def _bottleneck(self, in_channels, out_channels):
        """Bottleneck block"""
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1),
            nn.BatchNorm2d(out_channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.SiLU(inplace=True)
        )
    
    def forward(self, x):
        # Backbone forward
        p3 = self.backbone[4](self.backbone[3](self.backbone[2](self.backbone[1](self.backbone[0](x)))))
        p4 = self.backbone[6](self.backbone[5](p3))
        p5 = self.backbone[9](self.backbone[8](self.backbone[7](p4)))
        
        # Neck forward (simplified)
        f5 = p5
        f4 = p4 + self.neck[0](f5)
        f3 = p3 + self.neck[1](f4)
        
        # Detection outputs
        out_small = self.detect[0](f3)
        out_medium = self.detect[1](f4)
        out_large = self.detect[2](f5)
        
        return [out_small, out_medium, out_large]


def get_model(
    model_name: str,
    num_classes: int,
    pretrained: bool = True,
    **kwargs
) -> nn.Module:
    """
    Factory function to get detection model
    
    Args:
        model_name: Name of the model
        num_classes: Number of classes
        pretrained: Use pretrained weights
        **kwargs: Additional model-specific arguments
    
    Returns:
        Detection model
    """
    model_name = model_name.lower()
    
    if model_name == 'faster_rcnn':
        return get_faster_rcnn(num_classes, pretrained, **kwargs)
    elif model_name == 'retinanet':
        return get_retinanet(num_classes, pretrained, **kwargs)
    elif model_name == 'simple':
        return SimpleDetector(num_classes, **kwargs)
    elif model_name == 'yolov5':
        return YOLOv5(num_classes, **kwargs)
    else:
        raise ValueError(f"Unknown model: {model_name}")


if __name__ == "__main__":
    # Test models
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Test Faster R-CNN
    model = get_model('faster_rcnn', num_classes=21)
    model.eval()
    
    # Test input
    x = [torch.randn(3, 300, 400), torch.randn(3, 500, 400)]
    
    with torch.no_grad():
        predictions = model(x)
    
    print(f"Faster R-CNN predictions: {len(predictions)} images")
    print(f"First prediction keys: {predictions[0].keys()}")
    
    # Test YOLOv5
    model = YOLOv5(num_classes=80)
    x = torch.randn(2, 3, 640, 640)
    outputs = model(x)
    print(f"\nYOLOv5 outputs: {len(outputs)} detection layers")
    for i, out in enumerate(outputs):
        print(f"Layer {i} output shape: {out.shape}")