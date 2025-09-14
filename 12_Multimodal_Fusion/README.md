# Multimodal Fusion Project - Vision, Audio, and Text Integration

## 1. Problem Definition & Use Case

**Problem:** Effectively combine and integrate information from multiple modalities (vision, audio, text, sensor data) to create more robust, comprehensive AI systems that leverage complementary information sources.

**Use Case:** Multimodal fusion enables enhanced understanding through:
- Autonomous vehicles (camera, LiDAR, radar, GPS fusion)
- Medical diagnosis (imaging, clinical notes, lab results)
- Content understanding (video, audio, captions)
- Human-computer interaction (speech, gesture, facial expression)
- Social media analysis (images, text, hashtags)
- Robotics (vision, touch, proprioception)
- Security systems (face, voice, behavioral biometrics)

**Business Impact:** Multimodal systems achieve 15-30% higher accuracy than single-modal approaches, reduce false positives by 40%, and enable new applications worth $127B market opportunity by 2030.

## 2. Dataset Acquisition & Preprocessing

### Primary Datasets
- **MELD**: Multimodal emotion recognition (text, audio, video)
  ```python
  import pandas as pd
  meld_data = pd.read_csv('MELD_train.csv')
  # Audio features, text transcripts, visual features
  ```
- **VQA (Visual Question Answering)**: Image and text understanding
  ```python
  from datasets import load_dataset
  vqa_dataset = load_dataset('visual_question_answering')
  ```
- **MSR-VTT**: Video description with multiple modalities
  ```python
  msrvtt_data = load_dataset('msrvtt', split='train')
  ```
- **CMU-MOSI**: Multimodal sentiment analysis
  ```python
  mosi_data = load_multimodal_dataset('CMU-MOSI')
  ```

### Data Schema
```python
{
    'sample_id': str,         # Unique identifier
    'modalities': {
        'vision': {
            'frames': np.array,   # Video frames or single image
            'features': np.array, # CNN features (ResNet, etc.)
            'bounding_boxes': list, # Object detection results
            'resolution': tuple   # (width, height)
        },
        'audio': {
            'waveform': np.array,    # Raw audio signal
            'spectrogram': np.array, # Mel-spectrogram
            'mfcc': np.array,       # MFCC features
            'sample_rate': int      # Audio sample rate
        },
        'text': {
            'raw_text': str,        # Original text
            'tokens': list,         # Tokenized text
            'embeddings': np.array, # Pre-trained embeddings
            'sentiment': float      # Sentiment score
        },
        'sensor': {
            'accelerometer': np.array,  # Motion data
            'gyroscope': np.array,     # Orientation data
            'gps': dict               # Location information
        }
    },
    'labels': {
        'emotion': str,      # Primary emotion label
        'sentiment': int,    # Sentiment classification
        'action': str,       # Action recognition
        'scene': str        # Scene classification
    },
    'metadata': {
        'timestamp': datetime,
        'source': str,
        'quality_scores': dict
    }
}
```

### Preprocessing Pipeline
```python
import librosa
import cv2
from transformers import AutoTokenizer, AutoModel
import torch

class MultimodalPreprocessor:
    def __init__(self):
        self.text_tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
        self.text_model = AutoModel.from_pretrained('bert-base-uncased')
        
    def process_vision(self, frames):
        """Process visual data"""
        processed_frames = []
        
        for frame in frames:
            # Resize and normalize
            frame = cv2.resize(frame, (224, 224))
            frame = frame.astype(np.float32) / 255.0
            
            # Data augmentation
            if self.training:
                frame = self.augment_frame(frame)
            
            processed_frames.append(frame)
        
        return np.array(processed_frames)
    
    def process_audio(self, audio_path, target_sr=16000, max_length=10):
        """Process audio data"""
        # Load audio
        audio, sr = librosa.load(audio_path, sr=target_sr)
        
        # Trim silence
        audio = librosa.effects.trim(audio, top_db=20)[0]
        
        # Pad or truncate to fixed length
        target_length = target_sr * max_length
        if len(audio) > target_length:
            audio = audio[:target_length]
        else:
            audio = np.pad(audio, (0, target_length - len(audio)))
        
        # Extract features
        features = {
            'mfcc': librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13),
            'spectral_centroid': librosa.feature.spectral_centroid(y=audio, sr=sr),
            'spectral_rolloff': librosa.feature.spectral_rolloff(y=audio, sr=sr),
            'zero_crossing_rate': librosa.feature.zero_crossing_rate(audio),
            'mel_spectrogram': librosa.feature.melspectrogram(y=audio, sr=sr)
        }
        
        return audio, features
    
    def process_text(self, text):
        """Process text data"""
        # Clean text
        text = self.clean_text(text)
        
        # Tokenize
        tokens = self.text_tokenizer(
            text,
            max_length=512,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        # Get embeddings
        with torch.no_grad():
            outputs = self.text_model(**tokens)
            embeddings = outputs.last_hidden_state.mean(dim=1)
        
        return tokens, embeddings
    
    def synchronize_modalities(self, vision_data, audio_data, text_data):
        """Synchronize different modalities temporally"""
        # Align temporal sequences
        min_length = min(len(vision_data), len(audio_data), len(text_data))
        
        vision_sync = self.resample_sequence(vision_data, min_length)
        audio_sync = self.resample_sequence(audio_data, min_length)
        text_sync = self.resample_sequence(text_data, min_length)
        
        return vision_sync, audio_sync, text_sync
```

## 3. Exploratory Data Analysis (EDA)

### Cross-Modal Analysis
```python
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA

def analyze_multimodal_correlations(data):
    """Analyze correlations between modalities"""
    
    # Extract features from each modality
    vision_features = extract_vision_features(data['vision'])
    audio_features = extract_audio_features(data['audio'])
    text_features = extract_text_features(data['text'])
    
    # Canonical Correlation Analysis
    from sklearn.cross_decomposition import CCA
    cca = CCA(n_components=10)
    
    # Vision-Audio correlation
    va_canonical = cca.fit_transform(vision_features, audio_features)
    print(f"Vision-Audio correlation: {np.corrcoef(va_canonical[0].T, va_canonical[1].T).mean():.3f}")
    
    # Vision-Text correlation
    vt_canonical = cca.fit_transform(vision_features, text_features)
    print(f"Vision-Text correlation: {np.corrcoef(vt_canonical[0].T, vt_canonical[1].T).mean():.3f}")
    
    # Audio-Text correlation
    at_canonical = cca.fit_transform(audio_features, text_features)
    print(f"Audio-Text correlation: {np.corrcoef(at_canonical[0].T, at_canonical[1].T).mean():.3f}")

def visualize_modal_distributions(data):
    """Visualize feature distributions across modalities"""
    
    # t-SNE visualization of combined features
    combined_features = np.concatenate([
        data['vision_features'],
        data['audio_features'],
        data['text_features']
    ], axis=1)
    
    tsne = TSNE(n_components=2, random_state=42)
    embedded = tsne.fit_transform(combined_features)
    
    plt.figure(figsize=(12, 8))
    scatter = plt.scatter(embedded[:, 0], embedded[:, 1], 
                         c=data['labels'], cmap='tab10', alpha=0.7)
    plt.colorbar(scatter)
    plt.title('t-SNE Visualization of Multimodal Features')
    plt.xlabel('t-SNE Component 1')
    plt.ylabel('t-SNE Component 2')
    plt.show()
    
    # Modality importance analysis
    modality_importance = analyze_modality_importance(data)
    
    plt.figure(figsize=(10, 6))
    plt.bar(modality_importance.keys(), modality_importance.values())
    plt.title('Modality Importance for Classification')
    plt.ylabel('Importance Score')
    plt.show()
```

## 4. Feature Engineering & Selection

### Advanced Multimodal Features
```python
import torch.nn as nn
from transformers import CLIPModel, CLIPProcessor

class MultimodalFeatureEngineer:
    def __init__(self):
        self.clip_model = CLIPModel.from_pretrained('openai/clip-vit-base-patch32')
        self.clip_processor = CLIPProcessor.from_pretrained('openai/clip-vit-base-patch32')
        
    def extract_cross_modal_features(self, vision, audio, text):
        """Extract cross-modal interaction features"""
        features = {}
        
        # CLIP features for vision-text alignment
        inputs = self.clip_processor(text=text, images=vision, return_tensors="pt", padding=True)
        outputs = self.clip_model(**inputs)
        
        features['vision_text_similarity'] = torch.cosine_similarity(
            outputs.image_embeds, outputs.text_embeds
        )
        
        # Audio-text alignment using attention
        audio_text_attention = self.compute_attention_alignment(audio, text)
        features['audio_text_attention'] = audio_text_attention
        
        # Temporal synchronization features
        features['temporal_sync'] = self.compute_temporal_synchronization(vision, audio)
        
        # Cross-modal consistency
        features['modal_consistency'] = self.compute_modal_consistency(vision, audio, text)
        
        return features
    
    def create_fusion_features(self, modality_features):
        """Create features from modality fusion"""
        fusion_features = {}
        
        # Early fusion: concatenate raw features
        early_fusion = np.concatenate([
            modality_features['vision'].flatten(),
            modality_features['audio'].flatten(),
            modality_features['text'].flatten()
        ])
        
        # Statistical fusion features
        fusion_features['feature_variance'] = np.var(early_fusion)
        fusion_features['feature_skewness'] = scipy.stats.skew(early_fusion)
        fusion_features['feature_kurtosis'] = scipy.stats.kurtosis(early_fusion)
        
        # Attention-based fusion
        attention_weights = self.compute_modality_attention(modality_features)
        weighted_fusion = (
            attention_weights['vision'] * modality_features['vision'] +
            attention_weights['audio'] * modality_features['audio'] +
            attention_weights['text'] * modality_features['text']
        )
        
        fusion_features['attention_weighted'] = weighted_fusion
        
        return fusion_features
    
    def extract_hierarchical_features(self, data):
        """Extract features at multiple hierarchical levels"""
        hierarchical_features = {}
        
        # Low-level features
        hierarchical_features['low_level'] = {
            'vision': self.extract_low_level_vision(data['vision']),
            'audio': self.extract_low_level_audio(data['audio']),
            'text': self.extract_low_level_text(data['text'])
        }
        
        # Mid-level features
        hierarchical_features['mid_level'] = {
            'vision': self.extract_mid_level_vision(data['vision']),
            'audio': self.extract_mid_level_audio(data['audio']),
            'text': self.extract_mid_level_text(data['text'])
        }
        
        # High-level semantic features
        hierarchical_features['high_level'] = {
            'vision': self.extract_semantic_vision(data['vision']),
            'audio': self.extract_semantic_audio(data['audio']),
            'text': self.extract_semantic_text(data['text'])
        }
        
        return hierarchical_features
```

## 5. Model Architecture & Implementation

### Advanced Fusion Architectures
```python
import torch
import torch.nn as nn
import torch.nn.functional as F

# 1. Early Fusion Model
class EarlyFusionModel(nn.Module):
    def __init__(self, vision_dim, audio_dim, text_dim, num_classes):
        super().__init__()
        self.total_dim = vision_dim + audio_dim + text_dim
        
        self.fusion_layers = nn.Sequential(
            nn.Linear(self.total_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )
        
    def forward(self, vision, audio, text):
        # Concatenate all modalities
        fused = torch.cat([vision, audio, text], dim=1)
        return self.fusion_layers(fused)

# 2. Late Fusion Model
class LateFusionModel(nn.Module):
    def __init__(self, vision_dim, audio_dim, text_dim, num_classes):
        super().__init__()
        
        # Individual modality networks
        self.vision_net = nn.Sequential(
            nn.Linear(vision_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )
        
        self.audio_net = nn.Sequential(
            nn.Linear(audio_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )
        
        self.text_net = nn.Sequential(
            nn.Linear(text_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )
        
    def forward(self, vision, audio, text):
        vision_out = self.vision_net(vision)
        audio_out = self.audio_net(audio)
        text_out = self.text_net(text)
        
        # Average fusion
        return (vision_out + audio_out + text_out) / 3

# 3. Attention-based Fusion
class AttentionFusionModel(nn.Module):
    def __init__(self, vision_dim, audio_dim, text_dim, num_classes, hidden_dim=256):
        super().__init__()
        
        # Project all modalities to same dimension
        self.vision_proj = nn.Linear(vision_dim, hidden_dim)
        self.audio_proj = nn.Linear(audio_dim, hidden_dim)
        self.text_proj = nn.Linear(text_dim, hidden_dim)
        
        # Multi-head attention
        self.multihead_attn = nn.MultiheadAttention(hidden_dim, num_heads=8)
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim // 2, num_classes)
        )
        
    def forward(self, vision, audio, text):
        # Project to common space
        v_proj = self.vision_proj(vision).unsqueeze(0)  # [1, batch, hidden]
        a_proj = self.audio_proj(audio).unsqueeze(0)
        t_proj = self.text_proj(text).unsqueeze(0)
        
        # Combine modalities
        modalities = torch.cat([v_proj, a_proj, t_proj], dim=0)  # [3, batch, hidden]
        
        # Apply attention
        attn_out, attn_weights = self.multihead_attn(modalities, modalities, modalities)
        
        # Global average pooling
        fused_features = torch.mean(attn_out, dim=0)
        
        return self.classifier(fused_features), attn_weights

# 4. Transformer-based Multimodal Fusion
class MultimodalTransformer(nn.Module):
    def __init__(self, vision_dim, audio_dim, text_dim, num_classes, d_model=512, nhead=8, num_layers=6):
        super().__init__()
        
        self.d_model = d_model
        
        # Modality-specific encoders
        self.vision_encoder = nn.Linear(vision_dim, d_model)
        self.audio_encoder = nn.Linear(audio_dim, d_model)
        self.text_encoder = nn.Linear(text_dim, d_model)
        
        # Positional encoding for modalities
        self.modality_embeddings = nn.Embedding(3, d_model)
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(d_model, nhead, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)
        
        # Classification head
        self.classifier = nn.Linear(d_model, num_classes)
        
    def forward(self, vision, audio, text):
        batch_size = vision.size(0)
        
        # Encode modalities
        v_encoded = self.vision_encoder(vision)  # [batch, d_model]
        a_encoded = self.audio_encoder(audio)
        t_encoded = self.text_encoder(text)
        
        # Add modality embeddings
        v_encoded += self.modality_embeddings(torch.tensor(0, device=vision.device))
        a_encoded += self.modality_embeddings(torch.tensor(1, device=vision.device))
        t_encoded += self.modality_embeddings(torch.tensor(2, device=vision.device))
        
        # Stack modalities
        multimodal_input = torch.stack([v_encoded, a_encoded, t_encoded], dim=1)  # [batch, 3, d_model]
        
        # Apply transformer
        transformer_out = self.transformer(multimodal_input)  # [batch, 3, d_model]
        
        # Global average pooling
        pooled = torch.mean(transformer_out, dim=1)  # [batch, d_model]
        
        return self.classifier(pooled)

# 5. Cross-Modal Attention Model
class CrossModalAttention(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(dim * 2, dim),
            nn.Tanh(),
            nn.Linear(dim, 1)
        )
        
    def forward(self, query_modal, key_modal):
        # Compute attention between two modalities
        combined = torch.cat([query_modal, key_modal], dim=-1)
        attention_weights = F.softmax(self.attention(combined), dim=-1)
        
        attended = attention_weights * key_modal
        return attended, attention_weights

class MultimodalCrossAttentionModel(nn.Module):
    def __init__(self, vision_dim, audio_dim, text_dim, num_classes, hidden_dim=256):
        super().__init__()
        
        # Project all modalities
        self.vision_proj = nn.Linear(vision_dim, hidden_dim)
        self.audio_proj = nn.Linear(audio_dim, hidden_dim)
        self.text_proj = nn.Linear(text_dim, hidden_dim)
        
        # Cross-modal attention modules
        self.va_attention = CrossModalAttention(hidden_dim)  # Vision-Audio
        self.vt_attention = CrossModalAttention(hidden_dim)  # Vision-Text
        self.at_attention = CrossModalAttention(hidden_dim)  # Audio-Text
        
        # Final fusion
        self.fusion_layer = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, num_classes)
        )
        
    def forward(self, vision, audio, text):
        # Project to common space
        v_proj = self.vision_proj(vision)
        a_proj = self.audio_proj(audio)
        t_proj = self.text_proj(text)
        
        # Cross-modal attention
        va_attended, va_weights = self.va_attention(v_proj, a_proj)
        vt_attended, vt_weights = self.vt_attention(v_proj, t_proj)
        at_attended, at_weights = self.at_attention(a_proj, t_proj)
        
        # Combine attended features
        combined = torch.cat([va_attended, vt_attended, at_attended], dim=-1)
        
        output = self.fusion_layer(combined)
        
        return output, {'va_weights': va_weights, 'vt_weights': vt_weights, 'at_weights': at_weights}
```

## 6. Training Process & Hyperparameter Tuning

### Advanced Training Strategy
```python
import torch.optim as optim
from torch.utils.data import DataLoader
import wandb

class MultimodalTrainer:
    def __init__(self, model, device='cuda'):
        self.model = model.to(device)
        self.device = device
        self.optimizer = None
        self.scheduler = None
        
    def setup_training(self, learning_rate=1e-3, weight_decay=1e-4):
        """Setup optimizer and scheduler"""
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )
        
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=100, eta_min=1e-6
        )
        
    def compute_modality_weights(self, loss_per_modality):
        """Dynamic modality weighting based on performance"""
        # Inverse performance weighting
        weights = 1.0 / (loss_per_modality + 1e-8)
        weights = weights / weights.sum()
        return weights
    
    def train_epoch(self, train_loader, epoch):
        """Train for one epoch with advanced strategies"""
        self.model.train()
        total_loss = 0
        modality_losses = {'vision': 0, 'audio': 0, 'text': 0}
        
        for batch_idx, (vision, audio, text, labels) in enumerate(train_loader):
            vision, audio, text, labels = (
                vision.to(self.device), audio.to(self.device),
                text.to(self.device), labels.to(self.device)
            )
            
            self.optimizer.zero_grad()
            
            # Forward pass
            if isinstance(self.model, MultimodalCrossAttentionModel):
                outputs, attention_weights = self.model(vision, audio, text)
            else:
                outputs = self.model(vision, audio, text)
            
            # Multi-task loss (if applicable)
            main_loss = F.cross_entropy(outputs, labels)
            
            # Regularization losses
            reg_loss = 0
            
            # L2 regularization
            for param in self.model.parameters():
                reg_loss += torch.norm(param, 2)
            
            # Attention regularization (encourage diversity)
            if 'attention_weights' in locals():
                attention_entropy = -torch.sum(
                    attention_weights['va_weights'] * torch.log(attention_weights['va_weights'] + 1e-8)
                )
                reg_loss += 0.01 * attention_entropy
            
            total_loss_batch = main_loss + 0.001 * reg_loss
            
            # Backward pass
            total_loss_batch.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            self.optimizer.step()
            
            total_loss += total_loss_batch.item()
            
            # Log to wandb
            if batch_idx % 100 == 0:
                wandb.log({
                    'batch_loss': total_loss_batch.item(),
                    'learning_rate': self.optimizer.param_groups[0]['lr'],
                    'epoch': epoch
                })
        
        self.scheduler.step()
        return total_loss / len(train_loader)
    
    def validate(self, val_loader):
        """Validation with detailed metrics"""
        self.model.eval()
        total_loss = 0
        correct = 0
        predictions = []
        targets = []
        
        with torch.no_grad():
            for vision, audio, text, labels in val_loader:
                vision, audio, text, labels = (
                    vision.to(self.device), audio.to(self.device),
                    text.to(self.device), labels.to(self.device)
                )
                
                outputs = self.model(vision, audio, text)
                if isinstance(outputs, tuple):
                    outputs = outputs[0]
                
                loss = F.cross_entropy(outputs, labels)
                total_loss += loss.item()
                
                pred = outputs.argmax(dim=1)
                correct += pred.eq(labels).sum().item()
                
                predictions.extend(pred.cpu().numpy())
                targets.extend(labels.cpu().numpy())
        
        accuracy = correct / len(val_loader.dataset)
        avg_loss = total_loss / len(val_loader)
        
        return avg_loss, accuracy, predictions, targets
    
    def hyperparameter_search(self, train_loader, val_loader):
        """Hyperparameter optimization with Optuna"""
        import optuna
        
        def objective(trial):
            # Suggest hyperparameters
            lr = trial.suggest_loguniform('lr', 1e-5, 1e-2)
            weight_decay = trial.suggest_loguniform('weight_decay', 1e-6, 1e-3)
            batch_size = trial.suggest_categorical('batch_size', [16, 32, 64])
            hidden_dim = trial.suggest_categorical('hidden_dim', [128, 256, 512])
            
            # Create new model with suggested hyperparameters
            model = MultimodalTransformer(
                vision_dim=2048, audio_dim=128, text_dim=768,
                num_classes=10, d_model=hidden_dim
            )
            
            trainer = MultimodalTrainer(model, self.device)
            trainer.setup_training(lr, weight_decay)
            
            # Train for a few epochs
            best_val_acc = 0
            for epoch in range(5):
                train_loss = trainer.train_epoch(train_loader, epoch)
                val_loss, val_acc, _, _ = trainer.validate(val_loader)
                
                best_val_acc = max(best_val_acc, val_acc)
                
                # Pruning
                trial.report(val_acc, epoch)
                if trial.should_prune():
                    raise optuna.exceptions.TrialPruned()
            
            return best_val_acc
        
        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=50)
        
        return study.best_params
```

## 7. Model Evaluation & Metrics

### Comprehensive Multimodal Evaluation
```python
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt

def evaluate_multimodal_model(model, test_loader, device, class_names):
    """Comprehensive evaluation of multimodal model"""
    model.eval()
    
    all_predictions = []
    all_targets = []
    all_attention_weights = []
    modality_contributions = {'vision': [], 'audio': [], 'text': []}
    
    with torch.no_grad():
        for vision, audio, text, labels in test_loader:
            vision, audio, text, labels = (
                vision.to(device), audio.to(device),
                text.to(device), labels.to(device)
            )
            
            # Get predictions
            outputs = model(vision, audio, text)
            if isinstance(outputs, tuple):
                outputs, attention_weights = outputs
                all_attention_weights.append(attention_weights)
            
            predictions = outputs.argmax(dim=1)
            
            all_predictions.extend(predictions.cpu().numpy())
            all_targets.extend(labels.cpu().numpy())
            
            # Analyze modality contributions (ablation)
            if hasattr(model, 'get_modality_contributions'):
                contributions = model.get_modality_contributions(vision, audio, text)
                for modality, contrib in contributions.items():
                    modality_contributions[modality].extend(contrib)
    
    # Classification metrics
    accuracy = accuracy_score(all_targets, all_predictions)
    report = classification_report(all_targets, all_predictions, 
                                 target_names=class_names, output_dict=True)
    
    print(f"Overall Accuracy: {accuracy:.4f}")
    print("\nClassification Report:")
    print(classification_report(all_targets, all_predictions, target_names=class_names))
    
    # Confusion matrix
    cm = confusion_matrix(all_targets, all_predictions)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.show()
    
    return {
        'accuracy': accuracy,
        'classification_report': report,
        'confusion_matrix': cm,
        'attention_weights': all_attention_weights,
        'modality_contributions': modality_contributions
    }

def ablation_study(model, test_loader, device):
    """Perform ablation study to understand modality importance"""
    model.eval()
    
    # Test with all modalities
    full_accuracy = evaluate_single_config(model, test_loader, device, 
                                          use_vision=True, use_audio=True, use_text=True)
    
    # Test with single modalities
    vision_only = evaluate_single_config(model, test_loader, device,
                                       use_vision=True, use_audio=False, use_text=False)
    audio_only = evaluate_single_config(model, test_loader, device,
                                      use_vision=False, use_audio=True, use_text=False)
    text_only = evaluate_single_config(model, test_loader, device,
                                     use_vision=False, use_audio=False, use_text=True)
    
    # Test with pairs
    vision_audio = evaluate_single_config(model, test_loader, device,
                                        use_vision=True, use_audio=True, use_text=False)
    vision_text = evaluate_single_config(model, test_loader, device,
                                       use_vision=True, use_audio=False, use_text=True)
    audio_text = evaluate_single_config(model, test_loader, device,
                                      use_vision=False, use_audio=True, use_text=True)
    
    results = {
        'Full Model': full_accuracy,
        'Vision Only': vision_only,
        'Audio Only': audio_only,
        'Text Only': text_only,
        'Vision + Audio': vision_audio,
        'Vision + Text': vision_text,
        'Audio + Text': audio_text
    }
    
    # Visualize results
    plt.figure(figsize=(12, 6))
    plt.bar(results.keys(), results.values())
    plt.title('Ablation Study Results')
    plt.ylabel('Accuracy')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()
    
    return results

def analyze_attention_patterns(attention_weights, class_names):
    """Analyze attention patterns across classes and modalities"""
    
    # Average attention weights per class
    class_attention = {}
    
    for class_name in class_names:
        class_attention[class_name] = {
            'vision_audio': [],
            'vision_text': [],
            'audio_text': []
        }
    
    # Aggregate attention weights
    for weights in attention_weights:
        for modality_pair, attention in weights.items():
            # Process attention patterns
            pass
    
    # Visualization
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    modality_pairs = ['vision_audio', 'vision_text', 'audio_text']
    
    for i, pair in enumerate(modality_pairs):
        attention_matrix = np.array([class_attention[cls][pair] for cls in class_names])
        
        sns.heatmap(attention_matrix, annot=True, cmap='viridis',
                   xticklabels=class_names, yticklabels=class_names, ax=axes[i])
        axes[i].set_title(f'Attention Pattern: {pair.replace("_", " ").title()}')
    
    plt.tight_layout()
    plt.show()
```

## 8. Results & Performance Analysis

### Performance Comparison
```python
# Example comprehensive results
model_performance = {
    'Early Fusion': {
        'Accuracy': 0.742, 'Precision': 0.738, 'Recall': 0.745, 'F1': 0.741,
        'Training Time': '45min', 'Parameters': '2.3M'
    },
    'Late Fusion': {
        'Accuracy': 0.768, 'Precision': 0.771, 'Recall': 0.765, 'F1': 0.768,
        'Training Time': '52min', 'Parameters': '3.1M'
    },
    'Attention Fusion': {
        'Accuracy': 0.834, 'Precision': 0.831, 'Recall': 0.836, 'F1': 0.833,
        'Training Time': '73min', 'Parameters': '4.7M'
    },
    'Transformer': {
        'Accuracy': 0.892, 'Precision': 0.889, 'Recall': 0.894, 'F1': 0.891,
        'Training Time': '120min', 'Parameters': '8.2M'
    },
    'Cross-Modal Attention': {
        'Accuracy': 0.916, 'Precision': 0.913, 'Recall': 0.918, 'F1': 0.915,
        'Training Time': '95min', 'Parameters': '6.5M'
    }
}

# Modality contribution analysis
modality_contributions = {
    'Vision': 0.42,
    'Audio': 0.31,
    'Text': 0.27
}

# Cross-modal interactions
interaction_effects = {
    'Vision-Audio': 0.089,  # Improvement when combining
    'Vision-Text': 0.124,
    'Audio-Text': 0.076,
    'All Three': 0.156
}
```

### Key Findings
- **Cross-Modal Attention** achieved best performance (91.6% accuracy)
- **Text modality** most discriminative for sentiment tasks
- **Vision-Text** interactions provide strongest complementary information
- **Transformer architecture** best for sequential multimodal data
- **Early fusion** fastest but least effective approach

## 9. Production Deployment

### Scalable Multimodal System
```python
import asyncio
import aiohttp
from fastapi import FastAPI, UploadFile, File, Form
import uvicorn

class MultimodalInferenceService:
    def __init__(self):
        self.app = FastAPI(title="Multimodal Inference API")
        self.models = self.load_models()
        self.setup_routes()
        
    def load_models(self):
        """Load all multimodal models"""
        return {
            'cross_attention': torch.jit.load('models/cross_attention_model.pt'),
            'transformer': torch.jit.load('models/transformer_model.pt'),
            'ensemble': self.load_ensemble_models()
        }
    
    def setup_routes(self):
        @self.app.post("/predict/multimodal")
        async def predict_multimodal(
            image: UploadFile = File(...),
            audio: UploadFile = File(...),
            text: str = Form(...)
        ):
            try:
                # Process inputs concurrently
                vision_task = self.process_image(await image.read())
                audio_task = self.process_audio(await audio.read())
                text_task = self.process_text(text)
                
                vision_features, audio_features, text_features = await asyncio.gather(
                    vision_task, audio_task, text_task
                )
                
                # Run inference
                predictions = {}
                for model_name, model in self.models.items():
                    pred = model(vision_features, audio_features, text_features)
                    predictions[model_name] = {
                        'class': int(pred.argmax()),
                        'confidence': float(pred.max()),
                        'probabilities': pred.tolist()
                    }
                
                # Ensemble prediction
                ensemble_pred = self.ensemble_predict(predictions)
                
                return {
                    'ensemble_prediction': ensemble_pred,
                    'individual_predictions': predictions,
                    'modality_contributions': self.get_modality_contributions(
                        vision_features, audio_features, text_features
                    )
                }
                
            except Exception as e:
                return {'error': str(e)}
        
        @self.app.get("/health")
        async def health_check():
            return {
                'status': 'healthy',
                'models_loaded': len(self.models),
                'gpu_available': torch.cuda.is_available()
            }
    
    async def process_image(self, image_bytes):
        """Asynchronous image processing"""
        image = Image.open(io.BytesIO(image_bytes))
        # Preprocessing and feature extraction
        return image_features
    
    async def process_audio(self, audio_bytes):
        """Asynchronous audio processing"""
        # Audio processing and feature extraction
        return audio_features
    
    async def process_text(self, text):
        """Asynchronous text processing"""
        # Text preprocessing and embedding
        return text_features

# Kubernetes deployment configuration
k8s_deployment = '''
apiVersion: apps/v1
kind: Deployment
metadata:
  name: multimodal-inference
spec:
  replicas: 3
  selector:
    matchLabels:
      app: multimodal-inference
  template:
    metadata:
      labels:
        app: multimodal-inference
    spec:
      containers:
      - name: multimodal-api
        image: multimodal-inference:latest
        ports:
        - containerPort: 8000
        resources:
          requests:
            memory: "4Gi"
            cpu: "1000m"
            nvidia.com/gpu: 1
          limits:
            memory: "8Gi"
            cpu: "2000m"
            nvidia.com/gpu: 1
        env:
        - name: MODEL_PATH
          value: "/app/models"
        - name: BATCH_SIZE
          value: "8"
        volumeMounts:
        - name: model-storage
          mountPath: /app/models
      volumes:
      - name: model-storage
        persistentVolumeClaim:
          claimName: model-pvc
'''

# Docker configuration
dockerfile_multimodal = '''
FROM nvidia/cuda:11.8-runtime-ubuntu20.04

RUN apt-get update && apt-get install -y python3 python3-pip ffmpeg

WORKDIR /app
COPY requirements.txt .
RUN pip3 install -r requirements.txt

COPY . .
EXPOSE 8000

CMD ["uvicorn", "multimodal_api:app", "--host", "0.0.0.0", "--port", "8000"]
'''
```

## 10. Future Improvements & Extensions

### Advanced Research Directions

1. **Foundation Model Integration**
   ```python
   # CLIP, DALL-E, GPT integration
   class FoundationModelFusion:
       def __init__(self):
           self.clip = CLIPModel.from_pretrained('openai/clip-vit-large-patch14')
           self.gpt = GPTModel.from_pretrained('gpt-3.5-turbo')
           
       def unified_embedding_space(self, vision, audio, text):
           # Map all modalities to unified CLIP space
           vision_embed = self.clip.encode_image(vision)
           text_embed = self.clip.encode_text(text)
           audio_embed = self.map_audio_to_clip(audio)
           return vision_embed, audio_embed, text_embed
   ```

2. **Self-Supervised Learning**
   - Contrastive learning across modalities
   - Masked autoencoding for multimodal data
   - Cross-modal prediction tasks

3. **Few-Shot Multimodal Learning**
   - Meta-learning for rapid adaptation
   - Prototypical networks for multimodal data
   - In-context learning with multimodal prompts

4. **Causal Multimodal Models**
   ```python
   class CausalMultimodalModel:
       def learn_causal_structure(self, data):
           # Learn causal relationships between modalities
           causal_graph = self.discover_causal_structure(data)
           return causal_graph
   ```

5. **Continual Multimodal Learning**
   - Online adaptation to new modalities
   - Preventing catastrophic forgetting
   - Dynamic architecture expansion

### Emerging Applications
- **Multimodal Retrieval**: Search across vision, audio, and text
- **Embodied AI**: Robotics with multimodal perception
- **Digital Humans**: Realistic avatars with multimodal understanding
- **Scientific Discovery**: Multimodal analysis of research data
- **Creative AI**: Generating content across modalities

**Next Steps:**
1. Implement foundation model integration
2. Develop online learning capabilities  
3. Create multimodal retrieval system
4. Build real-time streaming pipeline
5. Explore causal multimodal reasoning