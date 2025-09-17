"""
Data loading and preprocessing for Multimodal Fusion
"""
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union
from PIL import Image
import librosa
from transformers import AutoTokenizer
import warnings
warnings.filterwarnings('ignore')


class MultimodalDataset(Dataset):
    """Dataset for multimodal learning"""
    
    def __init__(
        self,
        data_path: str,
        modalities: List[str] = ['image', 'text', 'audio'],
        image_transform: Optional[transforms.Compose] = None,
        text_tokenizer: Optional[object] = None,
        audio_sr: int = 16000,
        audio_duration: float = 3.0,
        max_text_length: int = 128
    ):
        """
        Args:
            data_path: Path to multimodal data
            modalities: List of modalities to use
            image_transform: Image transformations
            text_tokenizer: Text tokenizer
            audio_sr: Audio sample rate
            audio_duration: Audio duration in seconds
            max_text_length: Maximum text sequence length
        """
        self.data_path = Path(data_path)
        self.modalities = modalities
        self.audio_sr = audio_sr
        self.audio_duration = audio_duration
        self.audio_length = int(audio_sr * audio_duration)
        self.max_text_length = max_text_length
        
        # Setup transforms
        self.image_transform = image_transform or self._get_default_image_transform()
        self.text_tokenizer = text_tokenizer or self._get_default_tokenizer()
        
        # Load data manifest
        self.samples = self._load_manifest()
    
    def _get_default_image_transform(self):
        """Get default image transformation"""
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                              std=[0.229, 0.224, 0.225])
        ])
    
    def _get_default_tokenizer(self):
        """Get default text tokenizer"""
        return AutoTokenizer.from_pretrained('bert-base-uncased')
    
    def _load_manifest(self) -> List[Dict]:
        """Load data manifest"""
        manifest_file = self.data_path / 'manifest.csv'
        
        if manifest_file.exists():
            df = pd.read_csv(manifest_file)
            samples = df.to_dict('records')
        else:
            # Generate synthetic manifest
            samples = self._generate_synthetic_manifest()
        
        return samples
    
    def _generate_synthetic_manifest(self) -> List[Dict]:
        """Generate synthetic manifest for demonstration"""
        samples = []
        
        for i in range(1000):
            sample = {
                'id': i,
                'image_path': f'image_{i}.jpg',
                'text': f'This is sample text number {i} for multimodal learning.',
                'audio_path': f'audio_{i}.wav',
                'label': np.random.randint(0, 10)
            }
            samples.append(sample)
        
        return samples
    
    def _load_image(self, image_path: str) -> torch.Tensor:
        """Load and transform image"""
        full_path = self.data_path / 'images' / image_path
        
        if full_path.exists():
            image = Image.open(full_path).convert('RGB')
        else:
            # Generate synthetic image
            image = Image.fromarray(np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8))
        
        return self.image_transform(image)
    
    def _load_text(self, text: str) -> Dict[str, torch.Tensor]:
        """Tokenize and encode text"""
        encoded = self.text_tokenizer(
            text,
            padding='max_length',
            truncation=True,
            max_length=self.max_text_length,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoded['input_ids'].squeeze(0),
            'attention_mask': encoded['attention_mask'].squeeze(0)
        }
    
    def _load_audio(self, audio_path: str) -> torch.Tensor:
        """Load and process audio"""
        full_path = self.data_path / 'audio' / audio_path
        
        if full_path.exists():
            audio, sr = librosa.load(full_path, sr=self.audio_sr)
        else:
            # Generate synthetic audio
            t = np.linspace(0, self.audio_duration, self.audio_length)
            audio = np.sin(2 * np.pi * 440 * t) + np.random.randn(self.audio_length) * 0.1
        
        # Pad or crop to fixed length
        if len(audio) < self.audio_length:
            audio = np.pad(audio, (0, self.audio_length - len(audio)))
        else:
            audio = audio[:self.audio_length]
        
        # Extract features (MFCC)
        mfcc = librosa.feature.mfcc(y=audio, sr=self.audio_sr, n_mfcc=40)
        
        return torch.tensor(mfcc, dtype=torch.float32)
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        output = {'label': torch.tensor(sample['label'], dtype=torch.long)}
        
        # Load each modality
        if 'image' in self.modalities:
            output['image'] = self._load_image(sample['image_path'])
        
        if 'text' in self.modalities:
            text_data = self._load_text(sample['text'])
            output['text_ids'] = text_data['input_ids']
            output['text_mask'] = text_data['attention_mask']
        
        if 'audio' in self.modalities:
            output['audio'] = self._load_audio(sample['audio_path'])
        
        return output


class VQADataset(Dataset):
    """Visual Question Answering dataset"""
    
    def __init__(
        self,
        data_path: str,
        split: str = 'train',
        image_transform: Optional[transforms.Compose] = None,
        text_tokenizer: Optional[object] = None,
        max_question_length: int = 32,
        num_answers: int = 1000
    ):
        """
        Args:
            data_path: Path to VQA data
            split: Data split ('train', 'val', 'test')
            image_transform: Image transformations
            text_tokenizer: Text tokenizer
            max_question_length: Maximum question length
            num_answers: Number of possible answers
        """
        self.data_path = Path(data_path)
        self.split = split
        self.max_question_length = max_question_length
        self.num_answers = num_answers
        
        self.image_transform = image_transform or self._get_default_image_transform()
        self.text_tokenizer = text_tokenizer or AutoTokenizer.from_pretrained('bert-base-uncased')
        
        # Load VQA data
        self.samples = self._load_vqa_data()
        
        # Build answer vocabulary
        self.answer_to_idx = self._build_answer_vocab()
    
    def _get_default_image_transform(self):
        """Get default image transformation"""
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                              std=[0.229, 0.224, 0.225])
        ])
    
    def _load_vqa_data(self) -> List[Dict]:
        """Load VQA dataset"""
        # Placeholder - generate synthetic VQA data
        samples = []
        
        for i in range(1000):
            sample = {
                'image_id': i,
                'question': f'What is in the image {i}?',
                'answer': f'Object {i % 10}',
                'question_id': i
            }
            samples.append(sample)
        
        return samples
    
    def _build_answer_vocab(self) -> Dict[str, int]:
        """Build answer vocabulary"""
        vocab = {}
        
        # Get unique answers
        answers = set()
        for sample in self.samples:
            answers.add(sample['answer'])
        
        # Create vocabulary
        for idx, answer in enumerate(sorted(answers)):
            vocab[answer] = idx
        
        return vocab
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        # Load image
        image_path = self.data_path / 'images' / f"{sample['image_id']}.jpg"
        if image_path.exists():
            image = Image.open(image_path).convert('RGB')
        else:
            # Generate synthetic image
            image = Image.fromarray(np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8))
        
        image = self.image_transform(image)
        
        # Encode question
        question_encoded = self.text_tokenizer(
            sample['question'],
            padding='max_length',
            truncation=True,
            max_length=self.max_question_length,
            return_tensors='pt'
        )
        
        # Encode answer
        answer_idx = self.answer_to_idx.get(sample['answer'], 0)
        
        return {
            'image': image,
            'question_ids': question_encoded['input_ids'].squeeze(0),
            'question_mask': question_encoded['attention_mask'].squeeze(0),
            'answer': torch.tensor(answer_idx, dtype=torch.long)
        }


class ImageTextDataset(Dataset):
    """Image-Text matching dataset (e.g., CLIP-style)"""
    
    def __init__(
        self,
        data_path: str,
        image_transform: Optional[transforms.Compose] = None,
        text_tokenizer: Optional[object] = None,
        max_text_length: int = 77,
        negative_sampling: bool = True
    ):
        """
        Args:
            data_path: Path to image-text data
            image_transform: Image transformations
            text_tokenizer: Text tokenizer
            max_text_length: Maximum text length
            negative_sampling: Whether to include negative pairs
        """
        self.data_path = Path(data_path)
        self.max_text_length = max_text_length
        self.negative_sampling = negative_sampling
        
        self.image_transform = image_transform or self._get_default_image_transform()
        self.text_tokenizer = text_tokenizer or AutoTokenizer.from_pretrained('bert-base-uncased')
        
        # Load image-text pairs
        self.samples = self._load_image_text_pairs()
    
    def _get_default_image_transform(self):
        """Get default image transformation"""
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                              std=[0.229, 0.224, 0.225])
        ])
    
    def _load_image_text_pairs(self) -> List[Dict]:
        """Load image-text pairs"""
        # Placeholder - generate synthetic data
        samples = []
        
        for i in range(1000):
            sample = {
                'image_path': f'image_{i}.jpg',
                'caption': f'A description of image {i} with various objects and scenes.',
                'pair_id': i
            }
            samples.append(sample)
        
        return samples
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        # Load image
        image_path = self.data_path / 'images' / sample['image_path']
        if image_path.exists():
            image = Image.open(image_path).convert('RGB')
        else:
            # Generate synthetic image
            image = Image.fromarray(np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8))
        
        image = self.image_transform(image)
        
        # Get positive text
        positive_text = sample['caption']
        
        # Get negative text if needed
        if self.negative_sampling and np.random.random() > 0.5:
            # Sample random caption from another image
            neg_idx = np.random.randint(0, len(self.samples))
            while neg_idx == idx:
                neg_idx = np.random.randint(0, len(self.samples))
            
            text = self.samples[neg_idx]['caption']
            label = 0  # Negative pair
        else:
            text = positive_text
            label = 1  # Positive pair
        
        # Encode text
        text_encoded = self.text_tokenizer(
            text,
            padding='max_length',
            truncation=True,
            max_length=self.max_text_length,
            return_tensors='pt'
        )
        
        return {
            'image': image,
            'text_ids': text_encoded['input_ids'].squeeze(0),
            'text_mask': text_encoded['attention_mask'].squeeze(0),
            'label': torch.tensor(label, dtype=torch.float32)
        }


class MultimodalDataModule:
    """Data module for multimodal learning"""
    
    def __init__(
        self,
        dataset_name: str = 'multimodal',
        data_path: str = './data',
        modalities: List[str] = ['image', 'text'],
        batch_size: int = 32,
        num_workers: int = 4,
        train_split: float = 0.7,
        val_split: float = 0.15,
        test_split: float = 0.15
    ):
        self.dataset_name = dataset_name
        self.data_path = data_path
        self.modalities = modalities
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.train_split = train_split
        self.val_split = val_split
        self.test_split = test_split
        
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
    
    def setup(self):
        """Setup datasets"""
        if self.dataset_name == 'vqa':
            # VQA dataset
            self.train_dataset = VQADataset(
                self.data_path,
                split='train'
            )
            self.val_dataset = VQADataset(
                self.data_path,
                split='val'
            )
            self.test_dataset = VQADataset(
                self.data_path,
                split='test'
            )
        elif self.dataset_name == 'image_text':
            # Image-text matching dataset
            full_dataset = ImageTextDataset(self.data_path)
            
            # Split dataset
            n = len(full_dataset)
            train_size = int(n * self.train_split)
            val_size = int(n * self.val_split)
            test_size = n - train_size - val_size
            
            self.train_dataset, self.val_dataset, self.test_dataset = \
                torch.utils.data.random_split(
                    full_dataset,
                    [train_size, val_size, test_size],
                    generator=torch.Generator().manual_seed(42)
                )
        else:
            # Generic multimodal dataset
            full_dataset = MultimodalDataset(
                self.data_path,
                modalities=self.modalities
            )
            
            # Split dataset
            n = len(full_dataset)
            train_size = int(n * self.train_split)
            val_size = int(n * self.val_split)
            test_size = n - train_size - val_size
            
            self.train_dataset, self.val_dataset, self.test_dataset = \
                torch.utils.data.random_split(
                    full_dataset,
                    [train_size, val_size, test_size],
                    generator=torch.Generator().manual_seed(42)
                )
    
    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True
        )
    
    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True
        )
    
    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True
        )


if __name__ == "__main__":
    # Test data loading
    print("Testing multimodal data loading...")
    
    # Test multimodal dataset
    dataset = MultimodalDataset(
        data_path='./data',
        modalities=['image', 'text', 'audio']
    )
    
    print(f"Dataset size: {len(dataset)}")
    
    if len(dataset) > 0:
        sample = dataset[0]
        print(f"Sample keys: {sample.keys()}")
        if 'image' in sample:
            print(f"Image shape: {sample['image'].shape}")
        if 'text_ids' in sample:
            print(f"Text IDs shape: {sample['text_ids'].shape}")
        if 'audio' in sample:
            print(f"Audio shape: {sample['audio'].shape}")
    
    # Test data module
    dm = MultimodalDataModule(
        dataset_name='multimodal',
        batch_size=16,
        modalities=['image', 'text']
    )
    
    dm.setup()
    print(f"Train dataset size: {len(dm.train_dataset)}")
    print(f"Val dataset size: {len(dm.val_dataset)}")
    print(f"Test dataset size: {len(dm.test_dataset)}")