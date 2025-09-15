"""
Data loading and preprocessing for Automatic Speech Recognition
"""
import torch
from torch.utils.data import Dataset, DataLoader
import torchaudio
import torchaudio.transforms as T
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import json
from transformers import Wav2Vec2Processor, WhisperProcessor
import librosa


class ASRDataset(Dataset):
    """Dataset for Automatic Speech Recognition"""
    
    def __init__(
        self,
        data_path: str,
        manifest_file: Optional[str] = None,
        sr: int = 16000,
        max_duration: float = 20.0,
        min_duration: float = 0.5,
        augment: bool = False,
        processor = None,
        return_attention_mask: bool = True
    ):
        """
        Args:
            data_path: Path to audio files
            manifest_file: JSON manifest with audio paths and transcripts
            sr: Sample rate
            max_duration: Maximum audio duration in seconds
            min_duration: Minimum audio duration in seconds
            augment: Whether to apply augmentation
            processor: HuggingFace processor for tokenization
            return_attention_mask: Whether to return attention mask
        """
        self.data_path = Path(data_path)
        self.sr = sr
        self.max_duration = max_duration
        self.min_duration = min_duration
        self.augment = augment
        self.processor = processor
        self.return_attention_mask = return_attention_mask
        
        # Load manifest
        self.samples = []
        if manifest_file and Path(manifest_file).exists():
            self._load_manifest(manifest_file)
        else:
            self._scan_directory()
        
        # Filter by duration
        self.samples = self._filter_by_duration()
        
        # Initialize transforms
        self._init_transforms()
        
        print(f"Loaded {len(self.samples)} samples")
    
    def _load_manifest(self, manifest_file: str):
        """Load manifest file"""
        with open(manifest_file, 'r') as f:
            if manifest_file.endswith('.json'):
                data = json.load(f)
                for item in data:
                    self.samples.append({
                        'audio_path': str(self.data_path / item['audio_path']),
                        'transcript': item['transcript'],
                        'duration': item.get('duration', None)
                    })
            elif manifest_file.endswith('.csv'):
                df = pd.read_csv(manifest_file)
                for _, row in df.iterrows():
                    self.samples.append({
                        'audio_path': str(self.data_path / row['audio_path']),
                        'transcript': row['transcript'],
                        'duration': row.get('duration', None)
                    })
    
    def _scan_directory(self):
        """Scan directory for audio files and transcripts"""
        # Look for paired audio and text files
        for audio_file in self.data_path.glob('**/*.wav'):
            txt_file = audio_file.with_suffix('.txt')
            if txt_file.exists():
                with open(txt_file, 'r') as f:
                    transcript = f.read().strip()
                
                self.samples.append({
                    'audio_path': str(audio_file),
                    'transcript': transcript,
                    'duration': None
                })
    
    def _filter_by_duration(self) -> List[Dict]:
        """Filter samples by duration"""
        filtered_samples = []
        
        for sample in self.samples:
            if sample['duration'] is None:
                # Load audio to get duration
                try:
                    info = torchaudio.info(sample['audio_path'])
                    duration = info.num_frames / info.sample_rate
                    sample['duration'] = duration
                except:
                    continue
            
            duration = sample['duration']
            if self.min_duration <= duration <= self.max_duration:
                filtered_samples.append(sample)
        
        return filtered_samples
    
    def _init_transforms(self):
        """Initialize audio transforms"""
        self.resample = T.Resample(orig_freq=self.sr, new_freq=self.sr)
        
        if self.augment:
            self.time_mask = T.TimeMasking(time_mask_param=10)
            self.freq_mask = T.FrequencyMasking(freq_mask_param=10)
    
    def _add_noise(self, waveform: torch.Tensor, noise_level: float = 0.005) -> torch.Tensor:
        """Add white noise"""
        noise = torch.randn_like(waveform) * noise_level
        return waveform + noise
    
    def _speed_perturb(self, waveform: torch.Tensor, factor: float = None) -> torch.Tensor:
        """Apply speed perturbation"""
        if factor is None:
            factor = np.random.uniform(0.9, 1.1)
        
        # Use torchaudio's Resample for speed perturbation
        orig_freq = self.sr
        new_freq = int(self.sr * factor)
        
        resampler = T.Resample(orig_freq=orig_freq, new_freq=new_freq)
        perturbed = resampler(waveform)
        
        # Resample back to original sample rate
        resampler_back = T.Resample(orig_freq=new_freq, new_freq=orig_freq)
        waveform = resampler_back(perturbed)
        
        return waveform
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        # Load audio
        waveform, sr = torchaudio.load(sample['audio_path'])
        
        # Resample if necessary
        if sr != self.sr:
            waveform = self.resample(waveform)
        
        # Convert to mono
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)
        
        # Apply augmentation
        if self.augment:
            if np.random.random() > 0.5:
                waveform = self._add_noise(waveform)
            if np.random.random() > 0.5:
                waveform = self._speed_perturb(waveform)
        
        # Squeeze to 1D
        waveform = waveform.squeeze(0)
        
        # Process with HuggingFace processor if provided
        if self.processor:
            # Process audio
            inputs = self.processor(
                waveform.numpy(),
                sampling_rate=self.sr,
                return_tensors="pt",
                padding=True
            )
            
            # Process text
            with self.processor.as_target_processor():
                labels = self.processor(
                    sample['transcript'],
                    return_tensors="pt",
                    padding=True
                )
            
            return {
                'input_values': inputs.input_values.squeeze(0),
                'attention_mask': inputs.attention_mask.squeeze(0) if self.return_attention_mask else None,
                'labels': labels.input_ids.squeeze(0),
                'transcript': sample['transcript']
            }
        else:
            return {
                'waveform': waveform,
                'transcript': sample['transcript'],
                'audio_path': sample['audio_path'],
                'duration': sample['duration']
            }


class LibriSpeechDataset(ASRDataset):
    """LibriSpeech dataset loader"""
    
    def __init__(self, split: str = 'train-clean-100', **kwargs):
        """
        Args:
            split: LibriSpeech split to use
            **kwargs: Additional arguments for ASRDataset
        """
        # Download and prepare LibriSpeech
        from torchaudio.datasets import LIBRISPEECH
        
        self.split = split
        root_path = kwargs.pop('data_path', './data')
        
        # Load LibriSpeech dataset
        self.dataset = LIBRISPEECH(
            root=root_path,
            url=split,
            download=True
        )
        
        # Convert to our format
        samples = []
        for i in range(len(self.dataset)):
            waveform, sample_rate, transcript, speaker_id, chapter_id, utterance_id = self.dataset[i]
            
            # Save temporary audio file
            audio_path = f"/tmp/librispeech_{i}.wav"
            torchaudio.save(audio_path, waveform, sample_rate)
            
            samples.append({
                'audio_path': audio_path,
                'transcript': transcript.lower(),
                'duration': waveform.shape[1] / sample_rate,
                'speaker_id': speaker_id,
                'chapter_id': chapter_id
            })
        
        # Initialize parent class
        super().__init__(data_path=root_path, **kwargs)
        self.samples = samples


class CommonVoiceDataset(ASRDataset):
    """Common Voice dataset loader"""
    
    def __init__(self, csv_file: str, audio_dir: str, **kwargs):
        """
        Args:
            csv_file: Path to Common Voice CSV file
            audio_dir: Directory containing audio clips
            **kwargs: Additional arguments for ASRDataset
        """
        # Load CSV
        df = pd.read_csv(csv_file, sep='\t')
        
        # Filter valid samples
        df = df[df['sentence'].notna()]
        
        # Create samples
        samples = []
        audio_dir = Path(audio_dir)
        
        for _, row in df.iterrows():
            audio_path = audio_dir / row['path']
            if audio_path.exists():
                samples.append({
                    'audio_path': str(audio_path),
                    'transcript': row['sentence'].lower(),
                    'duration': row.get('duration', None)
                })
        
        # Initialize parent class
        super().__init__(data_path=audio_dir, **kwargs)
        self.samples = samples


class ASRDataModule:
    """Data module for ASR"""
    
    def __init__(
        self,
        dataset_name: str = 'librispeech',
        data_path: str = './data',
        batch_size: int = 8,
        num_workers: int = 4,
        processor = None,
        train_split: float = 0.8,
        val_split: float = 0.1,
        test_split: float = 0.1
    ):
        self.dataset_name = dataset_name
        self.data_path = data_path
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.processor = processor
        self.train_split = train_split
        self.val_split = val_split
        self.test_split = test_split
        
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
    
    def setup(self):
        """Setup datasets"""
        if self.dataset_name == 'librispeech':
            # Use different splits for train/val/test
            self.train_dataset = LibriSpeechDataset(
                split='train-clean-100',
                data_path=self.data_path,
                processor=self.processor,
                augment=True
            )
            
            self.val_dataset = LibriSpeechDataset(
                split='dev-clean',
                data_path=self.data_path,
                processor=self.processor,
                augment=False
            )
            
            self.test_dataset = LibriSpeechDataset(
                split='test-clean',
                data_path=self.data_path,
                processor=self.processor,
                augment=False
            )
        else:
            # Generic dataset - split manually
            full_dataset = ASRDataset(
                data_path=self.data_path,
                processor=self.processor,
                augment=False
            )
            
            # Split dataset
            total_size = len(full_dataset)
            train_size = int(self.train_split * total_size)
            val_size = int(self.val_split * total_size)
            test_size = total_size - train_size - val_size
            
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
            collate_fn=self.collate_fn,
            pin_memory=True
        )
    
    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=self.collate_fn,
            pin_memory=True
        )
    
    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=self.collate_fn,
            pin_memory=True
        )
    
    def collate_fn(self, batch: List[Dict]) -> Dict:
        """Custom collate function for variable length sequences"""
        if self.processor:
            # Batch processing with processor
            input_values = [item['input_values'] for item in batch]
            labels = [item['labels'] for item in batch]
            transcripts = [item['transcript'] for item in batch]
            
            # Pad input values
            input_values = torch.nn.utils.rnn.pad_sequence(
                input_values,
                batch_first=True,
                padding_value=0.0
            )
            
            # Pad labels
            labels = torch.nn.utils.rnn.pad_sequence(
                labels,
                batch_first=True,
                padding_value=-100
            )
            
            batch_dict = {
                'input_values': input_values,
                'labels': labels,
                'transcripts': transcripts
            }
            
            if batch[0]['attention_mask'] is not None:
                attention_mask = [item['attention_mask'] for item in batch]
                attention_mask = torch.nn.utils.rnn.pad_sequence(
                    attention_mask,
                    batch_first=True,
                    padding_value=0
                )
                batch_dict['attention_mask'] = attention_mask
            
            return batch_dict
        else:
            # Simple batching
            waveforms = [item['waveform'] for item in batch]
            transcripts = [item['transcript'] for item in batch]
            
            # Pad waveforms
            waveforms = torch.nn.utils.rnn.pad_sequence(
                waveforms,
                batch_first=True,
                padding_value=0.0
            )
            
            return {
                'waveforms': waveforms,
                'transcripts': transcripts
            }


def create_char_tokenizer(transcripts: List[str]) -> Dict:
    """Create character-level tokenizer"""
    # Get all unique characters
    chars = set()
    for transcript in transcripts:
        chars.update(transcript.lower())
    
    # Sort characters
    chars = sorted(list(chars))
    
    # Add special tokens
    vocab = ['<pad>', '<sos>', '<eos>'] + chars
    
    # Create mappings
    char_to_idx = {char: idx for idx, char in enumerate(vocab)}
    idx_to_char = {idx: char for char, idx in char_to_idx.items()}
    
    return {
        'vocab': vocab,
        'char_to_idx': char_to_idx,
        'idx_to_char': idx_to_char,
        'vocab_size': len(vocab)
    }


def tokenize_transcript(
    transcript: str,
    tokenizer: Dict,
    max_length: Optional[int] = None
) -> torch.Tensor:
    """Tokenize transcript using character tokenizer"""
    char_to_idx = tokenizer['char_to_idx']
    
    # Convert to lowercase
    transcript = transcript.lower()
    
    # Tokenize
    tokens = [char_to_idx.get(char, char_to_idx['<pad>']) for char in transcript]
    
    # Add SOS and EOS
    tokens = [char_to_idx['<sos>']] + tokens + [char_to_idx['<eos>']]
    
    # Pad or truncate
    if max_length:
        if len(tokens) < max_length:
            tokens += [char_to_idx['<pad>']] * (max_length - len(tokens))
        else:
            tokens = tokens[:max_length]
    
    return torch.tensor(tokens, dtype=torch.long)


def decode_predictions(
    predictions: torch.Tensor,
    tokenizer: Dict
) -> List[str]:
    """Decode model predictions to text"""
    idx_to_char = tokenizer['idx_to_char']
    decoded = []
    
    for pred in predictions:
        chars = []
        for idx in pred:
            if idx == tokenizer['char_to_idx']['<eos>']:
                break
            if idx not in [tokenizer['char_to_idx']['<pad>'], 
                          tokenizer['char_to_idx']['<sos>']]:
                chars.append(idx_to_char.get(idx.item(), ''))
        
        decoded.append(''.join(chars))
    
    return decoded


if __name__ == "__main__":
    # Test data loading
    print("Testing ASR dataset...")
    
    # Test basic dataset
    dataset = ASRDataset(
        data_path="./data/asr",
        sr=16000,
        augment=True
    )
    
    if len(dataset) > 0:
        sample = dataset[0]
        print(f"Waveform shape: {sample['waveform'].shape}")
        print(f"Transcript: {sample['transcript']}")
    
    # Test tokenizer
    transcripts = [
        "hello world",
        "automatic speech recognition",
        "deep learning"
    ]
    
    tokenizer = create_char_tokenizer(transcripts)
    print(f"Vocabulary size: {tokenizer['vocab_size']}")
    print(f"Vocabulary: {tokenizer['vocab'][:20]}...")
    
    # Test tokenization
    tokens = tokenize_transcript("hello", tokenizer)
    print(f"Tokenized 'hello': {tokens}")
    
    # Test decoding
    decoded = decode_predictions(tokens.unsqueeze(0), tokenizer)
    print(f"Decoded: {decoded}")