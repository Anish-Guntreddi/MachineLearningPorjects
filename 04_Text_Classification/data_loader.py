"""
Data loading and preprocessing for text classification
"""
import torch
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
from transformers import AutoTokenizer
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from sklearn.model_selection import train_test_split
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import spacy

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')


class TextPreprocessor:
    """Text preprocessing utility"""
    
    def __init__(self, remove_stopwords: bool = False, use_spacy: bool = False):
        self.remove_stopwords = remove_stopwords
        self.use_spacy = use_spacy
        
        if remove_stopwords:
            self.stop_words = set(stopwords.words('english'))
        
        if use_spacy:
            try:
                self.nlp = spacy.load('en_core_web_sm')
            except:
                print("Spacy model not found. Install with: python -m spacy download en_core_web_sm")
                self.use_spacy = False
    
    def clean_text(self, text: str) -> str:
        """Clean and normalize text"""
        # Remove HTML tags
        text = re.sub(r'<.*?>', '', text)
        
        # Remove URLs
        text = re.sub(r'http\S+|www.\S+', '', text)
        
        # Remove email addresses
        text = re.sub(r'\S+@\S+', '', text)
        
        # Remove special characters and digits
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        
        # Remove extra whitespace
        text = ' '.join(text.split())
        
        # Lowercase
        text = text.lower()
        
        return text
    
    def tokenize(self, text: str) -> List[str]:
        """Tokenize text"""
        if self.use_spacy and hasattr(self, 'nlp'):
            doc = self.nlp(text)
            tokens = [token.lemma_ for token in doc 
                     if not token.is_stop and not token.is_punct]
        else:
            tokens = word_tokenize(text)
            if self.remove_stopwords:
                tokens = [t for t in tokens if t not in self.stop_words]
        
        return tokens
    
    def process(self, text: str) -> str:
        """Full preprocessing pipeline"""
        text = self.clean_text(text)
        if self.use_spacy or self.remove_stopwords:
            tokens = self.tokenize(text)
            text = ' '.join(tokens)
        return text


class TextDataset(Dataset):
    """Custom dataset for text classification"""
    
    def __init__(
        self,
        texts: List[str],
        labels: List[int],
        tokenizer,
        max_length: int = 512,
        preprocessing: Optional[TextPreprocessor] = None
    ):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.preprocessing = preprocessing
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        
        # Apply preprocessing if specified
        if self.preprocessing:
            text = self.preprocessing.process(text)
        
        # Tokenize
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'label': torch.tensor(label, dtype=torch.long)
        }


def load_imdb_dataset(
    tokenizer_name: str = 'bert-base-uncased',
    max_length: int = 512,
    batch_size: int = 32,
    num_workers: int = 4,
    preprocessing: bool = False
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Load IMDb movie review dataset
    
    Returns:
        train_loader, val_loader, test_loader
    """
    # Load dataset
    dataset = load_dataset('imdb')
    
    # Initialize tokenizer
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    
    # Initialize preprocessor if needed
    preprocessor = TextPreprocessor(remove_stopwords=False) if preprocessing else None
    
    # Prepare data
    train_texts = dataset['train']['text']
    train_labels = dataset['train']['label']
    
    test_texts = dataset['test']['text']
    test_labels = dataset['test']['label']
    
    # Split train into train and validation
    train_texts, val_texts, train_labels, val_labels = train_test_split(
        train_texts, train_labels, test_size=0.1, random_state=42, stratify=train_labels
    )
    
    # Create datasets
    train_dataset = TextDataset(
        train_texts, train_labels, tokenizer, max_length, preprocessor
    )
    
    val_dataset = TextDataset(
        val_texts, val_labels, tokenizer, max_length, preprocessor
    )
    
    test_dataset = TextDataset(
        test_texts, test_labels, tokenizer, max_length, preprocessor
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size * 2,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size * 2,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, val_loader, test_loader


def load_ag_news_dataset(
    tokenizer_name: str = 'bert-base-uncased',
    max_length: int = 256,
    batch_size: int = 32,
    num_workers: int = 4
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Load AG News dataset
    
    Returns:
        train_loader, val_loader, test_loader
    """
    # Load dataset
    dataset = load_dataset('ag_news')
    
    # Initialize tokenizer
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    
    # Prepare data
    train_texts = dataset['train']['text']
    train_labels = dataset['train']['label']
    
    test_texts = dataset['test']['text']
    test_labels = dataset['test']['label']
    
    # Split train into train and validation
    train_texts, val_texts, train_labels, val_labels = train_test_split(
        train_texts[:100000], train_labels[:100000],  # Use subset for faster training
        test_size=0.1, random_state=42, stratify=train_labels[:100000]
    )
    
    # Create datasets
    train_dataset = TextDataset(
        train_texts, train_labels, tokenizer, max_length
    )
    
    val_dataset = TextDataset(
        val_texts, val_labels, tokenizer, max_length
    )
    
    test_dataset = TextDataset(
        test_texts, test_labels, tokenizer, max_length
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size * 2,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size * 2,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, val_loader, test_loader


class TFIDFDataset:
    """Dataset for traditional ML models using TF-IDF"""
    
    def __init__(self, texts: List[str], labels: List[int], preprocessor: Optional[TextPreprocessor] = None):
        self.texts = texts
        self.labels = labels
        self.preprocessor = preprocessor or TextPreprocessor()
    
    def get_processed_data(self) -> Tuple[List[str], np.ndarray]:
        """Get preprocessed texts and labels"""
        processed_texts = [self.preprocessor.process(text) for text in self.texts]
        return processed_texts, np.array(self.labels)


def create_data_loaders(
    dataset_name: str = 'imdb',
    model_type: str = 'transformer',
    tokenizer_name: str = 'bert-base-uncased',
    max_length: int = 512,
    batch_size: int = 32,
    num_workers: int = 4
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create data loaders based on dataset and model type
    
    Args:
        dataset_name: 'imdb' or 'ag_news'
        model_type: 'transformer' or 'traditional'
        tokenizer_name: Name of the tokenizer
        max_length: Maximum sequence length
        batch_size: Batch size
        num_workers: Number of workers
    
    Returns:
        train_loader, val_loader, test_loader
    """
    if dataset_name == 'imdb':
        return load_imdb_dataset(
            tokenizer_name=tokenizer_name,
            max_length=max_length,
            batch_size=batch_size,
            num_workers=num_workers
        )
    elif dataset_name == 'ag_news':
        return load_ag_news_dataset(
            tokenizer_name=tokenizer_name,
            max_length=max_length,
            batch_size=batch_size,
            num_workers=num_workers
        )
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")


if __name__ == "__main__":
    # Test data loading
    print("Testing IMDb dataset loading...")
    train_loader, val_loader, test_loader = load_imdb_dataset(batch_size=4)
    
    batch = next(iter(train_loader))
    print(f"Batch keys: {batch.keys()}")
    print(f"Input IDs shape: {batch['input_ids'].shape}")
    print(f"Labels shape: {batch['label'].shape}")
    
    # Test preprocessing
    preprocessor = TextPreprocessor(remove_stopwords=True)
    sample_text = "This is a great movie! I loved it. Visit http://example.com for more."
    processed = preprocessor.process(sample_text)
    print(f"\nOriginal: {sample_text}")
    print(f"Processed: {processed}")