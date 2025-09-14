"""
Data loading for text generation
"""
import torch
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
from transformers import AutoTokenizer, GPT2TokenizerFast
import numpy as np
from typing import Dict, List, Tuple, Optional
import os


class TextGenerationDataset(Dataset):
    """Dataset for text generation"""
    
    def __init__(
        self,
        texts: List[str],
        tokenizer,
        max_length: int = 512,
        stride: int = 256
    ):
        """
        Args:
            texts: List of text strings
            tokenizer: Tokenizer
            max_length: Maximum sequence length
            stride: Stride for creating overlapping sequences
        """
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.stride = stride
        
        # Tokenize all texts and create chunks
        self.examples = []
        
        for text in texts:
            # Tokenize
            tokenized = tokenizer(
                text,
                truncation=False,
                padding=False,
                return_tensors=None
            )
            
            input_ids = tokenized['input_ids']
            
            # Create overlapping chunks
            for i in range(0, len(input_ids) - max_length + 1, stride):
                chunk = input_ids[i:i + max_length]
                self.examples.append(chunk)
            
            # Add the last chunk if it's long enough
            if len(input_ids) > max_length:
                last_chunk = input_ids[-max_length:]
                if last_chunk not in self.examples:
                    self.examples.append(last_chunk)
    
    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, idx):
        input_ids = torch.tensor(self.examples[idx], dtype=torch.long)
        
        # For language modeling, labels are the same as inputs
        labels = input_ids.clone()
        
        return {
            'input_ids': input_ids,
            'labels': labels,
            'attention_mask': torch.ones_like(input_ids)
        }


class WikiTextDataset:
    """WikiText dataset for text generation"""
    
    @staticmethod
    def load_wikitext(
        version: str = 'wikitext-103-v1',
        tokenizer_name: str = 'gpt2',
        max_length: int = 512,
        stride: int = 256,
        batch_size: int = 8,
        num_workers: int = 4
    ) -> Tuple[DataLoader, DataLoader, DataLoader]:
        """
        Load WikiText dataset
        
        Args:
            version: WikiText version ('wikitext-2-v1' or 'wikitext-103-v1')
            tokenizer_name: Name of the tokenizer
            max_length: Maximum sequence length
            stride: Stride for overlapping sequences
            batch_size: Batch size
            num_workers: Number of workers
        
        Returns:
            train_loader, val_loader, test_loader
        """
        # Load dataset
        dataset = load_dataset('wikitext', version)
        
        # Initialize tokenizer
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        # Filter out empty texts
        train_texts = [text for text in dataset['train']['text'] if text.strip()]
        val_texts = [text for text in dataset['validation']['text'] if text.strip()]
        test_texts = [text for text in dataset['test']['text'] if text.strip()]
        
        # Create datasets
        train_dataset = TextGenerationDataset(
            train_texts, tokenizer, max_length, stride
        )
        
        val_dataset = TextGenerationDataset(
            val_texts, tokenizer, max_length, stride
        )
        
        test_dataset = TextGenerationDataset(
            test_texts, tokenizer, max_length, stride
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


class OpenWebTextDataset:
    """OpenWebText dataset for text generation"""
    
    @staticmethod
    def load_openwebtext(
        tokenizer_name: str = 'gpt2',
        max_length: int = 512,
        stride: int = 256,
        batch_size: int = 8,
        num_workers: int = 4,
        subset_size: Optional[int] = 10000
    ) -> Tuple[DataLoader, DataLoader, DataLoader]:
        """
        Load OpenWebText dataset
        
        Args:
            tokenizer_name: Name of the tokenizer
            max_length: Maximum sequence length
            stride: Stride for overlapping sequences
            batch_size: Batch size
            num_workers: Number of workers
            subset_size: Use only a subset for faster training
        
        Returns:
            train_loader, val_loader, test_loader
        """
        # Load dataset
        dataset = load_dataset('openwebtext', split='train')
        
        # Use subset if specified
        if subset_size:
            dataset = dataset.select(range(min(subset_size, len(dataset))))
        
        # Initialize tokenizer
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        # Get texts
        texts = dataset['text']
        
        # Split into train/val/test
        n = len(texts)
        train_size = int(0.8 * n)
        val_size = int(0.1 * n)
        
        train_texts = texts[:train_size]
        val_texts = texts[train_size:train_size + val_size]
        test_texts = texts[train_size + val_size:]
        
        # Create datasets
        train_dataset = TextGenerationDataset(
            train_texts, tokenizer, max_length, stride
        )
        
        val_dataset = TextGenerationDataset(
            val_texts, tokenizer, max_length, stride
        )
        
        test_dataset = TextGenerationDataset(
            test_texts, tokenizer, max_length, stride
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


def collate_fn(batch: List[Dict]) -> Dict:
    """Custom collate function for padding"""
    input_ids = [item['input_ids'] for item in batch]
    labels = [item['labels'] for item in batch]
    attention_mask = [item['attention_mask'] for item in batch]
    
    # Pad sequences
    max_len = max(len(ids) for ids in input_ids)
    
    padded_input_ids = []
    padded_labels = []
    padded_attention_mask = []
    
    for ids, labs, mask in zip(input_ids, labels, attention_mask):
        padding_len = max_len - len(ids)
        padded_input_ids.append(
            torch.cat([ids, torch.zeros(padding_len, dtype=torch.long)])
        )
        padded_labels.append(
            torch.cat([labs, torch.full((padding_len,), -100, dtype=torch.long)])
        )
        padded_attention_mask.append(
            torch.cat([mask, torch.zeros(padding_len, dtype=torch.long)])
        )
    
    return {
        'input_ids': torch.stack(padded_input_ids),
        'labels': torch.stack(padded_labels),
        'attention_mask': torch.stack(padded_attention_mask)
    }


def create_data_loaders(
    dataset_name: str = 'wikitext',
    tokenizer_name: str = 'gpt2',
    max_length: int = 512,
    stride: int = 256,
    batch_size: int = 8,
    num_workers: int = 4
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create data loaders for text generation
    
    Args:
        dataset_name: 'wikitext' or 'openwebtext'
        tokenizer_name: Name of the tokenizer
        max_length: Maximum sequence length
        stride: Stride for overlapping sequences
        batch_size: Batch size
        num_workers: Number of workers
    
    Returns:
        train_loader, val_loader, test_loader
    """
    if dataset_name == 'wikitext':
        return WikiTextDataset.load_wikitext(
            version='wikitext-103-v1',
            tokenizer_name=tokenizer_name,
            max_length=max_length,
            stride=stride,
            batch_size=batch_size,
            num_workers=num_workers
        )
    elif dataset_name == 'openwebtext':
        return OpenWebTextDataset.load_openwebtext(
            tokenizer_name=tokenizer_name,
            max_length=max_length,
            stride=stride,
            batch_size=batch_size,
            num_workers=num_workers,
            subset_size=10000  # Use subset for faster training
        )
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")


class TextDataCollator:
    """Data collator for language modeling"""
    
    def __init__(self, tokenizer, mlm: bool = False, mlm_probability: float = 0.15):
        self.tokenizer = tokenizer
        self.mlm = mlm
        self.mlm_probability = mlm_probability
    
    def __call__(self, examples: List[Dict]) -> Dict:
        # Pad sequences
        batch = self.tokenizer.pad(
            examples,
            padding=True,
            return_tensors='pt'
        )
        
        # Prepare labels
        if self.mlm:
            batch['input_ids'], batch['labels'] = self.mask_tokens(
                batch['input_ids']
            )
        else:
            batch['labels'] = batch['input_ids'].clone()
        
        return batch
    
    def mask_tokens(
        self,
        inputs: torch.Tensor,
        special_tokens_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Prepare masked tokens for masked language modeling
        """
        labels = inputs.clone()
        
        # Create probability matrix
        probability_matrix = torch.full(labels.shape, self.mlm_probability)
        
        if special_tokens_mask is not None:
            probability_matrix.masked_fill_(special_tokens_mask, value=0.0)
        
        masked_indices = torch.bernoulli(probability_matrix).bool()
        labels[~masked_indices] = -100  # Only compute loss on masked tokens
        
        # 80% of the time, replace with [MASK]
        indices_replaced = torch.bernoulli(torch.full(labels.shape, 0.8)).bool() & masked_indices
        inputs[indices_replaced] = self.tokenizer.mask_token_id
        
        # 10% of the time, replace with random word
        indices_random = torch.bernoulli(torch.full(labels.shape, 0.5)).bool() & masked_indices & ~indices_replaced
        random_words = torch.randint(len(self.tokenizer), labels.shape, dtype=torch.long)
        inputs[indices_random] = random_words[indices_random]
        
        return inputs, labels


if __name__ == "__main__":
    # Test data loading
    print("Testing WikiText dataset loading...")
    train_loader, val_loader, test_loader = WikiTextDataset.load_wikitext(
        version='wikitext-2-v1',  # Use smaller version for testing
        batch_size=2
    )
    
    batch = next(iter(train_loader))
    print(f"Batch keys: {batch.keys()}")
    print(f"Input IDs shape: {batch['input_ids'].shape}")
    print(f"Labels shape: {batch['labels'].shape}")
    
    # Test tokenizer
    tokenizer = GPT2TokenizerFast.from_pretrained('gpt2')
    sample_text = "This is a sample text for generation."
    tokens = tokenizer(sample_text, return_tensors='pt')
    print(f"\nTokenized sample: {tokens['input_ids'].shape}")
    
    # Decode back
    decoded = tokenizer.decode(tokens['input_ids'][0])
    print(f"Decoded: {decoded}")