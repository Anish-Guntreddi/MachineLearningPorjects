"""
Data loading for machine translation
"""
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer
import numpy as np
from typing import Dict, List, Tuple, Optional


class TranslationDataset(Dataset):
    """Dataset for machine translation"""

    def __init__(
        self,
        src_texts: List[str],
        tgt_texts: List[str],
        src_tokenizer,
        tgt_tokenizer,
        max_length: int = 128
    ):
        self.src_texts = src_texts
        self.tgt_texts = tgt_texts
        self.src_tokenizer = src_tokenizer
        self.tgt_tokenizer = tgt_tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.src_texts)

    def __getitem__(self, idx):
        src_text = self.src_texts[idx]
        tgt_text = self.tgt_texts[idx]

        # Tokenize source
        src_encoded = self.src_tokenizer(
            src_text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        # Tokenize target
        tgt_encoded = self.tgt_tokenizer(
            tgt_text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        return {
            'src_input_ids': src_encoded['input_ids'].squeeze(0),
            'src_attention_mask': src_encoded['attention_mask'].squeeze(0),
            'tgt_input_ids': tgt_encoded['input_ids'].squeeze(0),
            'tgt_attention_mask': tgt_encoded['attention_mask'].squeeze(0),
        }


class SyntheticTranslationDataset(Dataset):
    """Synthetic translation dataset for demonstration"""

    def __init__(
        self,
        num_samples: int = 1000,
        src_vocab_size: int = 32000,
        tgt_vocab_size: int = 32000,
        max_src_len: int = 30,
        max_tgt_len: int = 30
    ):
        self.num_samples = num_samples
        self.src_vocab_size = src_vocab_size
        self.tgt_vocab_size = tgt_vocab_size
        self.max_src_len = max_src_len
        self.max_tgt_len = max_tgt_len

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        src_len = np.random.randint(5, self.max_src_len)
        tgt_len = np.random.randint(5, self.max_tgt_len)

        src = torch.randint(1, self.src_vocab_size, (self.max_src_len,))
        src[src_len:] = 0  # Pad
        src_mask = torch.zeros(self.max_src_len, dtype=torch.long)
        src_mask[:src_len] = 1

        tgt = torch.randint(1, self.tgt_vocab_size, (self.max_tgt_len,))
        tgt[tgt_len:] = 0  # Pad
        tgt_mask = torch.zeros(self.max_tgt_len, dtype=torch.long)
        tgt_mask[:tgt_len] = 1

        return {
            'src_input_ids': src,
            'src_attention_mask': src_mask,
            'tgt_input_ids': tgt,
            'tgt_attention_mask': tgt_mask,
        }


def load_multi30k(
    tokenizer_name: str = 'Helsinki-NLP/opus-mt-en-de',
    max_length: int = 128,
    batch_size: int = 32,
    num_workers: int = 4
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Load Multi30k translation dataset via HuggingFace

    Args:
        tokenizer_name: Tokenizer to use
        max_length: Maximum sequence length
        batch_size: Batch size
        num_workers: Number of data loading workers

    Returns:
        train_loader, val_loader, test_loader
    """
    try:
        from datasets import load_dataset

        dataset = load_dataset('bentrevett/multi30k')

        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        def make_loader(split_data, shuffle=False):
            src_texts = [item['en'] for item in split_data]
            tgt_texts = [item['de'] for item in split_data]

            ds = TranslationDataset(
                src_texts, tgt_texts,
                tokenizer, tokenizer,
                max_length=max_length
            )
            return DataLoader(
                ds,
                batch_size=batch_size,
                shuffle=shuffle,
                num_workers=num_workers,
                pin_memory=True
            )

        train_loader = make_loader(dataset['train'], shuffle=True)
        val_loader = make_loader(dataset['validation'])
        test_loader = make_loader(dataset['test'])

        return train_loader, val_loader, test_loader

    except Exception as e:
        print(f"Could not load Multi30k: {e}")
        print("Falling back to synthetic data")
        return create_data_loaders(
            dataset_name='synthetic',
            batch_size=batch_size,
            num_workers=num_workers
        )


def create_data_loaders(
    dataset_name: str = 'synthetic',
    src_vocab_size: int = 32000,
    tgt_vocab_size: int = 32000,
    max_length: int = 30,
    batch_size: int = 32,
    num_workers: int = 4
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create data loaders for translation

    Args:
        dataset_name: 'synthetic' or 'multi30k'
        src_vocab_size: Source vocabulary size
        tgt_vocab_size: Target vocabulary size
        max_length: Maximum sequence length
        batch_size: Batch size
        num_workers: Number of data loading workers

    Returns:
        train_loader, val_loader, test_loader
    """
    if dataset_name == 'multi30k':
        return load_multi30k(
            max_length=max_length,
            batch_size=batch_size,
            num_workers=num_workers
        )

    # Synthetic data
    train_dataset = SyntheticTranslationDataset(
        num_samples=2000, src_vocab_size=src_vocab_size,
        tgt_vocab_size=tgt_vocab_size, max_src_len=max_length,
        max_tgt_len=max_length
    )
    val_dataset = SyntheticTranslationDataset(
        num_samples=400, src_vocab_size=src_vocab_size,
        tgt_vocab_size=tgt_vocab_size, max_src_len=max_length,
        max_tgt_len=max_length
    )
    test_dataset = SyntheticTranslationDataset(
        num_samples=400, src_vocab_size=src_vocab_size,
        tgt_vocab_size=tgt_vocab_size, max_src_len=max_length,
        max_tgt_len=max_length
    )

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size * 2, shuffle=False,
        num_workers=num_workers, pin_memory=True
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size * 2, shuffle=False,
        num_workers=num_workers, pin_memory=True
    )

    return train_loader, val_loader, test_loader


if __name__ == "__main__":
    # Test synthetic data loading
    train_loader, val_loader, test_loader = create_data_loaders(
        dataset_name='synthetic', batch_size=4, num_workers=0
    )

    batch = next(iter(train_loader))
    print(f"Batch keys: {batch.keys()}")
    print(f"Source input IDs shape: {batch['src_input_ids'].shape}")
    print(f"Target input IDs shape: {batch['tgt_input_ids'].shape}")
