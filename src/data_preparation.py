import os
import json
from datasets import load_dataset
from typing import List, Dict, Tuple
import torch
from torch.utils.data import Dataset
import numpy as np
from tqdm import tqdm

def download_and_prepare_data(save_dir: str = "data") -> Tuple[str, Dict]:
    """
    Download and prepare the daily dialog dataset for training.
    
    Args:
        save_dir: Directory to save the processed data
        
    Returns:
        Tuple of processed data path and tokenizer vocabulary
    """
    # Create data directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)
    
    # Load the daily dialog dataset
    print("Downloading dataset...")
    dataset = load_dataset("daily_dialog")
    
    # Process the dialogues
    processed_conversations = []
    
    print("Processing conversations...")
    for item in tqdm(dataset['train']):
        dialogue = item['dialog']
        # Join utterances with special tokens
        conversation = " <|endoftext|> ".join(dialogue) + " <|endoftext|>"
        processed_conversations.append(conversation)
    
    # Create vocabulary from the text
    print("Creating vocabulary...")
    text = " ".join(processed_conversations)
    chars = sorted(list(set(text)))
    vocab_size = len(chars)
    
    # Create token mappings
    stoi = {ch: i for i, ch in enumerate(chars)}
    itos = {i: ch for i, ch in enumerate(chars)}
    
    # Save the processed data
    data_path = os.path.join(save_dir, "processed_conversations.txt")
    with open(data_path, "w", encoding="utf-8") as f:
        f.write("\n".join(processed_conversations))
    
    # Save the vocabulary
    vocab_path = os.path.join(save_dir, "vocab.json")
    with open(vocab_path, "w") as f:
        json.dump({"stoi": stoi, "itos": itos}, f)
    
    print(f"Processed {len(processed_conversations)} conversations")
    print(f"Vocabulary size: {vocab_size}")
    
    return data_path, {"stoi": stoi, "itos": itos}

class ConversationDataset(Dataset):
    def __init__(self, data_path: str, vocab: Dict, block_size: int):
        with open(data_path, 'r', encoding='utf-8') as f:
            text = f.read()
        
        # Tokenize the text
        self.stoi = vocab["stoi"]
        data = torch.tensor(self.encode(text), dtype=torch.long)
        
        # Create examples
        n = len(data)
        self.block_size = block_size
        self.data = data[:n - (n % block_size)]  # Trim to block size
    
    def encode(self, text: str) -> List[int]:
        return [self.stoi[c] for c in text]
    
    def __len__(self):
        return len(self.data) - self.block_size

    def __getitem__(self, idx):
        x = self.data[idx:idx + self.block_size]
        y = self.data[idx + 1:idx + self.block_size + 1]
        return x, y

if __name__ == "__main__":
    # Test the data preparation
    data_path, vocab = download_and_prepare_data()
    dataset = ConversationDataset(data_path, vocab, block_size=128)
    print(f"Dataset size: {len(dataset)}")
    x, y = dataset[0]
    print(f"Sample input shape: {x.shape}")
    print(f"Sample target shape: {y.shape}") 