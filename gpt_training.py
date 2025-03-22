import torch
import torch.nn as nn
from torch.nn import functional as F
import numpy as np
from datasets import load_dataset
from transformers import GPT2Tokenizer, GPT2Config, GPT2LMHeadModel
import matplotlib.pyplot as plt
from tqdm.auto import tqdm

class Config:
    # Model architecture parameters
    n_layer = 6          # Number of transformer layers
    n_head = 8           # Number of attention heads
    n_embd = 512        # Embedding dimension
    block_size = 128     # Maximum sequence length
    dropout = 0.1        # Dropout rate
    
    # Training parameters
    batch_size = 32
    learning_rate = 3e-4
    max_iters = 5000
    eval_interval = 500   # How often to evaluate on validation set
    eval_iters = 200     # How many batches to evaluate on
    device = "cuda" if torch.cuda.is_available() else "cpu"

def load_and_preprocess_data(config):
    """Load and preprocess the Daily Dialog dataset using HuggingFace.
    
    Args:
        config: Configuration object containing model parameters
        
    Returns:
        tuple: (train_data, val_data, tokenizer)
    """
    # Load the dataset
    print("Loading dataset...")
    dataset = load_dataset("daily_dialog")
    
    # Initialize tokenizer
    print("Initializing tokenizer...")
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    def tokenize_function(examples):
        """Tokenize a batch of examples."""
        # Join all dialogues with spaces and add EOS token
        texts = [" ".join(d) + tokenizer.eos_token for d in examples["dialog"]]
        return tokenizer(
            texts,
            padding="max_length",
            truncation=True,
            max_length=config.block_size,
            return_tensors="pt"
        )
    
    # Process training data
    print("Processing training data...")
    train_data = dataset["train"].map(
        tokenize_function,
        batched=True,
        batch_size=100,
        remove_columns=dataset["train"].column_names,
        num_proc=4
    )
    
    # Process validation data
    print("Processing validation data...")
    val_data = dataset["validation"].map(
        tokenize_function,
        batched=True,
        batch_size=100,
        remove_columns=dataset["validation"].column_names,
        num_proc=4
    )
    
    # Convert to PyTorch datasets
    train_data.set_format(type="torch", columns=["input_ids", "attention_mask"])
    val_data.set_format(type="torch", columns=["input_ids", "attention_mask"])
    
    return train_data, val_data, tokenizer

def get_batch(split, train_data, val_data, config):
    """Get a batch of data for training or validation.
    
    Args:
        split (str): Either 'train' or 'val'
        train_data: Training dataset
        val_data: Validation dataset
        config: Configuration object
        
    Returns:
        tuple: (input_ids, target_ids)
    """
    data = train_data if split == 'train' else val_data
    
    # Generate random indices
    ix = torch.randint(len(data), (config.batch_size,))
    
    # Get the sequences
    batch = data[ix]
    x = batch["input_ids"]
    
    # Create input-target pairs by shifting
    # Input: all tokens except last
    # Target: all tokens except first
    x = x[:, :-1]
    y = batch["input_ids"][:, 1:]
    
    # Move to appropriate device
    x, y = x.to(config.device), y.to(config.device)
    
    return x, y

class GPT(nn.Module):
    """GPT Language Model using HuggingFace's GPT2"""
    
    def __init__(self, config):
        super().__init__()
        self.config = GPT2Config(
            n_layer=config.n_layer,
            n_head=config.n_head,
            n_embd=config.n_embd,
            vocab_size=50257,  # GPT-2 vocabulary size
            n_positions=config.block_size,
            n_ctx=config.block_size,
            dropout=config.dropout
        )
        self.transformer = GPT2LMHeadModel(self.config)
        self.block_size = config.block_size

    def forward(self, idx, targets=None):
        # Forward pass through GPT-2
        outputs = self.transformer(
            idx,
            labels=targets if targets is not None else None,
            return_dict=True
        )
        
        if targets is not None:
            # Return logits and loss if we have targets
            return outputs.logits, outputs.loss
        else:
            # Return only logits if we don't have targets
            return outputs.logits, None

@torch.no_grad()
def estimate_loss(model, train_data, val_data, config):
    """Estimate loss on train and validation sets"""
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(config.eval_iters)
        for k in range(config.eval_iters):
            X, Y = get_batch(split, train_data, val_data, config)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

@torch.no_grad()
def generate(model, tokenizer, prompt, max_tokens=50, temperature=1.0, top_k=None, config=None):
    """Generate text from a prompt."""
    # Encode the prompt
    encoded = tokenizer.encode(prompt, return_tensors='pt').to(config.device)
    
    # Generate using HuggingFace's built-in generation
    output_sequences = model.transformer.generate(
        encoded,
        max_length=len(encoded[0]) + max_tokens,
        temperature=temperature,
        top_k=top_k if top_k is not None else 50,
        top_p=0.9,
        do_sample=True,
        pad_token_id=tokenizer.pad_token_id,
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id
    )
    
    # Decode the generated text
    generated_text = tokenizer.decode(output_sequences[0], skip_special_tokens=True)
    return generated_text

def main():
    # Initialize configuration
    config = Config()
    
    # Load and preprocess data
    train_data, val_data, tokenizer = load_and_preprocess_data(config)
    
    # Initialize model
    model = GPT(config).to(config.device)
    
    # Create optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
    
    # Training loop
    losses = []
    for iter in tqdm(range(config.max_iters)):
        # Evaluate the model every `eval_interval` iterations
        if iter % config.eval_interval == 0:
            losses_dict = estimate_loss(model, train_data, val_data, config)
            print(f"step {iter}: train loss {losses_dict['train']:.4f}, val loss {losses_dict['val']:.4f}")
            losses.append(losses_dict['train'])

        # Sample a batch of data
        xb, yb = get_batch('train', train_data, val_data, config)
        
        # Forward and backward passes
        logits, loss = model(xb, yb)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

    # Plot the training loss
    plt.figure(figsize=(10, 5))
    plt.plot(losses)
    plt.title('Training Loss')
    plt.xlabel('Evaluation Step')
    plt.ylabel('Loss')
    plt.show()
    
    # Try generating some text
    prompt = "Hello, how are you?"
    generated_text = generate(model, tokenizer, prompt, temperature=0.8, top_k=40, config=config)
    print(f"Prompt: {prompt}")
    print(f"Generated: {generated_text}")

if __name__ == "__main__":
    main() 