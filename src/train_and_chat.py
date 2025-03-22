import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import os
from tqdm import tqdm
import json
import argparse
from data_preparation import download_and_prepare_data, ConversationDataset
from utils import set_seed, plot_training_progress, save_checkpoint, load_checkpoint

class Config:
    def __init__(self, vocab_size):
        self.batch_size = 64
        self.block_size = 256
        self.max_iters = 5000
        self.eval_interval = 100
        self.learning_rate = 3e-4
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.eval_iters = 200
        self.n_embd = 384
        self.n_head = 6
        self.n_layer = 6
        self.dropout = 0.2
        self.vocab_size = vocab_size

class ChatBot:
    def __init__(self, model, vocab, device, max_length=100):
        self.model = model
        self.vocab = vocab
        self.device = device
        self.max_length = max_length
    
    def encode(self, text):
        return [self.vocab["stoi"][c] for c in text]
    
    def decode(self, tokens):
        return ''.join([self.vocab["itos"][str(i.item())] for i in tokens])
    
    def generate_response(self, prompt, temperature=0.8):
        self.model.eval()
        
        # Encode the prompt
        encoded = torch.tensor(self.encode(prompt), dtype=torch.long, device=self.device).unsqueeze(0)
        
        # Generate response
        with torch.no_grad():
            for _ in range(self.max_length):
                # Take last block_size tokens
                if encoded.size(1) > self.model.block_size:
                    encoded = encoded[:, -self.model.block_size:]
                
                logits, _ = self.model(encoded)
                logits = logits[:, -1, :] / temperature
                probs = torch.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
                
                encoded = torch.cat([encoded, next_token], dim=1)
                
                # Stop if we generate end of text token
                if self.decode(next_token) == "<|endoftext|>":
                    break
        
        response = self.decode(encoded[0])
        # Clean up the response
        response = response.split("<|endoftext|>")[1].strip()
        return response

def train(config, train_dataset, val_dataset=None):
    from notebooks.01_training_simple_gpt import GPTLanguageModel
    
    # Create model
    model = GPTLanguageModel(config).to(config.device)
    print(f"Number of parameters: {sum(p.numel() for p in model.parameters())/1e6:.2f}M")
    
    # Create optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
    
    # Create data loader
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
    
    # Training loop
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    
    for epoch in range(config.max_iters):
        model.train()
        total_loss = 0
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}")
        
        for batch_idx, (x, y) in enumerate(progress_bar):
            x, y = x.to(config.device), y.to(config.device)
            
            # Forward pass
            logits, loss = model(x, y)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            total_loss += loss.item()
            progress_bar.set_postfix({'loss': loss.item()})
            
            # Evaluation
            if batch_idx % config.eval_interval == 0:
                model.eval()
                with torch.no_grad():
                    # Calculate validation loss
                    val_loss = 0
                    if val_dataset is not None:
                        val_loader = DataLoader(val_dataset, batch_size=config.batch_size)
                        for val_x, val_y in val_loader:
                            val_x, val_y = val_x.to(config.device), val_y.to(config.device)
                            _, loss = model(val_x, val_y)
                            val_loss += loss.item()
                        val_loss /= len(val_loader)
                        val_losses.append(val_loss)
                        
                        # Save best model
                        if val_loss < best_val_loss:
                            best_val_loss = val_loss
                            save_checkpoint(model, optimizer, epoch, val_loss, 'data/best_model.pt')
                
                train_losses.append(total_loss / (batch_idx + 1))
                model.train()
    
    # Plot training progress
    plot_training_progress(train_losses, val_losses, config.eval_interval)
    return model

def main():
    # Set random seed
    set_seed(42)
    
    # Prepare data
    data_path, vocab = download_and_prepare_data()
    
    # Create config
    config = Config(len(vocab["stoi"]))
    
    # Create datasets
    full_dataset = ConversationDataset(data_path, vocab, config.block_size)
    train_size = int(0.9 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(full_dataset, [train_size, val_size])
    
    # Train model
    model = train(config, train_dataset, val_dataset)
    
    # Create chatbot
    chatbot = ChatBot(model, vocab, config.device)
    
    # Interactive chat
    print("\nChatbot is ready! Type 'quit' to exit.")
    while True:
        user_input = input("\nYou: ")
        if user_input.lower() == 'quit':
            break
        
        response = chatbot.generate_response(user_input)
        print(f"Bot: {response}")

if __name__ == "__main__":
    main() 