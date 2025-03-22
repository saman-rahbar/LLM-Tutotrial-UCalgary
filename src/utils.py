import torch
import numpy as np
from typing import List, Tuple, Dict, Optional
import matplotlib.pyplot as plt
import seaborn as sns

def set_seed(seed: int) -> None:
    """Set random seed for reproducibility."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def plot_attention_weights(attention_weights: torch.Tensor,
                         tokens: List[str],
                         layer: int,
                         head: int) -> None:
    """
    Plot attention weights for a specific layer and head.
    
    Args:
        attention_weights: Tensor of shape [n_layer, n_head, seq_len, seq_len]
        tokens: List of tokens corresponding to the sequence
        layer: Layer index to plot
        head: Head index to plot
    """
    plt.figure(figsize=(10, 10))
    sns.heatmap(
        attention_weights[layer, head].detach().cpu().numpy(),
        xticklabels=tokens,
        yticklabels=tokens,
        cmap='viridis'
    )
    plt.title(f'Attention Weights (Layer {layer}, Head {head})')
    plt.xlabel('Key Tokens')
    plt.ylabel('Query Tokens')
    plt.show()

def calculate_perplexity(model: torch.nn.Module,
                        data: torch.Tensor,
                        block_size: int,
                        device: str) -> float:
    """
    Calculate perplexity on given data.
    
    Args:
        model: The language model
        data: Input tensor of token indices
        block_size: Maximum sequence length
        device: Device to run calculation on
    
    Returns:
        float: Perplexity score
    """
    model.eval()
    with torch.no_grad():
        total_loss = 0
        total_tokens = 0
        for i in range(0, len(data) - block_size, block_size):
            x = data[i:i + block_size].unsqueeze(0).to(device)
            y = data[i + 1:i + block_size + 1].unsqueeze(0).to(device)
            logits, loss = model(x, y)
            total_loss += loss.item() * block_size
            total_tokens += block_size
    return torch.exp(torch.tensor(total_loss / total_tokens)).item()

def top_k_top_p_filtering(logits: torch.Tensor,
                         top_k: int = 0,
                         top_p: float = 0.0,
                         filter_value: float = -float('Inf')) -> torch.Tensor:
    """
    Filter a distribution of logits using top-k and/or nucleus (top-p) filtering.
    
    Args:
        logits: Logits distribution shape (batch size, vocabulary size)
        top_k: Keep only top k tokens with highest probability (top-k filtering)
        top_p: Keep the top tokens with cumulative probability >= top_p (nucleus filtering)
        filter_value: Value to assign to filtered tokens
    
    Returns:
        torch.Tensor: Filtered distribution
    """
    assert logits.dim() == 2
    top_k = min(top_k, logits.size(-1))  # Safety check
    if top_k > 0:
        # Remove all tokens with a probability less than the last token of the top-k
        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
        logits[indices_to_remove] = filter_value

    if top_p > 0.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)

        # Remove tokens with cumulative probability above the threshold
        sorted_indices_to_remove = cumulative_probs > top_p
        # Shift the indices to the right to keep also the first token above the threshold
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0

        indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
        logits[indices_to_remove] = filter_value
    return logits

def plot_training_progress(train_losses: List[float],
                         val_losses: List[float],
                         eval_interval: int) -> None:
    """
    Plot training and validation losses.
    
    Args:
        train_losses: List of training losses
        val_losses: List of validation losses
        eval_interval: Number of steps between evaluations
    """
    plt.figure(figsize=(10, 6))
    steps = np.arange(len(train_losses)) * eval_interval
    plt.plot(steps, train_losses, label='Train Loss')
    plt.plot(steps, val_losses, label='Validation Loss')
    plt.xlabel('Training Steps')
    plt.ylabel('Loss')
    plt.title('Training Progress')
    plt.legend()
    plt.grid(True)
    plt.show()

def save_checkpoint(model: torch.nn.Module,
                   optimizer: torch.optim.Optimizer,
                   epoch: int,
                   loss: float,
                   filename: str) -> None:
    """
    Save model checkpoint.
    
    Args:
        model: The model to save
        optimizer: The optimizer to save
        epoch: Current epoch number
        loss: Current loss value
        filename: Path to save the checkpoint
    """
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }, filename)

def load_checkpoint(model: torch.nn.Module,
                   optimizer: torch.optim.Optimizer,
                   filename: str) -> Tuple[int, float]:
    """
    Load model checkpoint.
    
    Args:
        model: The model to load weights into
        optimizer: The optimizer to load state into
        filename: Path to the checkpoint file
    
    Returns:
        Tuple[int, float]: Epoch number and loss value from the checkpoint
    """
    checkpoint = torch.load(filename)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    loss = checkpoint['loss']
    return epoch, loss 