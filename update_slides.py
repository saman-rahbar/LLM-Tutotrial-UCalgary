import nbformat as nbf

# Create the technical deep dive notebook
nb = nbf.v4.new_notebook()

# Add cells
cells = [
    nbf.v4.new_markdown_cell('''# Technical Deep Dive: LLM Architecture
    
A detailed look at the inner workings of Large Language Models

University of Calgary - Graduate Tutorial'''),

    nbf.v4.new_markdown_cell('''## Transformer Architecture

1. Key Components:
   - Self-attention layers
   - Feed-forward networks
   - Layer normalization
   - Residual connections

2. Attention Mechanism:
   - Query, Key, Value matrices
   - Scaled dot-product attention
   - Multi-head attention'''),

    nbf.v4.new_markdown_cell('''## Self-Attention Mechanism

The core operation in transformers:

1. Input embeddings are transformed into Q, K, V
2. Attention scores = softmax(QK^T / √d)
3. Output = Attention scores × V

Benefits:
- Captures long-range dependencies
- Parallel processing
- Dynamic context understanding'''),

    nbf.v4.new_code_cell('''import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np

def visualize_attention(query, key, value):
    # Compute attention scores
    scores = torch.matmul(query, key.transpose(-2, -1)) / np.sqrt(key.size(-1))
    attention = F.softmax(scores, dim=-1)
    
    # Plot attention matrix
    plt.figure(figsize=(8, 6))
    plt.imshow(attention.detach().numpy(), cmap='viridis')
    plt.colorbar()
    plt.title('Attention Weights')
    plt.xlabel('Key/Value positions')
    plt.ylabel('Query positions')
    plt.show()
    
    return torch.matmul(attention, value)

# Example usage
seq_len = 10
d_model = 64
query = torch.randn(1, seq_len, d_model)
key = torch.randn(1, seq_len, d_model)
value = torch.randn(1, seq_len, d_model)

output = visualize_attention(query[0], key[0], value[0])'''),

    nbf.v4.new_markdown_cell('''## Position-wise Feed-Forward Networks

After attention, each position goes through:

1. Two linear transformations
2. GELU activation
3. Dropout for regularization

```python
class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model)
        )
```'''),

    nbf.v4.new_markdown_cell('''## Training Process

1. **Pre-training**
   - Masked language modeling
   - Next token prediction
   - Large-scale text corpora

2. **Fine-tuning**
   - Task-specific adaptation
   - Smaller learning rates
   - Limited data needed

3. **Evaluation**
   - Perplexity
   - Task-specific metrics
   - Human evaluation'''),

    nbf.v4.new_markdown_cell('''## Scaling Laws

Key findings about model scaling:

1. Performance improves predictably with:
   - Model size (parameters)
   - Dataset size
   - Compute budget

2. Empirical relationships:
   - Loss ~ 1/sqrt(parameters)
   - Loss ~ 1/sqrt(compute)
   - Loss ~ 1/sqrt(dataset size)'''),

    nbf.v4.new_markdown_cell('''## Optimization Techniques

1. **Memory Efficiency**
   - Gradient checkpointing
   - Mixed precision training
   - Model parallelism

2. **Training Stability**
   - Learning rate scheduling
   - Gradient clipping
   - Layer normalization

3. **Speed Improvements**
   - Efficient attention variants
   - Hardware optimization
   - Distributed training'''),

    nbf.v4.new_markdown_cell('''## Practical Considerations

1. **Hardware Requirements**
   - GPU memory needs
   - Training infrastructure
   - Inference optimization

2. **Common Challenges**
   - Gradient instability
   - Memory constraints
   - Long training times

3. **Best Practices**
   - Careful hyperparameter tuning
   - Regular checkpointing
   - Monitoring and debugging''')
]

nb['cells'] = cells

# Add metadata
nb['metadata'] = {
    'kernelspec': {
        'display_name': 'Python 3',
        'language': 'python',
        'name': 'python3'
    },
    'language_info': {
        'codemirror_mode': {'name': 'ipython', 'version': 3},
        'file_extension': '.py',
        'mimetype': 'text/x-python',
        'name': 'python',
        'nbconvert_exporter': 'python',
        'pygments_lexer': 'ipython3',
        'version': '3.12'
    },
    'rise': {
        'autolaunch': True,
        'enable_chalkboard': True,
        'progress': True,
        'scroll': True,
        'theme': 'simple'
    }
}

# Write the notebook
nbf.write(nb, 'slides/02_technical_deep_dive.ipynb') 