import nbformat as nbf

nb = nbf.v4.new_notebook()

# Introduction
nb['cells'] = [nbf.v4.new_markdown_cell("""# Training a Simple GPT Model

In this notebook, we'll train a small GPT model on the Daily Dialog dataset using Hugging Face's transformers library. We'll use a simplified architecture based on GPT-2 but with fewer parameters for faster training.""")]

# Imports
nb['cells'].append(nbf.v4.new_code_cell("""import sys
sys.path.append('..')

from gpt_training import (
    Config,
    load_and_preprocess_data,
    get_batch,
    GPT,
    estimate_loss,
    generate
)
import torch
import matplotlib.pyplot as plt
from tqdm.auto import tqdm"""))

# Model Configuration
nb['cells'].append(nbf.v4.new_markdown_cell("""## Model Configuration

First, let's set up our model configuration. Feel free to modify these parameters to experiment with different model sizes and training settings."""))

nb['cells'].append(nbf.v4.new_code_cell("""# Initialize configuration
config = Config()

# Print configuration
print(f"Model Configuration:")
print(f"- Number of layers: {config.n_layer}")
print(f"- Number of attention heads: {config.n_head}")
print(f"- Embedding dimension: {config.n_embd}")
print(f"- Maximum sequence length: {config.block_size}")
print(f"- Dropout rate: {config.dropout}")
print(f"\\nTraining Configuration:")
print(f"- Batch size: {config.batch_size}")
print(f"- Learning rate: {config.learning_rate}")
print(f"- Maximum iterations: {config.max_iters}")
print(f"- Device: {config.device}")"""))

# Data Loading
nb['cells'].append(nbf.v4.new_markdown_cell("""## Data Loading and Preprocessing

Now we'll load and preprocess the Daily Dialog dataset using Hugging Face's datasets library."""))

nb['cells'].append(nbf.v4.new_code_cell("""# Load and preprocess data
train_data, val_data, tokenizer = load_and_preprocess_data(config)

print(f"\\nDataset Statistics:")
print(f"- Training examples: {len(train_data)}")
print(f"- Validation examples: {len(val_data)}")
print(f"- Vocabulary size: {len(tokenizer)}")"""))

# Model Initialization
nb['cells'].append(nbf.v4.new_markdown_cell("""## Model Initialization

Let's initialize our GPT model and move it to the appropriate device."""))

nb['cells'].append(nbf.v4.new_code_cell("""# Initialize the model
model = GPT(config)
model.to(config.device)

# Print model summary
print("Model Architecture:")
total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"- Total parameters: {total_params:,}")
print(f"- Trainable parameters: {trainable_params:,}")"""))

# Training Loop
nb['cells'].append(nbf.v4.new_markdown_cell("""## Training Loop

Now we'll train the model, tracking both training and validation loss."""))

nb['cells'].append(nbf.v4.new_code_cell("""# Create optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)

# Lists to store losses for plotting
train_losses = []
val_losses = []
iterations = []

# Training loop
for iter in tqdm(range(config.max_iters), desc="Training"):
    # Sample a batch of data
    xb, yb = get_batch('train', train_data, val_data, config)
    
    # Forward pass
    logits, loss = model(xb, yb)
    
    # Backward pass
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()
    
    # Evaluate the model
    if iter % config.eval_interval == 0:
        losses = estimate_loss(model, train_data, val_data, config)
        print(f"Step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
        
        # Store losses for plotting
        train_losses.append(losses['train'])
        val_losses.append(losses['val'])
        iterations.append(iter)

# Plot learning curves
plt.figure(figsize=(10, 6))
plt.plot(iterations, train_losses, label='Train Loss')
plt.plot(iterations, val_losses, label='Validation Loss')
plt.xlabel('Iteration')
plt.ylabel('Loss')
plt.title('Training and Validation Loss')
plt.legend()
plt.grid(True)
plt.show()"""))

# Text Generation
nb['cells'].append(nbf.v4.new_markdown_cell("""## Text Generation

Let's test our trained model by generating some text from different prompts."""))

nb['cells'].append(nbf.v4.new_code_cell("""# Test text generation with different prompts and temperatures
prompts = [
    "Hello, how are you",
    "I'm planning to",
    "The weather is"
]

temperatures = [0.7, 1.0, 1.2]

for prompt in prompts:
    print(f"\\nPrompt: {prompt}")
    for temp in temperatures:
        generated = generate(model, tokenizer, prompt, max_tokens=50, temperature=temp, config=config)
        print(f"\\nTemperature {temp}:")
        print(generated)"""))

# Save Model
nb['cells'].append(nbf.v4.new_markdown_cell("""## Save the Model

Finally, let's save our trained model and its configuration."""))

nb['cells'].append(nbf.v4.new_code_cell("""# Create a directory for the model
import os
os.makedirs('models', exist_ok=True)

# Save the model
model_path = 'models/gpt_model'
model.transformer.save_pretrained(model_path)
tokenizer.save_pretrained(model_path)

print(f"Model and tokenizer saved to {model_path}")"""))

# Write the notebook
with open('notebooks/01_training_simple_gpt.ipynb', 'w') as f:
    nbf.write(nb, f) 