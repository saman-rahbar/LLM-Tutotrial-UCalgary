import json

notebook = {
    "cells": [
        {
            "cell_type": "markdown",
            "metadata": {"id": "intro"},
            "source": [
                "# Training a Simple GPT Model (Colab Version)\n\n"
                "In this notebook, we will train a small GPT model on the Daily Dialog dataset using Hugging Face transformers library. "
                "This version is specifically adapted for Google Colab and includes all necessary code to train the model.\n\n"
                "First, let's install the required packages:"
            ]
        },
        {
            "cell_type": "code",
            "metadata": {"id": "install_packages"},
            "source": ["!pip install transformers datasets torch tqdm matplotlib"],
            "execution_count": None,
            "outputs": []
        },
        {
            "cell_type": "code",
            "metadata": {"id": "imports"},
            "source": [
                "import torch\n"
                "import torch.nn as nn\n"
                "from torch.nn import functional as F\n"
                "import numpy as np\n"
                "from datasets import load_dataset\n"
                "from transformers import GPT2Tokenizer, GPT2Config, GPT2LMHeadModel\n"
                "import matplotlib.pyplot as plt\n"
                "from tqdm.auto import tqdm\n\n"
                "# Check if GPU is available\n"
                "device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')\n"
                "print(f'Using device: {device}')"
            ],
            "execution_count": None,
            "outputs": []
        },
        {
            "cell_type": "markdown",
            "metadata": {"id": "config_section"},
            "source": [
                "## Model Configuration\n\n"
                "First, let's define our model configuration. Feel free to modify these parameters to experiment with different model sizes and training settings."
            ]
        },
        {
            "cell_type": "code",
            "metadata": {"id": "config_implementation"},
            "source": [
                "class Config:\n"
                "    # Model architecture parameters\n"
                "    n_layer = 6          # Number of transformer layers\n"
                "    n_head = 8           # Number of attention heads\n"
                "    n_embd = 512        # Embedding dimension\n"
                "    block_size = 128     # Maximum sequence length\n"
                "    dropout = 0.1        # Dropout rate\n"
                "    \n"
                "    # Training parameters\n"
                "    batch_size = 32\n"
                "    learning_rate = 3e-4\n"
                "    max_iters = 5000\n"
                "    eval_interval = 500   # How often to evaluate on validation set\n"
                "    eval_iters = 200     # How many batches to evaluate on\n"
                "    device = \"cuda\" if torch.cuda.is_available() else \"cpu\"\n\n"
                "# Initialize configuration\n"
                "config = Config()\n\n"
                "# Print configuration\n"
                "print(f\"Model Configuration:\")\n"
                "print(f\"- Number of layers: {config.n_layer}\")\n"
                "print(f\"- Number of attention heads: {config.n_head}\")\n"
                "print(f\"- Embedding dimension: {config.n_embd}\")\n"
                "print(f\"- Maximum sequence length: {config.block_size}\")\n"
                "print(f\"- Dropout rate: {config.dropout}\")\n"
                "print(f\"\\nTraining Configuration:\")\n"
                "print(f\"- Batch size: {config.batch_size}\")\n"
                "print(f\"- Learning rate: {config.learning_rate}\")\n"
                "print(f\"- Maximum iterations: {config.max_iters}\")\n"
                "print(f\"- Device: {config.device}\")"
            ],
            "execution_count": None,
            "outputs": []
        },
        {
            "cell_type": "markdown",
            "metadata": {"id": "data_section"},
            "source": [
                "## Data Loading and Preprocessing\n\n"
                "Now we'll load and preprocess the Daily Dialog dataset using Hugging Face's datasets library."
            ]
        },
        {
            "cell_type": "code",
            "metadata": {"id": "data_implementation"},
            "source": [
                "def load_and_preprocess_data(config):\n"
                "    \"\"\"Load and preprocess the Daily Dialog dataset using HuggingFace.\"\"\"\n"
                "    # Load the dataset\n"
                "    print(\"Loading dataset...\")\n"
                "    dataset = load_dataset(\"daily_dialog\")\n"
                "    \n"
                "    # Initialize tokenizer\n"
                "    print(\"Initializing tokenizer...\")\n"
                "    tokenizer = GPT2Tokenizer.from_pretrained(\"gpt2\")\n"
                "    if tokenizer.pad_token is None:\n"
                "        tokenizer.pad_token = tokenizer.eos_token\n"
                "    \n"
                "    def tokenize_function(examples):\n"
                "        \"\"\"Tokenize a batch of examples.\"\"\"\n"
                "        # Join all dialogues with spaces and add EOS token\n"
                "        texts = [\" \".join(d) + tokenizer.eos_token for d in examples[\"dialog\"]]\n"
                "        return tokenizer(\n"
                "            texts,\n"
                "            padding=\"max_length\",\n"
                "            truncation=True,\n"
                "            max_length=config.block_size,\n"
                "            return_tensors=\"pt\"\n"
                "        )\n"
                "    \n"
                "    # Process training data\n"
                "    print(\"Processing training data...\")\n"
                "    train_data = dataset[\"train\"].map(\n"
                "        tokenize_function,\n"
                "        batched=True,\n"
                "        batch_size=100,\n"
                "        remove_columns=dataset[\"train\"].column_names,\n"
                "        num_proc=4\n"
                "    )\n"
                "    \n"
                "    # Process validation data\n"
                "    print(\"Processing validation data...\")\n"
                "    val_data = dataset[\"validation\"].map(\n"
                "        tokenize_function,\n"
                "        batched=True,\n"
                "        batch_size=100,\n"
                "        remove_columns=dataset[\"validation\"].column_names,\n"
                "        num_proc=4\n"
                "    )\n"
                "    \n"
                "    # Convert to PyTorch datasets\n"
                "    train_data.set_format(type=\"torch\", columns=[\"input_ids\", \"attention_mask\"])\n"
                "    val_data.set_format(type=\"torch\", columns=[\"input_ids\", \"attention_mask\"])\n"
                "    \n"
                "    return train_data, val_data, tokenizer\n\n"
                "def get_batch(split, train_data, val_data, config):\n"
                "    \"\"\"Get a batch of data for training or validation.\"\"\"\n"
                "    data = train_data if split == 'train' else val_data\n"
                "    \n"
                "    # Generate random indices\n"
                "    ix = torch.randint(len(data), (config.batch_size,))\n"
                "    \n"
                "    # Get the sequences\n"
                "    batch = data[ix]\n"
                "    x = batch[\"input_ids\"]\n"
                "    \n"
                "    # Create input-target pairs by shifting\n"
                "    x = x[:, :-1]\n"
                "    y = batch[\"input_ids\"][:, 1:]\n"
                "    \n"
                "    # Move to appropriate device\n"
                "    x, y = x.to(config.device), y.to(config.device)\n"
                "    \n"
                "    return x, y\n\n"
                "# Load and preprocess data\n"
                "train_data, val_data, tokenizer = load_and_preprocess_data(config)\n\n"
                "print(f\"\\nDataset Statistics:\")\n"
                "print(f\"- Training examples: {len(train_data)}\")\n"
                "print(f\"- Validation examples: {len(val_data)}\")\n"
                "print(f\"- Vocabulary size: {len(tokenizer)}\")"
            ],
            "execution_count": None,
            "outputs": []
        },
        {
            "cell_type": "markdown",
            "metadata": {"id": "model_section"},
            "source": [
                "## Model Implementation\n\n"
                "Let's implement our GPT model using HuggingFace's transformers library."
            ]
        },
        {
            "cell_type": "code",
            "metadata": {"id": "model_implementation"},
            "source": [
                "class GPT(nn.Module):\n"
                "    \"\"\"GPT Language Model using HuggingFace's GPT2\"\"\"\n"
                "    \n"
                "    def __init__(self, config):\n"
                "        super().__init__()\n"
                "        self.config = GPT2Config(\n"
                "            n_layer=config.n_layer,\n"
                "            n_head=config.n_head,\n"
                "            n_embd=config.n_embd,\n"
                "            vocab_size=50257,  # GPT-2 vocabulary size\n"
                "            n_positions=config.block_size,\n"
                "            n_ctx=config.block_size,\n"
                "            dropout=config.dropout\n"
                "        )\n"
                "        self.transformer = GPT2LMHeadModel(self.config)\n"
                "        self.block_size = config.block_size\n\n"
                "    def forward(self, idx, targets=None):\n"
                "        # Forward pass through GPT-2\n"
                "        outputs = self.transformer(\n"
                "            idx,\n"
                "            labels=targets if targets is not None else None,\n"
                "            return_dict=True\n"
                "        )\n"
                "        \n"
                "        if targets is not None:\n"
                "            return outputs.logits, outputs.loss\n"
                "        else:\n"
                "            return outputs.logits, None\n\n"
                "@torch.no_grad()\n"
                "def estimate_loss(model, train_data, val_data, config):\n"
                "    \"\"\"Estimate loss on train and validation sets\"\"\"\n"
                "    out = {}\n"
                "    model.eval()\n"
                "    for split in ['train', 'val']:\n"
                "        losses = torch.zeros(config.eval_iters)\n"
                "        for k in range(config.eval_iters):\n"
                "            X, Y = get_batch(split, train_data, val_data, config)\n"
                "            logits, loss = model(X, Y)\n"
                "            losses[k] = loss.item()\n"
                "        out[split] = losses.mean()\n"
                "    model.train()\n"
                "    return out\n\n"
                "@torch.no_grad()\n"
                "def generate(model, tokenizer, prompt, max_tokens=50, temperature=1.0, top_k=None, config=None):\n"
                "    \"\"\"Generate text from a prompt.\"\"\"\n"
                "    # Encode the prompt\n"
                "    encoded = tokenizer.encode(prompt, return_tensors='pt').to(config.device)\n"
                "    \n"
                "    # Generate using HuggingFace's built-in generation\n"
                "    output_sequences = model.transformer.generate(\n"
                "        encoded,\n"
                "        max_length=len(encoded[0]) + max_tokens,\n"
                "        temperature=temperature,\n"
                "        top_k=top_k if top_k is not None else 50,\n"
                "        top_p=0.9,\n"
                "        do_sample=True,\n"
                "        pad_token_id=tokenizer.pad_token_id,\n"
                "        bos_token_id=tokenizer.bos_token_id,\n"
                "        eos_token_id=tokenizer.eos_token_id\n"
                "    )\n"
                "    \n"
                "    # Decode the generated text\n"
                "    generated_text = tokenizer.decode(output_sequences[0], skip_special_tokens=True)\n"
                "    return generated_text\n\n"
                "# Initialize the model\n"
                "model = GPT(config)\n"
                "model.to(config.device)\n\n"
                "# Print model summary\n"
                "print(\"Model Architecture:\")\n"
                "total_params = sum(p.numel() for p in model.parameters())\n"
                "trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)\n"
                "print(f\"- Total parameters: {total_params:,}\")\n"
                "print(f\"- Trainable parameters: {trainable_params:,}\")"
            ],
            "execution_count": None,
            "outputs": []
        },
        {
            "cell_type": "markdown",
            "metadata": {"id": "training_section"},
            "source": [
                "## Training Loop\n\n"
                "Now we'll train the model, tracking both training and validation loss."
            ]
        },
        {
            "cell_type": "code",
            "metadata": {"id": "training_implementation"},
            "source": [
                "# Create optimizer\n"
                "optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)\n\n"
                "# Lists to store losses for plotting\n"
                "train_losses = []\n"
                "val_losses = []\n"
                "iterations = []\n\n"
                "# Training loop\n"
                "for iter in tqdm(range(config.max_iters), desc=\"Training\"):\n"
                "    # Sample a batch of data\n"
                "    xb, yb = get_batch('train', train_data, val_data, config)\n"
                "    \n"
                "    # Forward pass\n"
                "    logits, loss = model(xb, yb)\n"
                "    \n"
                "    # Backward pass\n"
                "    optimizer.zero_grad(set_to_none=True)\n"
                "    loss.backward()\n"
                "    optimizer.step()\n"
                "    \n"
                "    # Evaluate the model\n"
                "    if iter % config.eval_interval == 0:\n"
                "        losses = estimate_loss(model, train_data, val_data, config)\n"
                "        print(f\"Step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}\")\n"
                "        \n"
                "        # Store losses for plotting\n"
                "        train_losses.append(losses['train'])\n"
                "        val_losses.append(losses['val'])\n"
                "        iterations.append(iter)\n\n"
                "# Plot learning curves\n"
                "plt.figure(figsize=(10, 6))\n"
                "plt.plot(iterations, train_losses, label='Train Loss')\n"
                "plt.plot(iterations, val_losses, label='Validation Loss')\n"
                "plt.xlabel('Iteration')\n"
                "plt.ylabel('Loss')\n"
                "plt.title('Training and Validation Loss')\n"
                "plt.legend()\n"
                "plt.grid(True)\n"
                "plt.show()"
            ],
            "execution_count": None,
            "outputs": []
        },
        {
            "cell_type": "markdown",
            "metadata": {"id": "generation_section"},
            "source": [
                "## Text Generation\n\n"
                "Let's test our trained model by generating some text:"
            ]
        },
        {
            "cell_type": "code",
            "metadata": {"id": "generation_implementation"},
            "source": [
                "# Test text generation with different prompts and temperatures\n"
                "prompts = [\n"
                "    \"Hello, how are you\",\n"
                "    \"I'm planning to\",\n"
                "    \"The weather is\"\n"
                "]\n\n"
                "temperatures = [0.7, 1.0, 1.2]\n\n"
                "for prompt in prompts:\n"
                "    print(f\"\\nPrompt: {prompt}\")\n"
                "    for temp in temperatures:\n"
                "        generated = generate(model, tokenizer, prompt, max_tokens=50, temperature=temp, config=config)\n"
                "        print(f\"\\nTemperature {temp}:\")\n"
                "        print(generated)"
            ],
            "execution_count": None,
            "outputs": []
        },
        {
            "cell_type": "markdown",
            "metadata": {"id": "save_section"},
            "source": [
                "## Save the Model\n\n"
                "Finally, let's save our trained model and download it from Colab:"
            ]
        },
        {
            "cell_type": "code",
            "metadata": {"id": "save_implementation"},
            "source": [
                "# Create a directory for the model\n"
                "import os\n"
                "os.makedirs('models', exist_ok=True)\n\n"
                "# Save the model\n"
                "model_path = 'models/gpt_model'\n"
                "model.transformer.save_pretrained(model_path)\n"
                "tokenizer.save_pretrained(model_path)\n\n"
                "print(f\"Model and tokenizer saved to {model_path}\")\n\n"
                "# For Colab: Download the model\n"
                "from google.colab import files\n"
                "!zip -r models.zip models/\n"
                "files.download('models.zip')"
            ],
            "execution_count": None,
            "outputs": []
        }
    ],
    "metadata": {
        "accelerator": "GPU",
        "colab": {
            "name": "Training a Simple GPT Model",
            "provenance": []
        },
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3"
        },
        "language_info": {
            "name": "python",
            "version": "3.8.0"
        }
    },
    "nbformat": 4,
    "nbformat_minor": 5
}

with open('notebooks/01_training_simple_gpt_colab.ipynb', 'w') as f:
    json.dump(notebook, f, indent=1) 