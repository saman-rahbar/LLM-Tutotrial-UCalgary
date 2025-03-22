import nbformat as nbf

# Read the existing notebook
nb = nbf.read('notebooks/01_training_simple_gpt.ipynb', as_version=4)

# Add configuration section
nb['cells'].extend([
    nbf.v4.new_markdown_cell('## Configuration\n\nFirst, let\'s set up our model configuration:'),
    nbf.v4.new_code_cell('''class Config:
    # model parameters
    n_layer = 6
    n_head = 8
    n_embd = 512
    vocab_size = 50257  # GPT-2 vocab size
    block_size = 128
    dropout = 0.1
    
    # training parameters
    batch_size = 32
    learning_rate = 3e-4
    max_iters = 5000
    eval_interval = 500
    eval_iters = 200
    device = "cuda" if torch.cuda.is_available() else "cpu"

config = Config()''')
])

# Add data loading section
nb['cells'].extend([
    nbf.v4.new_markdown_cell('## Data Loading and Preparation\n\nWe\'ll use the Daily Dialog dataset, which contains natural conversations:'),
    nbf.v4.new_code_cell('''def load_and_preprocess_data():
    # Load the dataset
    dataset = load_dataset("daily_dialog")
    
    # Initialize tokenizer
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    
    # Prepare training data
    def prepare_data(examples):
        dialogs = [" ".join(d) for d in examples["dialog"]]
        return tokenizer(dialogs, truncation=True, max_length=config.block_size)
    
    tokenized_train = dataset["train"].map(prepare_data, remove_columns=dataset["train"].column_names)
    tokenized_valid = dataset["validation"].map(prepare_data, remove_columns=dataset["validation"].column_names)
    
    return tokenized_train, tokenized_valid, tokenizer

train_data, val_data, tokenizer = load_and_preprocess_data()''')
])

# Add model architecture section
nb['cells'].extend([
    nbf.v4.new_markdown_cell('## Model Architecture\n\nNow let\'s implement a simple GPT model:'),
    nbf.v4.new_code_cell('''class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.dropout = config.dropout
        self.register_buffer('bias', torch.tril(torch.ones(config.block_size, config.block_size))
                                    .view(1, 1, config.block_size, config.block_size))

    def forward(self, x):
        B, T, C = x.size()
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        att = (q @ k.transpose(-2, -1)) * (1.0 / np.sqrt(k.size(-1)))
        att = att.masked_fill(self.bias[:,:,:T,:T] == 0, float('-inf'))
        att = F.softmax(att, dim=-1)
        att = self.attn_dropout(att)
        y = att @ v
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.resid_dropout(self.c_proj(y))
        return y

class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln1 = nn.LayerNorm(config.n_embd)
        self.ln2 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.mlp = nn.Sequential(
            nn.Linear(config.n_embd, 4 * config.n_embd),
            nn.GELU(),
            nn.Linear(4 * config.n_embd, config.n_embd),
            nn.Dropout(config.dropout),
        )

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x

class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.tok_emb = nn.Embedding(config.vocab_size, config.n_embd)
        self.pos_emb = nn.Parameter(torch.zeros(1, config.block_size, config.n_embd))
        self.drop = nn.Dropout(config.dropout)
        self.blocks = nn.Sequential(*[Block(config) for _ in range(config.n_layer)])
        self.ln_f = nn.LayerNorm(config.n_embd)
        self.head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.block_size = config.block_size
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def forward(self, idx, targets=None):
        b, t = idx.size()
        assert t <= self.block_size, f"Cannot forward sequence of length {t}, block size is only {self.block_size}"
        pos = torch.arange(0, t, dtype=torch.long, device=idx.device).unsqueeze(0)

        tok_emb = self.tok_emb(idx)
        pos_emb = self.pos_emb[:, :t, :]
        x = self.drop(tok_emb + pos_emb)
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.head(x)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))

        return logits, loss

model = GPT(config).to(config.device)''')
])

# Add training section
nb['cells'].extend([
    nbf.v4.new_markdown_cell('## Training Loop\n\nLet\'s implement the training loop with evaluation:'),
    nbf.v4.new_code_cell('''def get_batch(split):
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data), (config.batch_size,))
    x = torch.stack([torch.tensor(data[i]['input_ids'][:config.block_size]) for i in ix])
    y = torch.stack([torch.tensor(data[i]['input_ids'][1:config.block_size+1]) for i in ix])
    x, y = x.to(config.device), y.to(config.device)
    return x, y

@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(config.eval_iters)
        for k in range(config.eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

# Create the optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)

# Training loop
losses = []
for iter in tqdm(range(config.max_iters)):
    if iter % config.eval_interval == 0:
        losses_dict = estimate_loss()
        print(f"step {iter}: train loss {losses_dict['train']:.4f}, val loss {losses_dict['val']:.4f}")
        losses.append(losses_dict['train'])

    xb, yb = get_batch('train')
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
plt.show()''')
])

# Add generation section
nb['cells'].extend([
    nbf.v4.new_markdown_cell('## Generate Text\n\nNow let\'s try generating some text with our trained model:'),
    nbf.v4.new_code_cell('''@torch.no_grad()
def generate(prompt, max_tokens=50, temperature=1.0, top_k=None):
    model.eval()
    tokens = tokenizer.encode(prompt)
    tokens = torch.tensor(tokens, dtype=torch.long, device=config.device)[None, ...]
    
    for _ in range(max_tokens):
        if tokens.size(1) >= config.block_size:
            tokens = tokens[:, -config.block_size:]
            
        logits, _ = model(tokens)
        logits = logits[:, -1, :] / temperature
        
        if top_k is not None:
            v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
            logits[logits < v[:, [-1]]] = -float('Inf')
            
        probs = F.softmax(logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)
        tokens = torch.cat((tokens, next_token), dim=1)
        
        if next_token.item() == tokenizer.eos_token_id:
            break
            
    return tokenizer.decode(tokens[0].tolist())

# Try generating some text
prompt = "Hello, how are you?"
generated_text = generate(prompt)
print(f"Prompt: {prompt}")
print(f"Generated: {generated_text}")''')
])

# Write the updated notebook
nbf.write(nb, 'notebooks/01_training_simple_gpt.ipynb') 