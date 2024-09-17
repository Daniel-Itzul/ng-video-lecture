import torch
import torch.nn as nn
from torch.nn import functional as F

# hyperparameters
batch_size = 32  # how many independent sequences will we process in parallel?
block_size = 8   # what is the maximum context length for predictions?
max_iters = 3000
eval_interval = 300
learning_rate = 1e-2
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters = 200
# Seed for reproducibility
torch.manual_seed(5000)

# Function to load the data
def get_training_data(path):
    with open(path, 'r', encoding='utf-8') as f:
        text = f.read()
    return text

# Function to create mappings from characters to integers and vice versa
def create_mappings(chars):
    stoi = {ch: i for i, ch in enumerate(chars)}
    itos = {i: ch for i, ch in enumerate(chars)}
    return stoi, itos

# Function to create encoders and decoders using the mappings
def create_encoders(stoi, itos):
    encode = lambda s: [stoi[ch] for ch in s]  # encoder: take a string, output a list of integers
    decode = lambda l: ''.join([itos[i] for i in l])  # decoder: take a list of integers, output a string
    return encode, decode

# Function to prepare the data for character-level modeling
def prepare_data(text, split_value):
    # Determine unique characters
    chars = sorted(list(set(text)))
    vocab_size = len(chars)
    
    # Create mappings from characters to integers and vice versa
    stoi, itos = create_mappings(chars)
    
    # Encode the text
    encode = lambda s: [stoi[ch] for ch in s]
    data = torch.tensor(encode(text), dtype=torch.long)

    # Train and test splits
    n = int(split_value * len(data))  # first 90% will be train, rest val
    train_data = data[:n]
    val_data = data[n:]
    
    return train_data, val_data, vocab_size, stoi, itos

# Data loading function
def get_batch(data):
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y

# Estimate loss function
@torch.no_grad()
def estimate_loss(model, train_data, val_data):
    out = {}
    model.eval()
    for split_name, data in [('train', train_data), ('val', val_data)]:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(data)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split_name] = losses.mean()
    model.train()
    return out

# Bigram model for character-level modeling
class BigramLanguageModel(nn.Module):

    def __init__(self, vocab_size):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, vocab_size)

    def forward(self, idx, targets=None):
        logits = self.token_embedding_table(idx)  # (B,T,C)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            logits, loss = self(idx)
            logits = logits[:, -1, :]  # (B, C)
            probs = F.softmax(logits, dim=-1)  # (B, C)
            idx_next = torch.multinomial(probs, num_samples=1)  # (B, 1)
            idx = torch.cat((idx, idx_next), dim=1)  # (B, T+1)
        return idx

# Function to train the model
def train_model(model, optimizer, max_iters, eval_interval, train_data, val_data):
    for iter in range(max_iters):
        # Every once in a while, evaluate the loss on train and val sets
        if iter % eval_interval == 0:
            losses = estimate_loss(model, train_data, val_data)
            print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

        # Sample a batch of data
        xb, yb = get_batch(train_data)

        # Forward pass and loss computation
        logits, loss = model(xb, yb)

        # Backward pass and parameter update
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

# Main Script

# Load the data
data = get_training_data('input.txt')

# Prepare the data (includes extracting unique characters and creating mappings)
train_data, val_data, vocab_size, stoi, itos = prepare_data(data, 0.8)

# Create encoders and decoders
encode, decode = create_encoders(stoi, itos)

# Instantiate the model and optimizer
model = BigramLanguageModel(vocab_size)
model = model.to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

# Train the model
train_model(model, optimizer, max_iters, eval_interval, train_data, val_data)

# Generate text from the model
context = torch.zeros((1, 1), dtype=torch.long, device=device)
generated_indices = model.generate(context, max_new_tokens=500)[0].tolist()
print(decode(generated_indices))
