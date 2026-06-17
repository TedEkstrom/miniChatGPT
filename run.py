import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import sys

# --- Same parameters as during training ---
block_size = 128
n_embd = 128
n_heads = 4

# --- Tokenizer from training ---
with open("chars.json", "r", encoding="utf-8") as f:
    chars = json.load(f)

stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for ch, i in stoi.items()}
vocab_size = len(chars)

def encode(s):
    return torch.tensor([stoi[c] for c in s], dtype=torch.long)

def decode(t):
    return "".join([itos[int(i)] for i in t])

# --- TinyGPT-model ---
class TinyGPT(nn.Module):
    def __init__(self, vocab_size, n_embd=n_embd):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, n_embd)
        self.position_embedding = nn.Embedding(block_size, n_embd)

        self.attn = nn.MultiheadAttention(
            n_embd,
            num_heads=n_heads,
            batch_first=True
        )
        self.ln1 = nn.LayerNorm(n_embd)

        self.ff = nn.Sequential(
            nn.Linear(n_embd, 4*n_embd),
            nn.ReLU(),
            nn.Linear(4*n_embd, n_embd)
        )
        self.ln2 = nn.LayerNorm(n_embd)

        self.lm_head = nn.Linear(n_embd, vocab_size)

    def forward(self, idx):
        B, T = idx.shape
        tok = self.token_embedding(idx)
        pos = self.position_embedding(torch.arange(T))
        x = tok + pos

        attn_out, _ = self.attn(x, x, x, need_weights=False)
        x = self.ln1(x + attn_out)

        ff_out = self.ff(x)
        x = self.ln2(x + ff_out)

        logits = self.lm_head(x)
        return logits

# --- Generate function ---
def generate(model, start="h", max_new_tokens=50, temperature=0.8):
    model.eval()
    idx = encode(start).unsqueeze(0)

    for _ in range(max_new_tokens):
        logits = model(idx[:, -block_size:])
        logits = logits[:, -1, :] / temperature
        probs = torch.softmax(logits, dim=-1)
        next_id = probs.multinomial(1)
        idx = torch.cat([idx, next_id], dim=1)

    return decode(idx[0])

# --- Load model ---
model = TinyGPT(vocab_size)
model.load_state_dict(torch.load("final_model.pt", map_location="cpu"))
model.eval()

# --- Promt-test ---
prompt = sys.argv[1] if len(sys.argv) > 1 else "hej"
print("Prompt:", prompt)
print(generate(model, prompt))
