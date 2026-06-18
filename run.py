import json
import torch
import torch.nn as nn
import sys

# =========================
# LOAD TOKENIZER
# =========================

with open("chars.json", "r", encoding="utf-8") as f:
    chars = json.load(f)

stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for ch, i in stoi.items()}
vocab_size = len(chars)

BLOCK_SIZE = 128
N_EMBD = 128
N_HEADS = 4

def encode(s):
    return torch.tensor([stoi[c] for c in s], dtype=torch.long)

def decode(t):
    return "".join([itos[int(i)] for i in t])

# =========================
# MODEL
# =========================

class TinyGPT(nn.Module):
    def __init__(self):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, N_EMBD)
        self.position_embedding = nn.Embedding(BLOCK_SIZE, N_EMBD)

        self.attn = nn.MultiheadAttention(
            N_EMBD, N_HEADS, batch_first=True
        )

        self.ln1 = nn.LayerNorm(N_EMBD)
        self.ff = nn.Sequential(
            nn.Linear(N_EMBD, 4*N_EMBD),
            nn.ReLU(),
            nn.Linear(4*N_EMBD, N_EMBD)
        )
        self.ln2 = nn.LayerNorm(N_EMBD)

        self.lm_head = nn.Linear(N_EMBD, vocab_size)

    def forward(self, idx):
        B, T = idx.shape
        tok = self.token_embedding(idx)
        pos = self.position_embedding(torch.arange(T))
        x = tok + pos

        attn_out, _ = self.attn(x, x, x)
        x = self.ln1(x + attn_out)
        x = self.ln2(x + self.ff(x))

        logits = self.lm_head(x)
        return logits

# =========================
# GENERATE
# =========================

def generate(model, start="h", max_new_tokens=100, temperature=0.8):
    model.eval()
    idx = encode(start).unsqueeze(0)

    for _ in range(max_new_tokens):
        logits = model(idx[:, -BLOCK_SIZE:])
        logits = logits[:, -1, :] / temperature
        probs = torch.softmax(logits, dim=-1)
        next_id = probs.multinomial(1)
        idx = torch.cat([idx, next_id], dim=1)

    return decode(idx[0])

# =========================
# LOAD MODEL + RUN
# =========================

model = TinyGPT()
model.load_state_dict(torch.load("final_model.pt", map_location="cpu"))
model.eval()

prompt = sys.argv[1] if len(sys.argv) > 1 else "hello"
print("Prompt:", prompt)
print(generate(model, prompt))
print("\n--- END OF MODEL OUTPUT ---\n")
