import os
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import seaborn as sns

# =========================
# CONFIG (OPTIMIZED)
# =========================

DATA_PATH = "english_training_data.txt"
CHECKPOINT_DIR = "checkpoints"
FINAL_MODEL_PATH = "final_model.pt"

TRAIN_STEPS = 5000        
START_STEP = 0

BLOCK_SIZE = 128          
N_EMBD = 128
N_HEADS = 4               
DROPOUT = 0.0           
LR = 1e-3                
BATCH_SIZE = 32           

device = "cuda" if torch.cuda.is_available() else "cpu"
os.makedirs(CHECKPOINT_DIR, exist_ok=True)


# =========================
# DATA & TOKENIZER
# =========================

with open(DATA_PATH, "r", encoding="utf-8") as f:
    custom_text = f.read()

chars = sorted(list(set(custom_text)))
stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for ch, i in stoi.items()}
vocab_size = len(stoi)

def encode(s: str) -> torch.Tensor:
    return torch.tensor([stoi[c] for c in s], dtype=torch.long)

def decode(t: torch.Tensor) -> str:
    return "".join([itos[int(i)] for i in t])

data = encode(custom_text)

def get_batch(batch_size=BATCH_SIZE, block_size=BLOCK_SIZE):
    ix = torch.randint(len(data) - block_size - 1, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    return x.to(device), y.to(device)


# =========================
# MODEL
# =========================

class TinyGPT(nn.Module):
    def __init__(self, vocab_size, n_embd=N_EMBD, block_size=BLOCK_SIZE, n_heads=N_HEADS, dropout=DROPOUT):
        super().__init__()
        self.block_size = block_size

        self.token_embedding = nn.Embedding(vocab_size, n_embd)
        self.position_embedding = nn.Embedding(block_size, n_embd)

        self.attn = nn.MultiheadAttention(
            embed_dim=n_embd,
            num_heads=n_heads,
            batch_first=True,
            dropout=dropout
        )

        self.ln1 = nn.LayerNorm(n_embd)
        self.ff = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout),
        )
        self.ln2 = nn.LayerNorm(n_embd)

        self.lm_head = nn.Linear(n_embd, vocab_size)

    def forward(self, idx):
        B, T = idx.shape
        assert T <= self.block_size, "Sekvenslängd > block_size"

        tok = self.token_embedding(idx)
        pos = self.position_embedding(torch.arange(T, device=idx.device))
        x = tok + pos

        # Causal mask: tillåt bara upp till diagonalen
        mask = torch.tril(torch.ones(T, T, device=idx.device)).bool()
        attn_out, attn_weights = self.attn(
            x, x, x,
            attn_mask=~mask,
            need_weights=True
        )

        x = self.ln1(x + attn_out)
        x = self.ln2(x + self.ff(x))

        logits = self.lm_head(x)
        return logits, attn_weights


# =========================
# TRANING
# =========================

def train():
    model = TinyGPT(vocab_size=vocab_size).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
    loss_history = []
    start_step = START_STEP

    # resume
    latest_ckpt = None
    ckpts = [f for f in os.listdir(CHECKPOINT_DIR) if f.startswith("step_") and f.endswith(".pt")]
    if ckpts:
        latest_ckpt = sorted(ckpts, key=lambda x: int(x.split("_")[1].split(".")[0]))[-1]

    if latest_ckpt:
        path = os.path.join(CHECKPOINT_DIR, latest_ckpt)
        ckpt = torch.load(path, map_location=device)
        model.load_state_dict(ckpt["model_state"])
        optimizer.load_state_dict(ckpt["optimizer_state"])
        start_step = ckpt["step"] + 1
        print(f"Återupptar från steg {start_step} ({path})")

    for step in range(start_step, TRAIN_STEPS):
        model.train()
        xb, yb = get_batch()

        logits, _ = model(xb)
        loss = F.cross_entropy(logits.view(-1, vocab_size), yb.view(-1))

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        loss_history.append(loss.item())

        if step % 200 == 0:
            print(f"Steg {step}, loss {loss.item():.4f}")

        if step % 1000 == 0 and step > 0:
            path = os.path.join(CHECKPOINT_DIR, f"step_{step}.pt")
            torch.save({
                "step": step,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict()
            }, path)
            print(f"Checkpoint sparad: {path}")

    torch.save(model.state_dict(), FINAL_MODEL_PATH)
    print(f"Slutmodell sparad: {FINAL_MODEL_PATH}")

    plt.figure()
    plt.plot(loss_history)
    plt.xlabel("Steg")
    plt.ylabel("Loss")
    plt.title("Träningsförlopp")
    plt.tight_layout()
    plt.savefig("training_loss.png")
    print("Loss‑graf sparad som training_loss.png")

    return model


# =========================
# GENERATION
# =========================

@torch.no_grad()
def generate(model, start="h", max_new_tokens=100, temperature=0.8, top_k=20):
    model.eval()
    idx = encode(start).unsqueeze(0).to(device)

    for _ in range(max_new_tokens):
        idx_cond = idx[:, -BLOCK_SIZE:]
        logits, _ = model(idx_cond)
        logits = logits[:, -1, :] / temperature

        if top_k is not None:
            v, _ = torch.topk(logits, top_k)
            logits[logits < v[:, [-1]]] = -float("inf")

        probs = torch.softmax(logits, dim=-1)
        next_id = probs.multinomial(1)
        idx = torch.cat([idx, next_id], dim=1)

    return decode(idx[0].cpu())


# =========================
# ATTENTION VISUALIZATION
# =========================

@torch.no_grad()
def visualize_attention(model, text):
    import numpy as np
    import seaborn as sns
    import matplotlib.pyplot as plt

    model.eval()
    idx = encode(text).unsqueeze(0).to(device)

    logits, attn = model(idx)  # attn shape: (B, T, T)
    attn_matrix = attn[0].detach().cpu().numpy()  # (T, T)

    plt.figure(figsize=(5, 4))
    sns.heatmap(attn_matrix, cmap="viridis", square=True)
    plt.title(f"Attention heatmap för '{text}'")
    plt.xlabel("Key position")
    plt.ylabel("Query position")
    plt.tight_layout()
    plt.savefig("attention.png")
    print("Attention‑graf sparad som attention.png")


# =========================
# SANITY-OVERFIT
# =========================

def sanity_overfit():
    print("Kör sanity_overfit på 64 tecken...")
    model = TinyGPT(vocab_size=vocab_size).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    seq_len = 64
    x = data[:seq_len].unsqueeze(0).to(device)
    y = data[1:seq_len+1].unsqueeze(0).to(device)

    for step in range(301):
        model.train()
        logits, _ = model(x)
        loss = F.cross_entropy(logits.view(-1, vocab_size), y.view(-1))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if step % 50 == 0:
            print(f"Overfit‑steg {step}, loss {loss.item():.4f}")



# =========================
# MAIN
# =========================

if __name__ == "__main__":
    print(f"Device: {device}")
    print(f"Vocab size: {vocab_size}, block_size: {BLOCK_SIZE}")

    sanity_overfit()  # Viktigt: kolla att modellen kan lära sig alls

    model = train()

    with open("chars.json", "w", encoding="utf-8") as f:
    	json.dump(chars, f, ensure_ascii=False, indent=2)
    print()
    print(chars)
    print()
    print()

    print("\nExempelgenerering:")
    print(generate(model, "hello", max_new_tokens=200))

    visualize_attention(model, "hello")



