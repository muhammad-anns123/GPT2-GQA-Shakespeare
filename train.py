"""Train GPT-2-style GQA on Tiny Shakespeare."""
import argparse, math, random
from pathlib import Path
import numpy as np
import torch
from model import GPTConfig, GPTLanguageModel
from tokenizer import ByteBPETokenizer

ROOT = Path(__file__).parent
DATA = ROOT / "data"
CKPT = ROOT / "checkpoints" / "best.pt"


def get_batch(data, batch_size, block_size, device):
    ix = torch.randint(len(data) - block_size - 1, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    return x.to(device), y.to(device)

@torch.no_grad()
def estimate_loss(model, train_data, val_data, batch_size, block_size, device, eval_iters=20):
    model.eval(); result = {}
    for name, data in (("train", train_data), ("val", val_data)):
        losses = []
        for _ in range(eval_iters):
            x, y = get_batch(data, batch_size, block_size, device)
            _, loss = model(x, y); losses.append(loss.item())
        result[name] = sum(losses) / len(losses)
    model.train(); return result


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--max-iters", type=int, default=5000)
    p.add_argument("--eval-interval", type=int, default=250)
    p.add_argument("--eval-iters", type=int, default=20)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--block-size", type=int, default=256)
    p.add_argument("--n-layer", type=int, default=4)
    p.add_argument("--n-head", type=int, default=4)
    p.add_argument("--n-kv-head", type=int, default=2)
    p.add_argument("--n-embd", type=int, default=128)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--learning-rate", type=float, default=3e-4)
    p.add_argument("--seed", type=int, default=1337)
    p.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda", "mps"])
    args = p.parse_args()
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    else: device = args.device
    if device == "cuda" and not torch.cuda.is_available(): raise RuntimeError("CUDA requested but unavailable")
    if device == "mps" and not torch.backends.mps.is_available(): raise RuntimeError("MPS requested but unavailable")
    tok_path = DATA / "tokenizer.json"
    train_path, val_path = DATA / "train_ids.npy", DATA / "val_ids.npy"
    if not (tok_path.exists() and train_path.exists() and val_path.exists()):
        raise FileNotFoundError("Dataset/tokenizer missing. Run: python prepare.py")
    tokenizer = ByteBPETokenizer.load(tok_path)
    train_data = torch.from_numpy(np.load(train_path).astype(np.int64))
    val_data = torch.from_numpy(np.load(val_path).astype(np.int64))
    config = GPTConfig(tokenizer.vocab_size, args.block_size, args.n_layer, args.n_head, args.n_kv_head, args.n_embd, args.dropout)
    model = GPTLanguageModel(config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, betas=(0.9, 0.95), weight_decay=0.1)
    scaler_enabled = device == "cuda"
    CKPT.parent.mkdir(parents=True, exist_ok=True)
    best_val = float("inf")
    print(f"Device: {device} | parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"GQA: {args.n_head} query heads / {args.n_kv_head} KV heads")
    for step in range(args.max_iters + 1):
        if step % args.eval_interval == 0 or step == args.max_iters:
            losses = estimate_loss(model, train_data, val_data, args.batch_size, args.block_size, device, args.eval_iters)
            print(f"step {step:5d} | train {losses['train']:.4f} | val {losses['val']:.4f}")
            if losses["val"] < best_val:
                best_val = losses["val"]
                torch.save({"model": model.state_dict(), "config": model.config_dict(), "step": step,
                            "best_val_loss": best_val, "tokenizer": str(tok_path)}, CKPT)
        if step == args.max_iters: break
        xb, yb = get_batch(train_data, args.batch_size, args.block_size, device)
        optimizer.zero_grad(set_to_none=True)
        _, loss = model(xb, yb)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
    print(f"Best checkpoint: {CKPT} (validation loss {best_val:.4f})")

if __name__ == "__main__": main()
