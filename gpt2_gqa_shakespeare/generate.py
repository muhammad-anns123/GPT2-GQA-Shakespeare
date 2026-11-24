"""Generate text from a trained checkpoint."""
import argparse
from pathlib import Path
import torch
from model import GPTConfig, GPTLanguageModel
from tokenizer import ByteBPETokenizer

ROOT = Path(__file__).parent

def load_model(checkpoint: Path, device: str):
    # Checkpoints are locally-created project artifacts; weights_only=False is
    # required because the checkpoint also contains a plain config dictionary.
    payload = torch.load(checkpoint, map_location=device, weights_only=False)
    config = GPTConfig(**payload["config"])
    model = GPTLanguageModel(config).to(device)
    model.load_state_dict(payload["model"])
    model.eval()
    tokenizer = ByteBPETokenizer.load(ROOT / "data" / "tokenizer.json")
    return model, tokenizer, payload

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--prompt", default="ROMEO:")
    p.add_argument("--max-new-tokens", type=int, default=400)
    p.add_argument("--temperature", type=float, default=0.8)
    p.add_argument("--top-k", type=int, default=50)
    p.add_argument("--checkpoint", type=Path, default=ROOT / "checkpoints" / "best.pt")
    p.add_argument("--device", default="auto")
    args = p.parse_args()
    device = args.device
    if device == "auto": device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    if not args.checkpoint.exists(): raise FileNotFoundError(f"No checkpoint at {args.checkpoint}. Train first with python train.py")
    model, tokenizer, meta = load_model(args.checkpoint, device)
    ids = tokenizer.encode(args.prompt)
    if not ids: ids = [ord(" ")]
    x = torch.tensor([ids], dtype=torch.long, device=device)
    with torch.no_grad(): out = model.generate(x, args.max_new_tokens, args.temperature, args.top_k)
    print(tokenizer.decode(out[0].tolist()))

if __name__ == "__main__": main()
