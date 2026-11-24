"""Simple interactive local terminal interface for the trained model."""
import argparse
from pathlib import Path
import torch
from generate import load_model, ROOT

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=Path, default=ROOT / "checkpoints" / "best.pt")
    p.add_argument("--max-new-tokens", type=int, default=180)
    p.add_argument("--temperature", type=float, default=0.8)
    p.add_argument("--top-k", type=int, default=40)
    p.add_argument("--device", default="auto")
    args = p.parse_args()
    device = args.device
    if device == "auto": device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    model, tokenizer, _ = load_model(args.checkpoint, device)
    print("TinyShakespeare GPT-2 GQA — type /exit to quit. This is a text-completion model, not an instruction-tuned assistant.")
    while True:
        try: prompt = input("\nPrompt> ")
        except (EOFError, KeyboardInterrupt): break
        if prompt.strip() == "/exit": break
        ids = tokenizer.encode(prompt)
        if not ids: print("Please enter a prompt."); continue
        x = torch.tensor([ids], dtype=torch.long, device=device)
        with torch.no_grad(): y = model.generate(x, args.max_new_tokens, args.temperature, args.top_k)
        print("\n" + tokenizer.decode(y[0].tolist()))

if __name__ == "__main__": main()
