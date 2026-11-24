"""End-to-end runner: prepare dataset, train the model, then generate a sample."""
import argparse
import subprocess
import sys
from pathlib import Path
ROOT = Path(__file__).parent

def run(command):
    print("\n$", " ".join(map(str, command)), flush=True)
    subprocess.run(command, cwd=ROOT, check=True)

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--quick", action="store_true", help="short smoke training run; not intended for best quality")
    p.add_argument("--max-iters", type=int, default=None)
    p.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda", "mps"])
    p.add_argument("--prompt", default="ROMEO:")
    args = p.parse_args()
    run([sys.executable, "prepare.py"])
    train = [sys.executable, "train.py", "--device", args.device]
    if args.quick:
        train += ["--max-iters", "100", "--eval-interval", "25", "--batch-size", "8", "--block-size", "128", "--n-layer", "2", "--n-head", "4", "--n-kv-head", "2", "--n-embd", "64"]
    elif args.max_iters is not None:
        train += ["--max-iters", str(args.max_iters)]
    run(train)
    run([sys.executable, "generate.py", "--prompt", args.prompt, "--max-new-tokens", "300", "--device", args.device])

if __name__ == "__main__": main()
