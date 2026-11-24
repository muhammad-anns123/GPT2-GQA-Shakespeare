"""Download Tiny Shakespeare and train/save the byte-level BPE tokenizer."""
import argparse
from pathlib import Path
from urllib.request import urlopen
from tokenizer import ByteBPETokenizer

DATA_DIR = Path(__file__).parent / "data"
DATA_URL = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--vocab-size", type=int, default=512)
    parser.add_argument("--force-download", action="store_true")
    args = parser.parse_args()
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    text_path = DATA_DIR / "tinyshakespeare.txt"
    if args.force_download or not text_path.exists():
        print(f"Downloading Tiny Shakespeare from {DATA_URL}")
        with urlopen(DATA_URL, timeout=30) as response:
            text = response.read().decode("utf-8")
        text_path.write_text(text, encoding="utf-8")
    text = text_path.read_text(encoding="utf-8")
    split = int(len(text) * 0.9)
    train_text, val_text = text[:split], text[split:]
    tokenizer = ByteBPETokenizer.train(train_text, vocab_size=args.vocab_size)
    tokenizer.save(DATA_DIR / "tokenizer.json")
    (DATA_DIR / "train.txt").write_text(train_text, encoding="utf-8")
    (DATA_DIR / "val.txt").write_text(val_text, encoding="utf-8")
    train_ids = tokenizer.encode(train_text)
    val_ids = tokenizer.encode(val_text)
    import numpy as np
    np.save(DATA_DIR / "train_ids.npy", np.asarray(train_ids, dtype=np.int32))
    np.save(DATA_DIR / "val_ids.npy", np.asarray(val_ids, dtype=np.int32))
    print(f"Characters: {len(text):,} | tokenizer vocab: {tokenizer.vocab_size:,}")
    print(f"Train tokens: {len(train_ids):,} | validation tokens: {len(val_ids):,}")
    print(f"Saved dataset and tokenizer in {DATA_DIR}")

if __name__ == "__main__":
    main()
