# GPT-2-style Shakespeare LM with Grouped-Query Attention

A small, self-contained educational decoder-only Transformer inspired by Andrej Karpathy's GPT code-along. It includes **byte-level BPE**, **Grouped-Query Attention (GQA)**, causal masking, GPT-style learned position embeddings, tied token embeddings, training, checkpointing, sampling, and an interactive terminal completion loop.

> This is a language-modeling project, not a ChatGPT replacement. It is trained on Tiny Shakespeare and learns Shakespeare-like continuation, not general instruction following.

## 1. Requirements

- Python 3.10+
- PyTorch 2.2+ (CPU works; CUDA/MPS is faster)
- NumPy 1.24+

## 2. Install

```bash
python -m venv .venv
# macOS/Linux
source .venv/bin/activate
# Windows PowerShell: .venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Install a PyTorch build suitable for your hardware if the default pip wheel does not support your GPU: https://pytorch.org/get-started/locally/.

## 3. Prepare Tiny Shakespeare + BPE

```bash
python prepare.py --vocab-size 512
```

This downloads the 1.1 MB Tiny Shakespeare text from Karpathy's public `char-rnn` repository, creates a chronological 90/10 train/validation split, trains the byte-level BPE tokenizer on the training split only, and saves encoded arrays under `data/`.

Internet access is required only for the first dataset download. If you already have the official `input.txt`, put it at `data/tinyshakespeare.txt` and run `python prepare.py` (no download will occur).

## 4. Train

Quick smoke run:

```bash
python train.py --max-iters 20 --eval-interval 10 --batch-size 8 --block-size 128 --n-layer 2 --n-head 4 --n-kv-head 2 --n-embd 64
```

Default training run:

```bash
python train.py
```

The default is a compact model (4 layers, 4 query heads, 2 KV heads, 128 embedding dimensions, context 256). It saves the best validation-loss checkpoint to `checkpoints/best.pt`. CPU training is supported but can be slow; use smaller values for a laptop CPU and more iterations/a GPU for better quality.

Useful options: `--device auto|cpu|cuda|mps`, `--max-iters`, `--batch-size`, `--block-size`, `--n-layer`, `--n-head`, `--n-kv-head`, `--n-embd`, `--learning-rate`.

## 5. Generate text

```bash
python generate.py --prompt "ROMEO:" --max-new-tokens 400 --temperature 0.8 --top-k 50
```

Sampling controls:
- Lower temperature (e.g. 0.6): more conservative output.
- Higher temperature (e.g. 1.0): more varied output.
- Lower top-k (e.g. 20): restricts sampling to the most likely next tokens.

## 6. Interactive terminal

```bash
python chat.py
```

Enter a Shakespeare-style prompt; type `/exit` to quit. This interface is intentionally a continuation interface. The model is not fine-tuned to obey questions or tools.

## 7. Tests

```bash
pip install pytest
pytest -q
```

## Architecture notes

- **GQA:** queries use `n_head` heads while keys/values use `n_kv_head` heads; KV heads are repeated across query-head groups before scaled dot-product attention.
- **Causality:** PyTorch scaled-dot-product attention uses `is_causal=True` so tokens cannot see future positions.
- **BPE:** UTF-8 bytes are the 256 base tokens; learned adjacent-pair merges extend the vocabulary. A regex pre-tokenizer preserves whitespace and punctuation boundaries. This is a compact educational byte-BPE implementation, not a drop-in reproduction of OpenAI's exact GPT-2 tokenizer regex/merge table.
- **GPT-2 style:** decoder-only blocks, learned positional embeddings, pre-layer normalization, GELU MLP, residual paths, and tied input/output token embeddings.

## Troubleshooting

- `Dataset/tokenizer missing`: run `python prepare.py` first.
- `No checkpoint`: run `python train.py` first.
- CUDA unavailable: use `--device cpu` or `--device auto`.
- Out-of-memory: lower batch size, block size, layer count, or embedding size.
- Poor or repetitive output: train longer, use the default model size or larger, and experiment with temperature/top-k.

## Dataset attribution

Tiny Shakespeare (about 1.1 MB; roughly 40,000 lines) is distributed from Andrej Karpathy's `char-rnn` repository:
https://github.com/karpathy/char-rnn/blob/master/data/tinyshakespeare/input.txt

This project is an independent educational implementation and is not affiliated with OpenAI or Andrej Karpathy.
