import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
from model import GPTConfig, GPTLanguageModel
from tokenizer import ByteBPETokenizer

def test_tokenizer_roundtrip():
    text = "Hello, world!\nTo be, or not to be."
    tok = ByteBPETokenizer.train(text * 5, vocab_size=280)
    assert tok.decode(tok.encode(text)) == text

def test_model_forward_and_generation():
    cfg = GPTConfig(vocab_size=300, block_size=32, n_layer=2, n_head=4, n_kv_head=2, n_embd=32, dropout=0.0)
    model = GPTLanguageModel(cfg)
    x = torch.randint(0, 300, (2, 12))
    logits, loss = model(x, x)
    assert logits.shape == (2, 12, 300)
    assert torch.isfinite(loss)
    generated = model.generate(x[:1, :4], 3, top_k=10)
    assert generated.shape == (1, 7)
