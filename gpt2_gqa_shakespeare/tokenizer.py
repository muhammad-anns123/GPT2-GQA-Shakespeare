"""Small, dependency-free byte-level BPE tokenizer.

This is GPT-2-style byte BPE (UTF-8 bytes + learned merges), with a lightweight
regex pre-tokenizer. It is deliberately self-contained for reproducibility.
"""
from __future__ import annotations
import json
import re
from collections import Counter
from pathlib import Path
from typing import Iterable

_PRETOKEN_RE = re.compile(r"\s+|[A-Za-z]+(?:'[A-Za-z]+)?|[0-9]+|[^\w\s]+", re.UNICODE)


class ByteBPETokenizer:
    def __init__(self, merges: list[tuple[int, int]] | None = None):
        self.merges = list(merges or [])
        self.byte_vocab = [bytes([i]) for i in range(256)]
        for a, b in self.merges:
            self.byte_vocab.append(self.byte_vocab[a] + self.byte_vocab[b])
        self._merge_rank = {pair: rank for rank, pair in enumerate(self.merges)}

    @property
    def vocab_size(self) -> int:
        return 256 + len(self.merges)

    @staticmethod
    def _pieces(text: str) -> Iterable[bytes]:
        for match in _PRETOKEN_RE.finditer(text):
            yield match.group(0).encode("utf-8")

    @classmethod
    def train(cls, text: str, vocab_size: int = 512, min_frequency: int = 2) -> "ByteBPETokenizer":
        if vocab_size < 256:
            raise ValueError("vocab_size must be at least 256 for byte-level BPE")
        counts = Counter(cls._pieces(text))
        sequences: dict[tuple[int, ...], int] = {}
        for piece, count in counts.items():
            sequences[tuple(piece)] = count
        merges: list[tuple[int, int]] = []
        for _ in range(vocab_size - 256):
            pair_counts: Counter[tuple[int, int]] = Counter()
            for seq, freq in sequences.items():
                for pair in zip(seq, seq[1:]):
                    pair_counts[pair] += freq
            if not pair_counts:
                break
            pair, frequency = pair_counts.most_common(1)[0]
            if frequency < min_frequency:
                break
            new_id = 256 + len(merges)
            merges.append(pair)
            updated: dict[tuple[int, ...], int] = {}
            for seq, freq in sequences.items():
                out = []
                i = 0
                while i < len(seq):
                    if i + 1 < len(seq) and (seq[i], seq[i + 1]) == pair:
                        out.append(new_id)
                        i += 2
                    else:
                        out.append(seq[i])
                        i += 1
                key = tuple(out)
                updated[key] = updated.get(key, 0) + freq
            sequences = updated
        return cls(merges)

    def encode(self, text: str) -> list[int]:
        output: list[int] = []
        for piece in self._pieces(text):
            ids = list(piece)
            # Apply merges in learned priority order; pieces are short, so this
            # transparent implementation is fast enough for this teaching model.
            for rank, pair in enumerate(self.merges):
                if len(ids) < 2:
                    break
                merged_id = 256 + rank
                i, new_ids = 0, []
                changed = False
                while i < len(ids):
                    if i + 1 < len(ids) and (ids[i], ids[i + 1]) == pair:
                        new_ids.append(merged_id)
                        i += 2
                        changed = True
                    else:
                        new_ids.append(ids[i])
                        i += 1
                if changed:
                    ids = new_ids
            output.extend(ids)
        return output

    def decode(self, ids: Iterable[int]) -> str:
        data = bytearray()
        for token_id in ids:
            if not 0 <= int(token_id) < self.vocab_size:
                raise ValueError(f"token id {token_id} is outside vocabulary")
            data.extend(self.byte_vocab[int(token_id)])
        return data.decode("utf-8", errors="replace")

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"format": "byte-bpe-v1", "merges": self.merges}, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "ByteBPETokenizer":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        if payload.get("format") != "byte-bpe-v1":
            raise ValueError("Unsupported tokenizer format")
        return cls([tuple(pair) for pair in payload["merges"]])
