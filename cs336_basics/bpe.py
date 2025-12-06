import regex as re
import json
from collections.abc import Iterable, Iterator

class Tokenizer:
    def __init__(
        self, vocab: dict[int, bytes], merges: list[tuple[bytes, bytes]], special_tokens: list[str] | None = None
    ):
        self._vocab = vocab
        self._merges = merges
        self._special_tokens = special_tokens
        self._vocab_inverse: dict[bytes, int] = {
            value: key for key, value in vocab.items()
        }
        # Build merge priority lookup: pair -> priority (lower = higher priority)
        self._merge_priority: dict[tuple[bytes, bytes], int] = {
            pair: i for i, pair in enumerate(merges)
        }

    def _pretokenize(self, text: str) -> list[bytes]:
        PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
        if not self._special_tokens:
            matches = re.findall(PAT, text)
            return [match.encode() for match in matches]

        sorted_tokens = sorted(self._special_tokens, key=len, reverse=True)
        escaped_tokens = [re.escape(token) for token in sorted_tokens]
        pattern = "(" + "|".join(escaped_tokens) + ")"
        pieces = re.split(pattern, text)
        results = []
        for piece in pieces:
            if not piece:
                continue
            if piece in self._special_tokens:
                results.append(piece.encode()) 
            else:
                matches = re.findall(PAT, piece)
                results.extend([match.encode() for match in matches])
        return results                

    def _apply_merges(self, chunk: bytes) -> list[bytes]:
        tokens = [bytes([b]) for b in chunk]
        while len(tokens) > 1:
            # Find the pair with lowest priority (highest precedence) that exists
            best_idx = -1
            best_priority = float('inf')
            for i in range(len(tokens) - 1):
                pair = (tokens[i], tokens[i + 1])
                if pair in self._merge_priority:
                    priority = self._merge_priority[pair]
                    if priority < best_priority:
                        best_priority = priority
                        best_idx = i
            if best_idx == -1:
                break  # No more merges possible
            # Apply the best merge
            tokens = tokens[:best_idx] + [tokens[best_idx] + tokens[best_idx + 1]] + tokens[best_idx + 2:]
        return tokens

    def encode(self, text: str) -> list[int]:
        chunks = self._pretokenize(text)
        ids = []
        for chunk in chunks:
            if self._special_tokens and chunk.decode() in self._special_tokens:
                ids.append(self._vocab_inverse[chunk])
            else:
                tokens = self._apply_merges(chunk)
                ids.extend([self._vocab_inverse[token] for token in tokens])
        return ids
    
    def decode(self, ids: list[int]) -> str:
        byte_sequence = b"".join(self._vocab[id] for id in ids)
        return byte_sequence.decode(errors='replace')
    
    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        buffer = ""
        for text in iterable:
            buffer += text
            chunks = self._pretokenize(buffer)
            for chunk in chunks[:-1]:
                if self._special_tokens and chunk.decode() in self._special_tokens:
                    yield self._vocab_inverse[chunk]
                else:
                    tokens = self._apply_merges(chunk)
                    for token in tokens:
                        yield self._vocab_inverse[token]
            if chunks:
                buffer = chunks[-1].decode()
            else:
                buffer = ""
        if buffer:
            chunks = self._pretokenize(buffer)
            for chunk in chunks:
                if self._special_tokens and chunk.decode() in self._special_tokens:
                    yield self._vocab_inverse[chunk]
                else:
                    tokens = self._apply_merges(chunk)
                    for token in tokens:
                        yield self._vocab_inverse[token]