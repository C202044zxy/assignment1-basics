import regex as re
import json
from collections.abc import Iterable, Iterator
import os
from typing import BinaryIO
from multiprocessing import Pool
from functools import lru_cache


def find_chunk_boundaries(
    file: BinaryIO,
    desired_num_chunks: int,
    split_special_token: bytes,
) -> list[int]:
    """
    Chunk the file into parts that can be counted independently.
    May return fewer chunks if the boundaries end up overlapping.
    """
    assert isinstance(split_special_token, bytes), "Must represent special token as a bytestring"

    # Get total file size in bytes
    file.seek(0, os.SEEK_END)
    file_size = file.tell()
    file.seek(0)

    chunk_size = file_size // desired_num_chunks

    # Initial guesses for chunk boundary locations, uniformly spaced
    # Chunks start on previous index, don't include last index
    chunk_boundaries = [i * chunk_size for i in range(desired_num_chunks + 1)]
    chunk_boundaries[-1] = file_size

    mini_chunk_size = 4096  # Read ahead by 4k bytes at a time

    for bi in range(1, len(chunk_boundaries) - 1):
        initial_position = chunk_boundaries[bi]
        file.seek(initial_position)  # Start at boundary guess
        while True:
            mini_chunk = file.read(mini_chunk_size)  # Read a mini chunk

            # If EOF, this boundary should be at the end of the file
            if mini_chunk == b"":
                chunk_boundaries[bi] = file_size
                break

            # Find the special token in the mini chunk
            found_at = mini_chunk.find(split_special_token)
            if found_at != -1:
                chunk_boundaries[bi] = initial_position + found_at
                break
            initial_position += mini_chunk_size

    # Make sure all boundaries are unique, but might be fewer than desired_num_chunks
    return sorted(set(chunk_boundaries))


def process_chunk(args) -> dict[tuple[bytes, ...], int]:
    input_path, start, end, special_tokens = args
    with open(input_path, "rb") as f:
        f.seek(start)
        chunk = f.read(end - start).decode("utf-8", errors="ignore")

    PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
    table = {}
    escaped_tokens = [re.escape(token) for token in special_tokens]
    pattern = "|".join(escaped_tokens)
    pieces = re.split(pattern, chunk)
    for piece in pieces:
        matches = re.finditer(PAT, piece)
        for match in matches:
            key = tuple(bytes([b]) for b in match.group().encode("utf-8"))
            table[key] = table.get(key, 0) + 1
    return table


def pretokenize(input_path: str | os.PathLike, special_tokens: list[str]) -> dict[tuple[bytes, ...], int]:
    frequency_table = {}
    with open(input_path, "rb") as f:
        num_processes = 4
        boundaries = find_chunk_boundaries(f, num_processes, b"<|endoftext|>")

        with Pool(num_processes) as pool:
            results = pool.map(
                process_chunk,
                [(input_path, start, end, special_tokens) for start, end in zip(boundaries[:-1], boundaries[1:])],
            )
        for table in results:
            for key, value in table.items():
                frequency_table[key] = frequency_table.get(key, 0) + value
    return frequency_table


def build_pair_counts(frequency_table: dict[tuple[bytes, ...], int]) -> dict[tuple[bytes, bytes], int]:
    """Build initial pair counts from frequency table."""
    pair_count = {}
    for key, value in frequency_table.items():
        for first, second in zip(key[:-1], key[1:]):
            pair = (first, second)
            pair_count[pair] = pair_count.get(pair, 0) + value
    return pair_count


def merge_optimized(
    frequency_table: dict[tuple[bytes, ...], int],
    pair_count: dict[tuple[bytes, bytes], int],
) -> tuple[dict[tuple[bytes, ...], int], dict[tuple[bytes, bytes], int], tuple[bytes, bytes]]:
    """
    Optimized merge that incrementally updates pair counts.
    Only updates counts for pairs that overlap with the merged pair.
    """
    # Select best pair
    best_pair = max(pair_count.keys(), key=lambda p: (pair_count[p], p))
    merged_token = best_pair[0] + best_pair[1]

    # Remove the merged pair from counts
    del pair_count[best_pair]

    merged_table = {}
    for key, value in frequency_table.items():
        new_key = []
        i = 0
        while i < len(key):
            if i + 1 < len(key) and key[i] == best_pair[0] and key[i + 1] == best_pair[1]:
                # Before merging: ... [i-1] [i] [i+1] [i+2] ...
                # After merging:  ... [i-1] [merged] [i+2] ...

                # Decrement count for pair (key[i-1], key[i]) - it no longer exists
                if i > 0:
                    old_left_pair = (key[i - 1], key[i])
                    pair_count[old_left_pair] = pair_count.get(old_left_pair, 0) - value
                    if pair_count[old_left_pair] <= 0:
                        pair_count.pop(old_left_pair, None)

                # Decrement count for pair (key[i+1], key[i+2]) - it no longer exists
                if i + 2 < len(key):
                    old_right_pair = (key[i + 1], key[i + 2])
                    pair_count[old_right_pair] = pair_count.get(old_right_pair, 0) - value
                    if pair_count[old_right_pair] <= 0:
                        pair_count.pop(old_right_pair, None)

                # Increment count for new pair (key[i-1], merged_token)
                if i > 0:
                    new_left_pair = (key[i - 1], merged_token)
                    pair_count[new_left_pair] = pair_count.get(new_left_pair, 0) + value

                # Increment count for new pair (merged_token, key[i+2])
                if i + 2 < len(key):
                    new_right_pair = (merged_token, key[i + 2])
                    pair_count[new_right_pair] = pair_count.get(new_right_pair, 0) + value

                new_key.append(merged_token)
                i += 2
            else:
                new_key.append(key[i])
                i += 1

        merged_table[tuple(new_key)] = merged_table.get(tuple(new_key), 0) + value

    return (merged_table, pair_count, best_pair)


def train_bpe(
    input_path: str | os.PathLike, vocab_size: int, special_tokens: list[str]
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    vocab = {}
    for i, special_token in enumerate(special_tokens):
        vocab[i] = special_token.encode()
    for i in range(256):
        vocab[i + len(special_tokens)] = bytes([i])
    merge_count = vocab_size - len(special_tokens) - 256
    if merge_count <= 0:
        return tuple(vocab, {})

    merges = []
    frequency_table = pretokenize(input_path, special_tokens)

    # Build initial pair counts once
    pair_count = build_pair_counts(frequency_table)

    for i in range(merge_count):
        frequency_table, pair_count, best_pair = merge_optimized(frequency_table, pair_count)
        vocab[i + len(special_tokens) + 256] = best_pair[0] + best_pair[1]
        merges.append(best_pair)
    return (vocab, merges)


class Tokenizer:
    def __init__(
        self, vocab: dict[int, bytes], merges: list[tuple[bytes, bytes]], special_tokens: list[str] | None = None
    ):
        self._vocab = vocab
        self._merges = merges
        self._special_tokens = special_tokens
        self._vocab_inverse: dict[bytes, int] = {value: key for key, value in vocab.items()}
        # Build merge priority lookup: pair -> priority (lower = higher priority)
        self._merge_priority: dict[tuple[bytes, bytes], int] = {pair: i for i, pair in enumerate(merges)}

    @staticmethod
    @lru_cache(maxsize=1)
    def _gpt2_bytes_to_unicode() -> dict[int, str]:
        """
        Returns a mapping between every possible byte (0..255) and a printable unicode
        representation. This is the same mapping used by the GPT-2 tokenizer.
        """
        bs = (
            list(range(ord("!"), ord("~") + 1))
            + list(range(ord("¡"), ord("¬") + 1))
            + list(range(ord("®"), ord("ÿ") + 1))
        )
        cs = bs[:]
        n = 0
        for b in range(2**8):
            if b not in bs:
                bs.append(b)
                cs.append(2**8 + n)
                n += 1
        return dict(zip(bs, [chr(n) for n in cs]))

    @classmethod
    def from_files(cls, vocab_filepath: str, merges_filepath: str, special_tokens: list[str] | None = None) -> "Tokenizer":
        """
        Load a GPT-2 style tokenizer from:
        - `vocab_filepath`: JSON mapping token_string -> token_id
        - `merges_filepath`: text file with one merge per line: "<token1> <token2>"

        The token strings use GPT-2's reversible byte<->unicode mapping; we decode them
        back into raw bytes for this assignment's tokenizer representation.
        """
        byte_encoder = cls._gpt2_bytes_to_unicode()
        byte_decoder = {v: k for k, v in byte_encoder.items()}

        with open(vocab_filepath, "r", encoding="utf-8") as vocab_f:
            gpt2_vocab: dict[str, int] = json.load(vocab_f)

        vocab: dict[int, bytes] = {
            token_id: bytes([byte_decoder[ch] for ch in token_str]) for token_str, token_id in gpt2_vocab.items()
        }

        merges: list[tuple[bytes, bytes]] = []
        with open(merges_filepath, "r", encoding="utf-8") as merges_f:
            for line in merges_f:
                cleaned = line.rstrip("\n")
                if not cleaned:
                    continue
                parts = cleaned.split(" ")
                if len(parts) != 2:
                    # e.g. GPT-2 merge files often include a header like "#version: 0.2"
                    continue
                t1, t2 = parts
                merges.append(
                    (
                        bytes([byte_decoder[ch] for ch in t1]),
                        bytes([byte_decoder[ch] for ch in t2]),
                    )
                )

        if special_tokens:
            existing = set(vocab.values())
            next_id = (max(vocab.keys()) + 1) if vocab else 0
            for special_token in special_tokens:
                b = special_token.encode("utf-8")
                if b not in existing:
                    vocab[next_id] = b
                    existing.add(b)
                    next_id += 1

        return cls(vocab, merges, special_tokens)

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
            best_priority = float("inf")
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
            tokens = tokens[:best_idx] + [tokens[best_idx] + tokens[best_idx + 1]] + tokens[best_idx + 2 :]
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
        return byte_sequence.decode(errors="replace")

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
