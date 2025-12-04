import os
from typing import BinaryIO
import regex as re
from multiprocessing import Pool


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


def merge(frequency_table: dict[tuple[bytes, ...], int]) -> tuple[dict[tuple[bytes, ...], int], tuple[bytes, bytes]]:
    # count adjacent pairs
    pair_count = {}
    for key, value in frequency_table.items():
        for first, second in zip(key[:-1], key[1:]):
            pair = tuple([first, second])
            pair_count[pair] = pair_count.get(pair, 0) + value
    # select the pair with max weight, tiebreak by max lexicographic order
    best_pair = max(pair_count.keys(), key=lambda p: (pair_count[p], p))
    merged_table = {}
    for key, value in frequency_table.items():
        new_key = []
        i = 0
        while i < len(key):
            if i + 1 < len(key) and key[i] == best_pair[0] and key[i + 1] == best_pair[1]:
                new_key.append(best_pair[0] + best_pair[1])
                i += 2
            else:
                new_key.append(key[i])
                i += 1
        merged_table[tuple(new_key)] = merged_table.get(tuple(new_key), 0) + value
    return (merged_table, best_pair)


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
    for i in range(merge_count):
        frequency_table, best_pair = merge(frequency_table)
        vocab[i + len(special_tokens) + 256] = best_pair[0] + best_pair[1]
        merges.append(best_pair)
    return (vocab, merges)
