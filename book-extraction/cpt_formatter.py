"""CPT Formatter for Sliding Window Chunking.

Takes text and splits it into overlapping token windows.
Short final chunks are padded backwards (merged with prev chunk tail)
instead of being discarded. Zero facts lost.
"""

def sliding_window_chunk(
    text: str,
    tokenizer,
    target_tokens: int = 2048,
    stride_tokens: int = 256,
    pad_up_min_tokens: int = 512,
) -> list[str]:
    """
    Splits text into overlapping token windows.
    Short final chunks are padded backwards (merged with prev chunk tail)
    instead of being discarded. Zero facts lost.
    """
    token_ids = tokenizer.encode(text, add_special_tokens=False)
    chunks = []
    start = 0

    while start < len(token_ids):
        end = min(start + target_tokens, len(token_ids))
        chunk_ids = token_ids[start:end]

        # Pad-up: if this is the final chunk and it's too short,
        # extend backwards by prepending from the previous chunk's tail
        if end == len(token_ids) and len(chunk_ids) < pad_up_min_tokens:
            needed = pad_up_min_tokens - len(chunk_ids)
            pad_start = max(0, start - needed)
            chunk_ids = token_ids[pad_start:end]

        chunks.append(tokenizer.decode(chunk_ids))
        start += target_tokens - stride_tokens
        if end == len(token_ids):
            break

    return chunks
