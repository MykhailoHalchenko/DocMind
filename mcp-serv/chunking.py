from dataclasses import dataclass

from config import settings


@dataclass
class TextChunk:
    text: str
    index: int
    start_char: int
    end_char: int


def split_into_chunks(
    text: str,
    chunk_size: int | None = None,
    chunk_overlap: int | None = None,
) -> list[TextChunk]:
    size = chunk_size or settings.chunk_size
    overlap = chunk_overlap or settings.chunk_overlap

    if overlap >= size:
        raise ValueError("chunk_overlap must be less than chunk_size")

    words = text.split()
    chunks: list[TextChunk] = []
    step = size - overlap
    index = 0

    for start in range(0, len(words), step):
        window = words[start : start + size]
        if not window:
            break

        chunk_text = " ".join(window)

        start_char = len(" ".join(words[:start])) + (1 if start > 0 else 0)
        end_char = start_char + len(chunk_text)

        chunks.append(
            TextChunk(
                text=chunk_text,
                index=index,
                start_char=start_char,
                end_char=end_char,
            )
        )
        index += 1

        if start + size >= len(words):
            break

    return chunks


def chunk_documents(
    documents: list[dict],
    text_field: str = "text",
) -> list[dict]:
    result: list[dict] = []

    for doc in documents:
        raw_text = doc.get(text_field, "")
        if not raw_text:
            continue

        chunks = split_into_chunks(raw_text)
        meta = {k: v for k, v in doc.items() if k != text_field}

        for chunk in chunks:
            result.append(
                {
                    "text": chunk.text,
                    "chunk_index": chunk.index,
                    "source_start_char": chunk.start_char,
                    "source_end_char": chunk.end_char,
                    **meta,
                }
            )

    return result