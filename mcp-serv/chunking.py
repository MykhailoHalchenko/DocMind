import re
import logging
from dataclasses import dataclass, field

from config import settings

_logger = logging.getLogger(__name__)


@dataclass
class TextChunk:
    text: str
    index: int
    start_char: int
    end_char: int
    entities: list[dict] = field(default_factory=list)


def clean_pdf_text(text: str) -> str:
    text = re.sub(r'\f', '\n', text)
    text = re.sub(r'(\d+\s*\n)', '', text)
    text = re.sub(r'\n{3,}', '\n\n', text)
    text = re.sub(r'[ \t]+', ' ', text)
    text = re.sub(r'^\s*(fig\.|figure|table)\s*\d+.*$', '', text, flags=re.IGNORECASE | re.MULTILINE)
    return text.strip()


def extract_entities(text: str) -> list[dict]:
    try:
        import spacy
        nlp = spacy.load("en_core_web_sm")
        doc = nlp(text)
        return [
            {"text": ent.text, "label": ent.label_}
            for ent in doc.ents
            if ent.label_ in ("PERSON", "ORG", "DATE", "GPE", "WORK_OF_ART")
        ]
    except Exception as e:
        _logger.warning(f"NER extraction failed: {e}")
        return []


def split_into_chunks(
    text: str,
    chunk_size: int | None = None,
    chunk_overlap: int | None = None,
    use_ner: bool = False,
) -> list[TextChunk]:
    size = chunk_size or settings.chunk_size
    overlap = chunk_overlap or settings.chunk_overlap

    if overlap >= size:
        raise ValueError("chunk_overlap must be less than chunk_size")

    words = text.split()
    chunks: list[TextChunk] = []
    step = size - overlap

    for i, start in enumerate(range(0, len(words), step)):
        window = words[start: start + size]
        if not window:
            break

        chunk_text = " ".join(window)
        start_char = len(" ".join(words[:start])) + (1 if start > 0 else 0)
        end_char = start_char + len(chunk_text)

        entities = extract_entities(chunk_text) if use_ner else []
        chunks.append(TextChunk(text=chunk_text, index=i, start_char=start_char, end_char=end_char, entities=entities))

        if start + size >= len(words):
            break

    return chunks


def chunk_documents(
    documents: list[dict],
    text_field: str = "text",
    use_ner: bool = False,
    clean_text: bool = False,
) -> list[dict]:
    result: list[dict] = []

    for doc in documents:
        raw_text = doc.get(text_field, "")
        if not raw_text or not raw_text.strip():
            continue

        if clean_text:
            raw_text = clean_pdf_text(raw_text)

        if not raw_text.strip():
            continue

        chunks = split_into_chunks(raw_text, use_ner=use_ner)
        meta = {k: v for k, v in doc.items() if k != text_field}

        for chunk in chunks:
            if chunk.text and chunk.text.strip():
                result.append({
                    "text": chunk.text,
                    "chunk_index": chunk.index,
                    "source_start_char": chunk.start_char,
                    "source_end_char": chunk.end_char,
                    "entities": chunk.entities,
                    **meta,
                })

    return result