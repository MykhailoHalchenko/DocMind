import re
from pathlib import Path


def extract_text_from_pdf(path: str | Path) -> str:
    try:
        import fitz  # PyMuPDF
    except ImportError:
        raise ImportError("Install PyMuPDF: pip install pymupdf")

    doc = fitz.open(str(path))
    pages = []

    for page in doc:
        text = page.get_text("text")
        text = _clean_page(text)
        if text:
            pages.append(text)

    doc.close()
    return "\n\n".join(pages)


def _clean_page(text: str) -> str:
    text = re.sub(r'\f', '', text)
    text = re.sub(r'^\d+\s*$', '', text, flags=re.MULTILINE)
    text = re.sub(r'^(https?://\S+)$', '', text, flags=re.MULTILINE)
    text = re.sub(r'[ \t]{2,}', ' ', text)
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text.strip()


def pdf_to_document(path: str | Path, source_name: str | None = None) -> dict:
    text = extract_text_from_pdf(path)
    return {
        "text": text,
        "source": source_name or Path(path).name,
        "type": "pdf",
    }