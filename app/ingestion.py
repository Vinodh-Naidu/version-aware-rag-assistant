
import hashlib
import re
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path

from pypdf import PdfReader


CHUNK_SIZE = 1000
CHUNK_OVERLAP = 150
MAX_FILE_SIZE = 5 * 1024 * 1024  # 5 MB


class DocumentError(ValueError):
    """Raised when a document cannot be processed safely."""


@dataclass
class ProcessedDocument:
    filename: str
    content_hash: str
    text: str
    chunks: list[dict]


def normalize_text(text: str) -> str:
    """Normalize extracted text before hashing and chunking."""
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def calculate_hash(text: str) -> str:
    """Hash normalized text so irrelevant formatting changes are ignored."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def extract_text(filename: str, content: bytes) -> str:
    extension = Path(filename).suffix.lower()

    if extension not in {".txt", ".pdf"}:
        raise DocumentError("Only PDF and TXT files are supported.")

    if not content:
        raise DocumentError("The uploaded file is empty.")

    if len(content) > MAX_FILE_SIZE:
        raise DocumentError("Maximum file size is 5 MB.")

    if extension == ".txt":
        try:
            text = content.decode("utf-8-sig")
        except UnicodeDecodeError as exc:
            raise DocumentError(
                "TXT files must use UTF-8 encoding."
            ) from exc

    else:
        try:
            reader = PdfReader(BytesIO(content), strict=True)

            if reader.is_encrypted:
                raise DocumentError(
                    "Encrypted PDFs are not supported."
                )

            pages = [
                page.extract_text() or ""
                for page in reader.pages
            ]
            text = "\n\n".join(pages)

        except DocumentError:
            raise
        except Exception as exc:
            raise DocumentError(
                "Unable to read the PDF. Check that it is valid."
            ) from exc

    text = normalize_text(text)

    if not text:
        raise DocumentError(
            "No extractable text found. Scanned PDFs may require OCR."
        )

    return text


def chunk_text(
        text: str,
        chunk_size: int = CHUNK_SIZE,
        overlap: int = CHUNK_OVERLAP,
) -> list[dict]:
    """Split text into overlapping chunks with character offsets."""
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive.")

    if overlap < 0 or overlap >= chunk_size:
        raise ValueError(
            "overlap must be >= 0 and smaller than chunk_size."
        )

    chunks = []
    start = 0
    index = 0

    while start < len(text):
        end = min(start + chunk_size, len(text))

        # Prefer a nearby word boundary without producing tiny chunks.
        if end < len(text):
            boundary = text.rfind(" ", start + chunk_size // 2, end)
            if boundary > start:
                end = boundary

        chunk = text[start:end].strip()

        if chunk:
            chunks.append({
                "chunk_index": index,
                "text": chunk,
                "start_char": start,
                "end_char": end,
            })
            index += 1

        if end >= len(text):
            break

        start = end - overlap

    return chunks


def process_document(
        filename: str,
        content: bytes,
) -> ProcessedDocument:
    text = extract_text(filename, content)

    return ProcessedDocument(
        filename=Path(filename).name,
        content_hash=calculate_hash(text),
        text=text,
        chunks=chunk_text(text),
    )
