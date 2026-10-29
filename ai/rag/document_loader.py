
from pathlib import Path
import logging
import re

import pymupdf

logger = logging.getLogger("JustAnotherBot.RAG")

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DOCUMENTS_DIR = PROJECT_ROOT / "documents"

SUPPORTED_EXTENSIONS = {".pdf", ".txt"}

# Common chapter-heading formats.
CHAPTER_PATTERN = re.compile(
    r"^\s*(?:chapter|chap\.?)\s+"
    r"(?:\d+|[IVXLCDM]+|one|two|three|four|five|six|seven|eight|nine|ten)"
    r"(?:\s*[:.\-–—]\s*.*|\s+.*)?$",
    re.IGNORECASE,
)

# Common numbered section headings, such as:
# 1 Introduction
# 2.1 Data Collection
# 3.2.1 Evaluation
NUMBERED_HEADING_PATTERN = re.compile(
    r"^\s*\d+(?:\.\d+){0,4}\.?\s+[A-Z][^\n]{0,120}$"
)

# Common standalone headings.
STANDALONE_HEADING_PATTERN = re.compile(
    r"^\s*(?:"
    r"abstract|introduction|background|"
    r"literature review|related work|"
    r"methodology|methods|materials and methods|"
    r"results|discussion|conclusion|conclusions|"
    r"references|bibliography|appendix|"
    r"acknowledgements|acknowledgments"
    r")\s*:?\s*$",
    re.IGNORECASE,
)


def normalize_text(text: str) -> str:
    """Normalize whitespace while preserving paragraphs and page markers."""
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = text.replace("\x00", "")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)

    lines = []

    for line in text.splitlines():
        stripped = line.strip()

        if not stripped:
            lines.append("")
            continue

        # Preserve existing page markers.
        if re.fullmatch(r"\[Page\s+\d+\]", stripped, re.IGNORECASE):
            lines.append(stripped)
            continue

        # Preserve existing Markdown headings.
        if stripped.startswith("#"):
            lines.append(stripped)
            continue

        if CHAPTER_PATTERN.match(stripped):
            lines.append(f"\n## {stripped}\n")
        elif STANDALONE_HEADING_PATTERN.match(stripped):
            lines.append(f"\n### {stripped}\n")
        elif NUMBERED_HEADING_PATTERN.match(stripped):
            lines.append(f"\n### {stripped}\n")
        else:
            lines.append(stripped)

    text = "\n".join(lines)
    text = re.sub(r"\n{3,}", "\n\n", text)

    return text.strip()


def load_pdf(file_path: Path) -> str:
    """Extract PDF text while preserving page boundaries."""
    pages = []

    with pymupdf.open(file_path) as document:
        for page_number, page in enumerate(document, start=1):
            page_text = page.get_text("text").strip()

            if not page_text:
                continue

            page_text = normalize_text(page_text)

            if page_text:
                pages.append(f"[Page {page_number}]\n{page_text}")

    return "\n\n".join(pages)


def load_txt(file_path: Path) -> str:
    """Read and normalize a UTF-8 text file."""
    text = file_path.read_text(encoding="utf-8-sig")
    return normalize_text(text)


def detect_structure(text: str) -> dict:
    """Detect basic document structure without assuming every file has chapters."""
    chapter_headings = []
    section_headings = []
    page_markers = []

    current_chapter = None

    for line in text.splitlines():
        stripped = line.strip()

        page_match = re.fullmatch(
            r"\[Page\s+(\d+)\]",
            stripped,
            re.IGNORECASE,
        )

        if page_match:
            page_markers.append(int(page_match.group(1)))
            continue

        if stripped.startswith("## "):
            heading = stripped[3:].strip()

            if CHAPTER_PATTERN.match(heading):
                current_chapter = heading
                chapter_headings.append(heading)
            else:
                section_headings.append(heading)

        elif stripped.startswith("### "):
            section_headings.append(stripped[4:].strip())

    return {
        "chapter_count": len(chapter_headings),
        "chapter_headings": chapter_headings,
        "section_headings": section_headings,
        "page_count": len(set(page_markers)),
    }


def load_document(file_path: str | Path) -> dict:
    """Load one supported document and return its text and structure metadata."""
    file_path = Path(file_path).expanduser().resolve()

    if not file_path.is_file():
        raise FileNotFoundError(f"Document not found: {file_path}")

    extension = file_path.suffix.lower()

    if extension not in SUPPORTED_EXTENSIONS:
        raise ValueError(
            f"Unsupported file type: {extension}. "
            "Supported formats: PDF and TXT."
        )

    if extension == ".pdf":
        text = load_pdf(file_path)
    else:
        text = load_txt(file_path)

    if not text.strip():
        raise ValueError(
            f"No extractable text found in {file_path.name}. "
            "The PDF might be scanned or image-only."
        )

    structure = detect_structure(text)

    logger.info(
        "Loaded %s: %s characters, %s chapter headings, "
        "%s section headings, %s page markers.",
        file_path.name,
        len(text),
        structure["chapter_count"],
        len(structure["section_headings"]),
        structure["page_count"],
    )

    return {
        "source": file_path.name,
        "text": text,
        "structure": structure,
    }


def load_all_documents() -> list[dict]:
    """Load supported documents from the documents directory."""
    DOCUMENTS_DIR.mkdir(parents=True, exist_ok=True)

    documents = []

    for file_path in sorted(DOCUMENTS_DIR.iterdir()):
        if not file_path.is_file():
            continue

        if file_path.suffix.lower() not in SUPPORTED_EXTENSIONS:
            continue

        try:
            documents.append(load_document(file_path))
        except Exception:
            logger.exception(
                "Could not load document: %s",
                file_path.name,
            )

    return documents
