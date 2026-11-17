"""General-purpose structure-aware text splitting with page metadata."""

import re

PAGE_RE = re.compile(r"^\s*\[Page\s*:?[ \t]*(\d+)\]\s*$", re.IGNORECASE)
HEADING_RE = re.compile(r"^\s{0,3}(#{1,6})\s+(.+?)\s*#*\s*$")
CHAPTER_RE = re.compile(
    r"^\s*(?:#{1,6}\s*)?(?:chapter|chap\.?|part|book)\s+"
    r"(one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|"
    r"first|second|third|fourth|fifth|sixth|seventh|eighth|ninth|"
    r"tenth|eleventh|twelfth|[IVXLCDM]+|\d+)\b\s*[:.\-–—]?\s*(.*?)\s*$",
    re.IGNORECASE,
)

WORDS = {
    "one": 1, "first": 1, "two": 2, "second": 2,
    "three": 3, "third": 3, "four": 4, "fourth": 4,
    "five": 5, "fifth": 5, "six": 6, "sixth": 6,
    "seven": 7, "seventh": 7, "eight": 8, "eighth": 8,
    "nine": 9, "ninth": 9, "ten": 10, "tenth": 10,
    "eleven": 11, "eleventh": 11, "twelve": 12, "twelfth": 12,
}
ROMANS = {
    "i": 1, "ii": 2, "iii": 3, "iv": 4, "v": 5, "vi": 6,
    "vii": 7, "viii": 8, "ix": 9, "x": 10, "xi": 11, "xii": 12,
}


def _chapter_heading(line: str):
    """Return (normalized chapter label, chapter number) for explicit headings."""
    clean = line.strip()
    heading_match = HEADING_RE.match(clean)
    if heading_match:
        clean = heading_match.group(2).strip()
    match = CHAPTER_RE.match(clean)
    if not match:
        return None, None

    label = match.group(1)
    label_lower = label.lower()
    if label_lower.isdigit():
        number = int(label_lower)
    else:
        number = WORDS.get(label_lower, ROMANS.get(label_lower, -1))

    title = (match.group(2) or "").strip(" .:-–—")
    kind_match = re.match(r"(?i)(chapter|chap\.?|part|book)", clean)
    kind = kind_match.group(1).title() if kind_match else "Chapter"
    heading = f"{kind} {label.upper() if label_lower in ROMANS else label}"
    if title:
        heading += f": {title}"
    return heading, number


def split_text(text: str, chunk_size: int = 1000, chunk_overlap: int = 150) -> list[str]:
    """Split text on structural boundaries, but do not flush at every page.

    Page markers remain inside the text so chunks can span pages without losing
    page provenance. Each emitted chunk also receives a page metadata prefix
    containing the first page represented in that chunk.
    """
    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than zero.")
    if chunk_overlap < 0 or chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be >= 0 and smaller than chunk_size.")
    if not text or not text.strip():
        return []

    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text).strip()

    # A section is a logical heading-delimited region. Page markers are kept
    # in its body instead of flushing it, preventing tiny page-sized chunks.
    sections = []
    lines = []
    chapter = "Unknown"
    chapter_number = -1
    section = "Unknown"

    def flush():
        nonlocal lines
        body = "\n".join(lines).strip()
        if body:
            sections.append((chapter, chapter_number, section, body))
        lines = []

    for line in text.splitlines():
        heading, number = _chapter_heading(line)
        heading_match = HEADING_RE.match(line.strip())
        if heading:
            flush()
            chapter = heading
            chapter_number = number
            section = heading
            lines.append(line.strip())
        elif heading_match:
            flush()
            section = heading_match.group(2).strip()
            lines.append(line.strip())
        else:
            # Preserve page markers within the content for page-aware chunking.
            lines.append(line)
    flush()

    output = []
    for chap, chap_no, sect, body in sections:
        # Reserve space for the metadata prefix, so actual content stays close
        # to the requested chunk size.
        base_prefix = f"[Chapter: {chap}]\n[ChapterNumber: {chap_no}]\n[Section: {sect}]"
        budget = max(200, chunk_size - len(base_prefix) - 35)
        overlap = min(chunk_overlap, max(0, budget // 4))
        pieces = _split_body(body, budget, overlap)

        current_page = None
        for piece in pieces:
            page = _first_page_in_text(piece)
            if page is not None:
                current_page = page
            prefix = base_prefix
            if current_page is not None:
                prefix += f"\n[Page: {current_page}]"
            output.append(f"{prefix}\n{piece}")
    return output


def _first_page_in_text(text: str):
    for line in text.splitlines():
        match = PAGE_RE.match(line.strip())
        if match:
            return int(match.group(1))
    return None


def _split_body(text: str, size: int, overlap: int) -> list[str]:
    """Split near paragraph/sentence boundaries, with safe character fallback."""
    text = text.strip()
    if not text:
        return []
    if len(text) <= size:
        return [text]

    chunks = []
    start = 0
    text_length = len(text)
    while start < text_length:
        end = min(start + size, text_length)
        if end < text_length:
            floor = start + int(size * 0.55)
            # Prefer paragraph, then line, then sentence boundaries.
            paragraph_boundary = text.rfind("\n\n", floor, end)
            line_boundary = text.rfind("\n", floor, end)
            sentence_boundaries = [
                text.rfind(". ", floor, end),
                text.rfind("? ", floor, end),
                text.rfind("! ", floor, end),
            ]
            if paragraph_boundary > start:
                boundary = paragraph_boundary
                end = boundary + 2
            elif line_boundary > start:
                boundary = line_boundary
                end = boundary + 1
            else:
                boundary = max(sentence_boundaries)
                if boundary > start:
                    end = boundary + 2

        piece = text[start:end].strip()
        if piece:
            chunks.append(piece)
        if end >= text_length:
            break

        next_start = max(start + 1, end - overlap)
        # Avoid starting in the middle of a word.
        while next_start < end and not text[next_start].isspace():
            next_start += 1
        start = min(next_start + 1, end)

    return chunks
