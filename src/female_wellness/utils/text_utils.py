"""Text utility functions for cleaning, normalization, hashing, and sentence splitting."""

import hashlib
import re
import unicodedata

# Common abbreviations that should not trigger sentence breaks
_ABBREVIATIONS = (
    r"(?:[A-Za-z]\.){2,}",  # e.g., i.e., e.t.c.
    r"Dr\.",
    r"Prof\.",
    r"vs\.",
    r"Fig\.",
    r"approx\.",
    r"dept\.",
    r"min\.",
    r"max\.",
    r"no\.",
    r"vol\.",
    r"al\.",  # et al.
)

_ABBREV_PATTERN = re.compile(rf"\b(?:{'|'.join(_ABBREVIATIONS)})$", re.IGNORECASE)

# Line-break hyphenation pattern (e.g. "anti-\ninflammatory" -> "anti-inflammatory")
_HYPHENATED_LINE_BREAK = re.compile(r"(\w+)-\s*\n\s*(\w+)")


def clean_text(text: str) -> str:
    """Normalize and clean raw extracted document text.

    Performs:
    1. Unicode NFKC normalization.
    2. Replacement of non-standard whitespaces (non-breaking space, em-space).
    3. Reconnection of hyphenated words split across line breaks.
    4. Collapse of redundant horizontal whitespace while preserving paragraph breaks.
    5. Stripping of control characters.

    Args:
        text: Raw extracted text string.

    Returns:
        Cleaned, normalized text.
    """
    if not text:
        return ""

    # 1. Unicode normalization
    text = unicodedata.normalize("NFKC", text)

    # 2. Normalize whitespace characters
    text = text.replace("\xa0", " ").replace("\u200b", "").replace("\r\n", "\n")

    # 3. Fix hyphenated line breaks
    text = _HYPHENATED_LINE_BREAK.sub(r"\1\2", text)

    # 4. Remove unwanted non-printable control characters (keep \n and \t)
    text = "".join(ch for ch in text if ch in "\n\t" or unicodedata.category(ch)[0] != "C")

    # 5. Clean consecutive horizontal spaces on each line
    lines = [re.sub(r"[ \t]+", " ", line).strip() for line in text.split("\n")]

    # 6. Collapse excessive empty lines (maximum 2 consecutive newlines)
    cleaned = re.sub(r"\n{3,}", "\n\n", "\n".join(lines))

    return cleaned.strip()


def compute_content_hash(content: str | bytes) -> str:
    """Compute a deterministic SHA-256 hash of text or binary content.

    Used for duplicate document detection and tracking changes.

    Args:
        content: String or bytes to hash.

    Returns:
        Hexadecimal SHA-256 digest string.
    """
    if isinstance(content, str):
        content = content.encode("utf-8")
    return hashlib.sha256(content).hexdigest()


def split_into_sentences(text: str) -> list[str]:
    """Split text into sentences while respecting common abbreviations and boundaries.

    Args:
        text: Input paragraph or text block.

    Returns:
        List of non-empty sentence strings.
    """
    text = text.strip()
    if not text:
        return []

    # Fast check: if short and no punctuation, return as single sentence
    if not re.search(r"[.!?]", text):
        return [text]

    # Split tentatively on sentence-ending punctuation followed by whitespace and a capital letter or start
    raw_splits = re.split(r"(?<=[.!?])\s+(?=[A-Z0-9\"'(\[])", text)
    sentences: list[str] = []
    buffer = ""

    for segment in raw_splits:
        segment = segment.strip()
        if not segment:
            continue

        if buffer:
            combined = f"{buffer} {segment}"
        else:
            combined = segment

        # Check if the combined ends with an abbreviation
        match = _ABBREV_PATTERN.search(combined)
        if match:
            # Keep in buffer to avoid false break
            buffer = combined
        else:
            sentences.append(combined)
            buffer = ""

    if buffer:
        sentences.append(buffer)

    return [s for s in sentences if s]

