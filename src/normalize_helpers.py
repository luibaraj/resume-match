import html
import re
import unicodedata

CONTROL_CHARS = "".join(chr(c) for c in range(0x20) if chr(c) not in ("\n", "\t", "\r"))
CONTROL_RE = re.compile(f"[{re.escape(CONTROL_CHARS)}]")
TAG_RE = re.compile(r"<[^>]+>")
MARKDOWN_RE = re.compile(r"(\[(?P<text>[^]]+)\]\([^)]+\)|`{1,3}[^`]*`{1,3}|[*_]{1,3})")
WEIRD_BULLETS = {
    "\u2022": "-",  # bullet
    "\u2023": "-",  # triangular bullet
    "\u2043": "-",  # hyphen bullet
    "\u2219": "-",  # bullet operator
    "\uf0b7": "-",  # private-use bullet
}


def ensure_utf8(text: bytes | str) -> str:
    """Decode bytes defensively and coerce surrogates to valid UTF-8."""
    if isinstance(text, bytes):
        return text.decode("utf-8", errors="replace")
    return text.encode("utf-8", errors="replace").decode("utf-8")


def strip_markup(text: str) -> str:
    """Drop HTML tags, Markdown artifacts, boilerplate, and control characters."""
    text = TAG_RE.sub(" ", text)  # remove HTML
    text = MARKDOWN_RE.sub(lambda m: m.group("text") or " ", text)  # strip markdown
    text = CONTROL_RE.sub(" ", text)  # remove control chars except whitespace
    return html.unescape(text)


def normalize_unicode(text: str) -> str:
    """Normalize Unicode to NFC, collapsing compatibility equivalents."""
    return unicodedata.normalize("NFC", text)


def normalize_whitespace(text: str) -> str:
    """Collapse repeated whitespace and trim ends."""
    return re.sub(r"\s+", " ", text).strip()


def normalize_bullets(text: str) -> str:
    """Replace odd bullet characters with a plain hyphen."""
    for weird, replacement in WEIRD_BULLETS.items():
        text = text.replace(weird, replacement)
    return text


def remove_template_junk(text: str) -> str:
    """Drop leftover template delimiters ({{ }}, [[]], etc.)."""
    text = re.sub(r"\{\{[^}]+\}\}", " ", text)
    text = re.sub(r"\[\[[^\]]+\]\]", " ", text)
    text = re.sub(r"\|{2,}", " ", text)
    return text


def clean_text_blob(raw: bytes | str) -> str:
    """Full pipeline applying all helpers in a predictable order."""
    text = ensure_utf8(raw)
    text = strip_markup(text)
    text = remove_template_junk(text)
    text = normalize_unicode(text)
    text = normalize_bullets(text)
    text = normalize_whitespace(text)
    return text
