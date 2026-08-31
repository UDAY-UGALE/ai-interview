"""Turn an uploaded document into plain text -- once, at upload time.

This is deliberately NOT on the interview hot path. A question is answered
in ~1.4-2.2s end to end (see ARCHITECTURE.md) and parsing a PDF takes a
noticeable slice of that, so extraction runs when the user hands us the
file, the resulting text is what goes into the session store, and every
subsequent question reads a string that is already plain text.

Three formats, because those are the three ways people actually have their
project notes: a PDF export, a Word document, or text they paste.

DOCX is read with the standard library rather than python-docx. A .docx is
an OPC zip whose word/document.xml holds the whole body, and pulling
paragraphs and table rows out of it is about forty lines -- against a new
dependency on the deployed box, an extra import to fail at runtime, and a
second thing to keep pinned. pypdf is already a dependency (the existing
resume upload uses it), so PDF keeps using it.

Every failure here is a *user* error the user can fix -- wrong file type,
scanned PDF, empty document, corrupt file -- so each one raises
DocumentExtractionError carrying a message meant to be read by a person,
which the route turns into a 400. Nothing in this module can affect a
running interview: it is called from the upload routes only.
"""

from __future__ import annotations

import io
import re
import zipfile
from dataclasses import dataclass
from xml.etree import ElementTree


# WordprocessingML namespace. Every element in document.xml is in it.
_W = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"

SUPPORTED_EXTENSIONS: tuple[str, ...] = (".pdf", ".docx", ".txt", ".md", ".text")
# Hard ceiling on what we keep from ONE document. Nothing in the pipeline
# sends this much to a model -- the context builder budgets far below it --
# but the stored value travels through the session store (and Redis, on a
# multi-instance deployment), so an unbounded 200-page PDF has no business
# sitting in it. Generous enough for real project notes, and the trim is
# reported back to the user rather than done silently.
MAX_STORED_CHARACTERS = 60_000

# Below this, whatever came out of the file is not a document -- it is a
# stray header line or an empty template, and treating it as interview
# context would be worse than having none.
MIN_USEFUL_CHARACTERS = 20


class DocumentExtractionError(ValueError):
    """A document we could not turn into usable text. The message is shown
    to the user verbatim, so it says what to do about it."""


@dataclass(frozen=True, slots=True)
class ExtractedDocument:
    filename: str
    kind: str  # "pdf" | "docx" | "text"
    text: str
    truncated: bool = False

    @property
    def characters(self) -> int:
        return len(self.text)


def extract_document_text(filename: str, raw: bytes, *, max_bytes: int) -> ExtractedDocument:
    """Validate an uploaded file and return its text.

    Raises DocumentExtractionError for anything the user can act on: an
    empty file, an unsupported extension, an oversized upload, a corrupt
    document, or a PDF that is really a scan with no text layer.
    """
    name = (filename or "").strip() or "document"
    lowered = name.lower()

    if not raw:
        raise DocumentExtractionError(
            f"{name} is empty (0 bytes). Nothing to add as interview context."
        )
    if len(raw) > max_bytes:
        raise DocumentExtractionError(
            f"{name} is {len(raw) / 1_000_000:.1f}MB, over the "
            f"{max_bytes / 1_000_000:.0f}MB limit. Project notes are text -- if this "
            "is a slide deck or a document full of screenshots, paste the text instead."
        )

    if lowered.endswith(".pdf"):
        kind, text = "pdf", _extract_pdf(name, raw)
    elif lowered.endswith(".docx"):
        kind, text = "docx", _extract_docx(name, raw)
    elif lowered.endswith((".txt", ".md", ".text")):
        kind, text = "text", _extract_plain_text(name, raw)
    elif lowered.endswith(".doc"):
        # Worth its own message: .doc is the pre-2007 binary format, and a
        # user who has one will otherwise read "unsupported" and give up.
        raise DocumentExtractionError(
            f"{name} is a legacy .doc file, which cannot be read directly. Open it "
            "in Word and save as .docx or PDF, then upload that."
        )
    else:
        raise DocumentExtractionError(
            f"{name} is not a supported file type. Upload PDF, DOCX, or TXT, or "
            "paste the text directly."
        )

    text = _normalize_whitespace(text)

    if len(text) < MIN_USEFUL_CHARACTERS:
        detail = (
            "A PDF that is a scan or an export of images has no text layer to read "
            "-- retype or paste the parts that matter."
            if kind == "pdf"
            else "Check the file actually has content, or paste the text instead."
        )
        raise DocumentExtractionError(
            f"Almost no text came out of {name} ({len(text)} characters). {detail}"
        )

    truncated = len(text) > MAX_STORED_CHARACTERS
    if truncated:
        text = text[:MAX_STORED_CHARACTERS].rstrip()

    return ExtractedDocument(filename=name, kind=kind, text=text, truncated=truncated)


def _extract_pdf(name: str, raw: bytes) -> str:
    try:
        import pypdf
    except ImportError as exc:  # pragma: no cover - pypdf is in requirements
        raise DocumentExtractionError(
            "PDF support needs pypdf on the server (pip install pypdf). Upload a "
            "DOCX or paste the text instead."
        ) from exc

    try:
        reader = pypdf.PdfReader(io.BytesIO(raw))
        if getattr(reader, "is_encrypted", False):
            # An owner-password PDF usually still decrypts with an empty
            # user password; try that before refusing the file.
            try:
                reader.decrypt("")
            except Exception as exc:
                raise DocumentExtractionError(
                    f"{name} is password-protected. Remove the password and upload "
                    "again, or paste the text."
                ) from exc
        pages = [(page.extract_text() or "") for page in reader.pages]
    except DocumentExtractionError:
        raise
    except Exception as exc:
        raise DocumentExtractionError(
            f"Could not read {name} as a PDF -- the file looks corrupt or is not "
            f"really a PDF ({type(exc).__name__}). Try re-exporting it."
        ) from exc

    return "\n\n".join(page for page in pages if page.strip())


def _extract_docx(name: str, raw: bytes) -> str:
    try:
        archive = zipfile.ZipFile(io.BytesIO(raw))
    except zipfile.BadZipFile as exc:
        raise DocumentExtractionError(
            f"Could not read {name} -- a .docx is a zip archive and this one is "
            "corrupt, or the file was renamed from another format. Re-save it from "
            "Word and try again."
        ) from exc

    try:
        with archive:
            document_xml = archive.read("word/document.xml")
    except KeyError as exc:
        raise DocumentExtractionError(
            f"{name} is a zip file but not a Word document (no word/document.xml "
            "inside). Upload the original .docx, or paste the text."
        ) from exc
    except Exception as exc:
        raise DocumentExtractionError(
            f"Could not read {name} ({type(exc).__name__}). Re-save it from Word "
            "and try again."
        ) from exc

    try:
        root = ElementTree.fromstring(document_xml)
    except ElementTree.ParseError as exc:
        raise DocumentExtractionError(
            f"{name} has malformed document XML inside it. Re-save it from Word "
            "and try again."
        ) from exc

    body = root.find(f"{_W}body")
    if body is None:
        return ""
    return "\n".join(_docx_block_lines(body))


def _docx_block_lines(element) -> list[str]:
    """Body text in document order, paragraphs and table rows on their own
    lines. Structure is worth preserving: project notes are usually headed
    sections and bullet lists, and a wall of run-together text reads far
    worse to a model than the same words laid out.

    Anything that is neither a paragraph nor a table (section wrappers,
    content controls, revision containers) is recursed into rather than
    special-cased, so an unfamiliar element never hides text.
    """
    lines: list[str] = []
    for child in element:
        tag = child.tag
        if tag == f"{_W}p":
            lines.append(_docx_paragraph_text(child))
        elif tag == f"{_W}tbl":
            for row in child.iter(f"{_W}tr"):
                cells = [
                    " ".join(
                        text
                        for text in (_docx_paragraph_text(p) for p in cell.iter(f"{_W}p"))
                        if text
                    ).strip()
                    for cell in row.findall(f"{_W}tc")
                ]
                if any(cells):
                    lines.append(" | ".join(cells))
        elif tag in (f"{_W}sectPr", f"{_W}bookmarkStart", f"{_W}bookmarkEnd"):
            continue
        else:
            nested = _docx_block_lines(child)
            if nested:
                lines.extend(nested)
            else:
                # A container holding runs directly rather than paragraphs.
                # Recursing alone would walk straight past it and drop the
                # text, which is the one thing this walker promises not to
                # do -- so anything with no paragraph or table inside it,
                # but with text, contributes that text as a line.
                text = _docx_paragraph_text(child)
                if text:
                    lines.append(text)
    return lines


def _docx_paragraph_text(paragraph) -> str:
    parts: list[str] = []
    for node in paragraph.iter():
        tag = node.tag
        if tag == f"{_W}t":
            parts.append(node.text or "")
        elif tag == f"{_W}tab":
            parts.append("\t")
        elif tag in (f"{_W}br", f"{_W}cr"):
            parts.append("\n")
    return "".join(parts).strip()


def _extract_plain_text(name: str, raw: bytes) -> str:
    if b"\x00" in raw[:4096]:
        raise DocumentExtractionError(
            f"{name} looks like a binary file rather than text, despite the "
            "extension. Upload the real PDF/DOCX, or paste the text."
        )
    for encoding in ("utf-8-sig", "utf-8", "cp1252", "latin-1"):
        try:
            return raw.decode(encoding)
        except UnicodeDecodeError:
            continue
    raise DocumentExtractionError(
        f"Could not decode {name} as text in any common encoding. Save it as UTF-8 "
        "and try again."
    )


def _normalize_whitespace(text: str) -> str:
    """Trailing spaces off every line, runs of blank lines down to one.

    PDF extraction in particular produces a lot of both, and they are pure
    cost: every wasted character is a token re-sent on every question of
    the interview.
    """
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = "\n".join(line.rstrip() for line in text.split("\n"))
    text = re.sub(r"[ \t]{2,}", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()
