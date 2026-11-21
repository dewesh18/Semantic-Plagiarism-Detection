"""
Text extraction module for PlagiarismAI.
Supports PDF, DOCX, TXT bytes and uploaded files.
"""

import io
import re
from docx import Document
from pypdf import PdfReader


def extract_text_from_pdf(data, max_pages=None):
    """Extract text from PDF bytes."""
    reader = PdfReader(io.BytesIO(data))
    if reader.is_encrypted:
        try:
            reader.decrypt("")
        except Exception:
            raise ValueError("This PDF is password-protected.")
    pages = reader.pages if max_pages is None else reader.pages[:max_pages]
    parts = []
    for page in pages:
        try:
            parts.append(page.extract_text() or "")
        except Exception:
            continue
    text = "\n\n".join(parts).strip()
    if not text:
        raise ValueError("No text found in the PDF (it may be a scanned image).")
    return text


def extract_text_from_docx(data):
    """Extract paragraph and table text from DOCX bytes."""
    doc = Document(io.BytesIO(data))
    parts = [p.text for p in doc.paragraphs if p.text.strip()]
    for table in doc.tables:
        for row in table.rows:
            for cell in row.cells:
                if cell.text.strip():
                    parts.append(cell.text.strip())
    text = "\n\n".join(parts).strip()
    if not text:
        raise ValueError("No text found in the DOCX file.")
    return text


def extract_text_from_txt(data):
    """Decode TXT bytes (UTF-8 first, then Latin-1 as fallback)."""
    for encoding in ("utf-8-sig", "utf-8", "latin-1"):
        try:
            return data.decode(encoding)
        except UnicodeDecodeError:
            continue
    return data.decode("utf-8", errors="ignore")


def extract_text_from_upload(uploaded_file):
    """Pick the right extractor for a Streamlit uploaded file."""
    name = uploaded_file.name.lower()
    data = uploaded_file.getvalue()
    if name.endswith(".pdf"):
        return extract_text_from_pdf(data)
    if name.endswith(".docx"):
        return extract_text_from_docx(data)
    if name.endswith(".txt"):
        return extract_text_from_txt(data)
    raise ValueError(f"Unsupported file type: {uploaded_file.name} (use PDF, DOCX or TXT).")
