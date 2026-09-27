"""Extract raw text from uploaded study materials (PDF, DOCX, TXT)."""

import io


def extract_text(file_storage):
    """Read an uploaded file (Flask FileStorage) and return its text."""
    filename = (file_storage.filename or "").lower()
    data = file_storage.read()
    if not data:
        raise ValueError("The uploaded file is empty.")

    if filename.endswith(".pdf"):
        import pdfplumber

        parts = []
        with pdfplumber.open(io.BytesIO(data)) as pdf:
            for page in pdf.pages:
                parts.append(page.extract_text() or "")
        text = "\n".join(parts)
    elif filename.endswith(".docx"):
        import docx

        doc = docx.Document(io.BytesIO(data))
        text = "\n".join(p.text for p in doc.paragraphs)
    else:  # .txt and anything else: try plain text
        text = data.decode("utf-8", errors="ignore")

    text = " ".join(text.split())
    if not text.strip():
        raise ValueError("No readable text found in the uploaded file.")
    return text
