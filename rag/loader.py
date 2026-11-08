"""Knowledge base document loader (Section 9.1).

Reads policy markdown files from kb/*.md and parses them into
structured records: [{doc_id, title, text}].
"""

import config  # Ground Rule 3: must be first import
from pathlib import Path


def parse_document(doc_id: str, raw_content: str) -> dict:
    """Parse raw markdown content into doc_id, title, and body text."""
    lines = [line.strip() for line in raw_content.strip().splitlines() if line.strip()]
    if not lines:
        return {"doc_id": doc_id, "title": doc_id, "text": ""}

    if lines[0].startswith("#"):
        title = lines[0].lstrip("#").strip()
        text = " ".join(lines[1:]).strip()
    else:
        title = lines[0]
        text = " ".join(lines[1:]).strip()

    return {
        "doc_id": doc_id,
        "title": title,
        "text": text,
    }


def load_documents(kb_dir: Path | None = None) -> list[dict]:
    """Load all markdown documents from the knowledge base directory."""
    target_dir = kb_dir or config.KB_DIR
    docs = []
    for fpath in sorted(target_dir.glob("*.md")):
        raw = fpath.read_text(encoding="utf-8")
        doc = parse_document(fpath.stem, raw)
        docs.append(doc)
    return docs

