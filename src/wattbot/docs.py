"""Documents: read metadata, download the PDFs, extract text page by page, and cut it into chunks.

The competition data may not be redistributed, so nothing here is committed: the PDFs are downloaded at run time
from the URLs in metadata.csv.
"""

import json
import re
import time
from pathlib import Path

import pandas as pd


def load_metadata(path) -> pd.DataFrame:
    """metadata.csv is not UTF-8 (Windows-1252), so try the encodings in order."""
    for enc in ("utf-8", "cp1252", "latin-1"):
        try:
            meta = pd.read_csv(path, encoding=enc)
            break
        except UnicodeDecodeError:
            continue
    meta.columns = [c.strip() for c in meta.columns]
    meta["id"] = meta["id"].str.strip()
    meta["url"] = meta["url"].str.strip()
    return meta


def first_author(citation: str) -> str:
    """'Jesse Dodge, Taylor Prewitt, ...' -> 'dodge'; 'Amazon Staff. (2023)' -> 'amazon'."""
    head = re.split(r"[,(.&]", str(citation), maxsplit=1)[0].strip()
    words = re.findall(r"[A-Za-z\-']+", head)
    return words[-1].lower() if words else ""


def download_pdfs(meta: pd.DataFrame, out_dir, retries: int = 3) -> dict:
    """Download each document once. Returns {doc_id: path or None if it failed}."""
    import requests

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    result = {}
    for row in meta.itertuples():
        path = out_dir / f"{row.id}.pdf"
        if path.exists() and path.stat().st_size > 10_000:
            result[row.id] = path
            continue
        result[row.id] = None
        for attempt in range(retries):
            try:
                resp = requests.get(row.url, timeout=120, headers={"User-Agent": "Mozilla/5.0 (WattBot research)"})
                if resp.ok and resp.content[:4] == b"%PDF":
                    path.write_bytes(resp.content)
                    result[row.id] = path
                    break
            except requests.RequestException:
                pass
            time.sleep(3 * (attempt + 1))
    return result


def extract_pages(pdf_path) -> list:
    """Text of each page (pdfplumber, MIT licence). Tables come out as text rows in reading order."""
    import pdfplumber

    pages = []
    with pdfplumber.open(str(pdf_path)) as pdf:
        for i, page in enumerate(pdf.pages, 1):
            text = page.extract_text(x_tolerance=1.5, y_tolerance=3) or ""
            pages.append({"page": i, "text": _clean(text)})
    return pages


def _clean(text: str) -> str:
    text = re.sub(r"-\n(?=[a-z])", "", text)  # re-join words hyphenated across lines
    text = re.sub(r"[ \t]+", " ", text)
    return text.strip()


def chunk_pages(doc_id: str, title: str, pages: list, size: int = 220, overlap: int = 60) -> list:
    """Word windows within a page, so every chunk keeps its page number for the citation.

    Each chunk's search text starts with the document ID and title, so a question naming the paper
    (\"JetMoE\", \"Power Hungry Processing\") matches all of that paper's chunks."""
    chunks = []
    step = size - overlap
    for p in pages:
        words = p["text"].split()
        if not words:
            continue
        for start in range(0, max(len(words) - overlap, 1), step):
            body = " ".join(words[start:start + size])
            chunks.append({
                "chunk_id": f"{doc_id}:p{p['page']}:{start // step}",
                "doc_id": doc_id,
                "page": p["page"],
                "text": body,
                "search_text": f"{doc_id} {title}. {body}",
            })
    return chunks


def build_chunks(meta: pd.DataFrame, pdf_paths: dict, cache_path=None) -> list:
    """Parse every downloaded PDF into chunks; cached as JSON because parsing takes a few minutes."""
    if cache_path and Path(cache_path).exists():
        return json.loads(Path(cache_path).read_text())
    chunks = []
    for row in meta.itertuples():
        path = pdf_paths.get(row.id)
        if path:
            chunks += chunk_pages(row.id, row.title, extract_pages(path))
    if cache_path:
        Path(cache_path).write_text(json.dumps(chunks))
    return chunks
