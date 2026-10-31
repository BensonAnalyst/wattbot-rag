"""Run the pipeline over a question set: embeddings, parallel LLM calls with a reply cache, output rows."""

import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd

from .answer import COLUMNS, answer_question, na_row


def make_embedder(model_name: str = "BAAI/bge-small-en-v1.5", device: str = None):
    """Dense embeddings (sentence-transformers, Apache 2.0; bge-small, MIT). Queries get bge's search prefix."""
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(model_name, device=device)
    prefix = "Represent this sentence for searching relevant passages: "

    def embed(texts):
        is_query = len(texts) == 1
        inputs = [prefix + t for t in texts] if is_query else texts
        return model.encode(inputs, batch_size=64, normalize_embeddings=True, show_progress_bar=not is_query)

    return embed


def load_cache(path) -> dict:
    p = Path(path)
    return json.loads(p.read_text()) if p.exists() else {}


def save_cache(cache: dict, path):
    Path(path).write_text(json.dumps(cache))


def answer_all(questions: pd.DataFrame, retriever, llm_fn, urls: dict, cache_path, model: str,
               k: int = 8, threads: int = 4):
    """Answer every question. Returns (submission DataFrame, {id: retrieved chunks}).

    Retrieval runs first (fast, single thread); LLM calls then run in parallel. Replies are cached on disk,
    so a re-run after a crash, or after changing only the post-processing, costs no LLM calls."""
    cache = load_cache(cache_path)
    records = questions.to_dict("records")
    retrieved = {q["id"]: retriever.search(q["question"], k=k) for q in records}

    def one(q):
        try:
            return answer_question(q, retrieved[q["id"]], llm_fn, urls, cache, model)
        except Exception as exc:  # one failed call should not lose the run
            return na_row(q["id"], q["question"], f"ERROR: {str(exc)[:200]}")

    rows = []
    with ThreadPoolExecutor(max_workers=threads) as pool:
        for i, row in enumerate(pool.map(one, records), 1):
            rows.append(row)
            if i % 20 == 0:
                save_cache(cache, cache_path)
                print(f"{i}/{len(records)} answered")
    save_cache(cache, cache_path)
    return pd.DataFrame(rows, columns=COLUMNS), retrieved
