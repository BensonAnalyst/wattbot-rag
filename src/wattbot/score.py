"""Offline copy of the WattBot score, so the pipeline can be scored on train_QA.csv before submitting.

Score per question = 0.75 * answer_value + 0.15 * ref_id + 0.10 * is_NA.
- answer_value: numbers within 0.1% relative tolerance; ranges compared end by end; text exact after normalisation.
- ref_id: Jaccard overlap of the document ID sets (case-insensitive).
- is_NA: for an unanswerable question, every evidence field must be is_blank. For an answerable question the
  competition does not say how this component is scored; here it counts as correct when the prediction is not
  marked unanswerable (an assumption, stated in the README).
"""

import ast
import re

BLANK = "is_blank"
NA_FIELDS = ("answer_value", "answer_unit", "ref_id", "ref_url", "supporting_materials")
REL_TOL = 1e-3


def is_blank(value) -> bool:
    return value is None or (isinstance(value, float) and value != value) or str(value).strip().lower() in ("", BLANK)


def parse_number(value):
    """Return a float, or None when the value is not a plain number."""
    text = str(value).strip().replace(",", "")
    try:
        return float(text)
    except ValueError:
        return None


def parse_range(value):
    text = str(value).strip()
    if not (text.startswith("[") and text.endswith("]")):
        return None
    parts = [parse_number(p) for p in text[1:-1].split(",")]
    return parts if len(parts) == 2 and None not in parts else None


def _norm_text(value) -> str:
    return re.sub(r"\s+", " ", str(value).strip().lower())


def _close(a: float, b: float) -> bool:
    return a == b if b == 0 else abs(a - b) / abs(b) <= REL_TOL


def value_correct(pred, truth) -> bool:
    if is_blank(truth):
        return is_blank(pred)
    if is_blank(pred):
        return False
    t_range, p_range = parse_range(truth), parse_range(pred)
    if t_range is not None:
        return p_range is not None and all(_close(p, t) for p, t in zip(p_range, t_range))
    t_num, p_num = parse_number(truth), parse_number(pred)
    if t_num is not None:
        return p_num is not None and _close(p_num, t_num)
    return _norm_text(pred) == _norm_text(truth)


def parse_refs(value) -> set:
    if is_blank(value):
        return {BLANK}
    text = str(value).strip()
    try:
        items = ast.literal_eval(text) if text.startswith("[") else [text]
    except (ValueError, SyntaxError):
        items = re.findall(r"[A-Za-z]+\d{4}[a-z]?", text)
    if isinstance(items, str):
        items = [items]
    return {str(i).strip().lower() for i in items if str(i).strip()} or {BLANK}


def ref_score(pred, truth) -> float:
    p, t = parse_refs(pred), parse_refs(truth)
    return len(p & t) / len(p | t)


def na_score(pred_row: dict, truth_row: dict) -> float:
    if is_blank(truth_row["answer_value"]):
        return float(all(is_blank(pred_row.get(f)) for f in NA_FIELDS))
    return float(not is_blank(pred_row.get("answer_value")))


def score_row(pred_row: dict, truth_row: dict) -> dict:
    v = float(value_correct(pred_row.get("answer_value"), truth_row["answer_value"]))
    r = ref_score(pred_row.get("ref_id"), truth_row["ref_id"])
    n = na_score(pred_row, truth_row)
    return {"value": v, "ref": round(r, 3), "na": n, "score": round(0.75 * v + 0.15 * r + 0.10 * n, 4)}


def score_frame(pred_df, truth_df):
    """Join on id and return a per-question DataFrame plus the mean WattBot score."""
    preds = {r["id"]: r for r in pred_df.to_dict("records")}
    rows = []
    for t in truth_df.to_dict("records"):
        p = preds.get(t["id"], {})
        rows.append({"id": t["id"], "question": t["question"], "truth": t["answer_value"],
                     "pred": p.get("answer_value"), "truth_ref": t["ref_id"], "pred_ref": p.get("ref_id"),
                     **score_row(p, t)})
    import pandas as pd
    out = pd.DataFrame(rows)
    return out, round(out["score"].mean(), 4)
