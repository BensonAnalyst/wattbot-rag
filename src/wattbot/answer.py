"""Prompt, LLM call, and turning the model's JSON into one submission row in the train_QA.csv format."""

import hashlib
import json
import re

from .score import BLANK, parse_number

NA_TEXT = "Unable to answer with confidence based on the provided documents."
COLUMNS = ["id", "question", "answer", "answer_value", "answer_unit", "ref_id", "ref_url",
           "supporting_materials", "explanation"]

PROMPT = """You answer questions about the environmental impact of AI using ONLY the numbered excerpts below.
Each excerpt shows its document ID and page.

QUESTION: {question}
REQUIRED UNIT FOR answer_value: {unit}

EXCERPTS:
{sources}

Rules:
1. Use only the excerpts. If they do not contain the answer, set "answerable": false. Do not use outside knowledge,
   even for well-known facts.
2. answer_value must be expressed in the required unit:
   - a number: digits only, no commas, symbols (~, <, >, %, $) or words; convert the unit if needed
     (for example 5.439 million liters -> 5439000 when the unit is liters).
   - a range: [low,high]
   - True/False questions (required unit is_blank): 1 for true, 0 for false.
   - a name or term (required unit is_blank): the exact wording used in the document.
3. Simple arithmetic on numbers from the excerpts is allowed (ratios, differences, unit conversions). Show it in
   "explanation".
4. ref_ids: the document IDs of the excerpts you used, as written in the excerpt headers.
5. supporting_materials: a short verbatim quote (or "Table N" / "Figure N") from those excerpts.

Reply with JSON only:
{{"answerable": true, "answer": "short natural-language answer with unit", "answer_value": "...",
"ref_ids": ["docid"], "supporting_materials": "...", "explanation": "..."}}"""


def format_sources(chunks: list) -> str:
    return "\n\n".join(f"[{c['doc_id']}, page {c['page']}]\n{c['text']}" for c in chunks)


def build_messages(question: str, unit: str, chunks: list) -> list:
    return [{"role": "user", "content": PROMPT.format(question=question.strip(), unit=unit or BLANK,
                                                      sources=format_sources(chunks))}]


def cache_key(messages: list, model: str) -> str:
    return hashlib.sha256((model + json.dumps(messages, sort_keys=True)).encode()).hexdigest()[:16]


def parse_reply(raw: str):
    text = re.sub(r"```(?:json)?", "", raw or "").strip()
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        return None
    try:
        data = json.loads(match.group(0))
    except json.JSONDecodeError:
        return None
    return data if isinstance(data, dict) else None


_SCALE = {"thousand": 1e3, "million": 1e6, "billion": 1e9, "trillion": 1e12}


def normalise_value(value, unit: str) -> str:
    """Clean the model's answer_value into the competition's conventions."""
    if value is None:
        return BLANK
    if isinstance(value, bool):
        return "1" if value else "0"
    if isinstance(value, (int, float)):
        return _fmt(float(value))
    if isinstance(value, list) and len(value) == 2:
        nums = [parse_number(normalise_value(v, unit)) for v in value]
        return f"[{_fmt(nums[0])},{_fmt(nums[1])}]" if None not in nums else BLANK
    text = str(value).strip()
    low = text.lower()
    if low in ("true", "yes"):
        return "1"
    if low in ("false", "no"):
        return "0"
    if text.startswith("[") and text.endswith("]"):
        return normalise_value([p.strip() for p in text[1:-1].split(",")], unit) if text.count(",") == 1 else text
    num = re.fullmatch(r"[~<>≈$\s]*(-?[\d,]*\.?\d+(?:e[+-]?\d+)?)\s*(thousand|million|billion|trillion)?\s*[%x×]?",
                       low)
    if num:
        n = float(num.group(1).replace(",", ""))
        scale = num.group(2)
        if scale and scale not in (unit or "").lower():
            n *= _SCALE[scale]
        return _fmt(n)
    return text


def _fmt(n: float) -> str:
    return str(int(n)) if float(n).is_integer() and abs(n) < 1e15 else repr(float(n))


def na_row(qid: str, question: str, explanation: str = "") -> dict:
    return {"id": qid, "question": question, "answer": NA_TEXT, "answer_value": BLANK, "answer_unit": BLANK,
            "ref_id": BLANK, "ref_url": BLANK, "supporting_materials": BLANK, "explanation": explanation or BLANK}


def to_row(qid: str, question: str, unit: str, reply, urls: dict, valid_ids: set) -> dict:
    """Build one submission row. Anything unusable becomes a full is_blank row (scored as unanswerable)."""
    if not reply or not reply.get("answerable", True):
        return na_row(qid, question, (reply or {}).get("explanation", "No supporting evidence retrieved."))
    value = normalise_value(reply.get("answer_value"), unit)
    refs = [r for r in dict.fromkeys(str(r).strip() for r in (reply.get("ref_ids") or [])) if r in valid_ids]
    if value == BLANK or not refs:
        return na_row(qid, question, reply.get("explanation", ""))
    return {
        "id": qid, "question": question,
        "answer": str(reply.get("answer") or value),
        "answer_value": value,
        "answer_unit": unit or BLANK,
        "ref_id": "[" + ",".join(f"'{r}'" for r in refs) + "]",
        "ref_url": "[" + ",".join(f"'{urls[r]}'" for r in refs) + "]",
        "supporting_materials": str(reply.get("supporting_materials") or BLANK),
        "explanation": str(reply.get("explanation") or BLANK),
    }


def answer_question(q: dict, chunks: list, llm_fn, urls: dict, cache: dict, model: str) -> dict:
    """One submission row. cache maps prompt hash -> raw reply, so re-runs cost no LLM calls."""
    messages = build_messages(q["question"], q.get("answer_unit"), chunks)
    key = cache_key(messages, model)
    if key not in cache:
        cache[key] = llm_fn(messages)
    return to_row(q["id"], q["question"], q.get("answer_unit"), parse_reply(cache[key]), urls, set(urls))
