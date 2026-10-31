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
1. Use only the excerpts; no outside knowledge, even for well-known facts.
   Answer whenever the excerpts contain the needed facts, directly or through simple arithmetic, even if the
   wording differs from the question. Set "answerable": false ONLY when the excerpts do not cover the subject or
   lack a number you need. Do not refuse over minor details (an unstated year, a paraphrased claim).
2. Prefer the most exact figure: if a table gives 5.439 and the text says "about 5.4", use 5.439. Do not compute a
   value yourself when a document reports it directly. "Up to X%" in text is weaker than an exact table value.
3. answer_value must be expressed in the required unit:
   - a number: digits only, no commas, symbols (~, <, >, %, $) or words; convert the unit if needed
     (for example 5.439 million liters -> 5439000 when the unit is liters).
   - a range: [low,high]
   - True/False questions (required unit is_blank): 1 if the excerpts support the statement, 0 if they
     contradict it. Judge the statement exactly as written.
   - a name or term (required unit is_blank): the exact wording used in the document.
4. Arithmetic on numbers from the excerpts is allowed (ratios, differences, unit conversions, combining two
   documents). Show it in "explanation".
5. ref_ids: the document IDs of the excerpts you used, as written in the excerpt headers.
6. supporting_materials: a short verbatim quote (or "Table N" / "Figure N") from those excerpts.

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


def ask(q: dict, chunks: list, llm_fn, cache: dict, model: str):
    messages = build_messages(q["question"], q.get("answer_unit"), chunks)
    key = cache_key(messages, model)
    if key not in cache:
        cache[key] = llm_fn(messages)
    return parse_reply(cache[key])


def answer_question(q: dict, chunks: list, llm_fn, urls: dict, cache: dict, model: str,
                    retry_chunks: list = None) -> tuple:
    """One submission row, plus whether the wider retry was used.

    If the model finds no answer in the first excerpts, it gets one more try with a wider set (retry_chunks):
    in the training run, 7 of 9 refusals were answerable, most needing a fact that ranked just below the cut."""
    reply = ask(q, chunks, llm_fn, cache, model)
    retried = False
    if retry_chunks and len(retry_chunks) > len(chunks) and (not reply or not reply.get("answerable", True)):
        reply, retried = ask(q, retry_chunks, llm_fn, cache, model), True
    return to_row(q["id"], q["question"], q.get("answer_unit"), reply, urls, set(urls)), retried
