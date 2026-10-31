# WattBot 2025: RAG over AI-sustainability research

**A retrieval-augmented pipeline that answers precise questions about AI's energy, water and carbon footprint from 32 documents, cites its sources, and says "unable to answer" when the documents don't support an answer.** Built as a late submission to the [WattBot 2025](https://www.kaggle.com/competitions/WattBot2025) Kaggle competition (UW–Madison ML+X).

> **Status: Day 1, first pipeline.** Scores will be added after the first runs.

## The task

282 test questions, mostly asking for an exact number in a given unit:

| Type | Test | Example |
|---|---|---|
| Numeric (within ±0.1%) | 217 | *"How many days did it take to train GPT-3 using 10,000 V100 GPUs?"* |
| True / False | 37 | *"True or False: Estimating GPU energy from TDP is a reliable method."* |
| Name or term | 28 | *"Which software package was used to measure energy during inference?"* |
| Unanswerable (hidden among the above) | ? | *"How many miles is the Earth from the Sun?"* (not in the documents) |

**Score per question** = 0.75 × exact answer value + 0.15 × cited documents (Jaccard) + 0.10 × correctly flagging unanswerable questions.

## Pipeline

```
metadata.csv ──► download 32 PDFs ──► page text (pdfplumber) ──► 220-word chunks, tagged doc + page
                                                                        │
question ──► author named? ("Dodge et al.") restrict to that paper ──► BM25 + bge-small embeddings
                                                                        │  reciprocal rank fusion, top 8
                                                                        ▼
                       gpt-oss-120b: answer ONLY from the excerpts, in the required unit, or "answerable": false
                                                                        │
                    normalise value (ranges, 1/0, units, million → digits) ──► submission row + citations
```

## Evaluation approach

- **Offline copy of the competition metric** (`src/wattbot/score.py`), checked against the training answers (they score 0.995 against themselves; answering nothing scores 0.049).
- **Retrieval measured separately from answering**: document recall@8 on the training questions, BM25 vs hybrid, before any LLM call.
- **LLM replies cached by prompt hash**, so changing post-processing or re-running costs nothing.
- Only **41 labelled questions**, so offline differences of a few points are noise; the leaderboard (5 submissions a day) confirms larger changes.

## Rules followed

- Answers are produced by code end to end; no manual lookups.
- **No competition data in this repo** (it may not be redistributed): the notebook downloads it at run time.
- Open-source components only, with licences that allow commercial use: pdfplumber (MIT), sentence-transformers (Apache 2.0), bge-small-en-v1.5 (MIT), gpt-oss-120b (Apache 2.0).

## Run it

**Kaggle:** open `notebooks/wattbot_pipeline.ipynb` as a Kaggle notebook, add the competition data, turn Internet on, and add two secrets: `DATABRICKS_HOST` and `DATABRICKS_TOKEN` (a Databricks serving endpoint for `databricks-gpt-oss-120b`).

**Tests (offline, no data needed):**
```bash
pip install -r requirements-dev.txt
pytest -q
```

## Notes

- The training file keeps a unit (e.g. `lbs`) on its unanswerable rows, while the rules say every evidence field, including the unit, must be `is_blank`. This pipeline follows the written rule.
- How the `is_NA` component scores *answerable* questions isn't documented; the offline metric assumes it counts as correct when the prediction is not marked unanswerable.

## Repo structure

```
src/wattbot/
  docs.py       metadata, PDF download, page text, chunking
  retrieve.py   BM25, dense embeddings, rank fusion, author-name filter
  answer.py     prompt, JSON parsing, value normalisation, submission rows
  llm.py        Databricks serving endpoint client
  run.py        embeddings, parallel answering with a reply cache
  score.py      offline WattBot metric
notebooks/wattbot_pipeline.ipynb   end-to-end Kaggle notebook
tests/                             offline tests with invented examples
```
