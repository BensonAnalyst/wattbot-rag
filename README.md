# WattBot 2025: RAG over AI-sustainability research

**A retrieval-augmented pipeline that answers precise questions about AI's energy, water and carbon footprint from 32 documents, cites its sources, and says "unable to answer" when the documents don't support an answer.** Built as a late submission to the [WattBot 2025](https://www.kaggle.com/competitions/WattBot2025) Kaggle competition (UW–Madison ML+X).

> **Status: iterating.** First pipeline: **0.754** offline WattBot score on the 41 training questions. Error analysis and fixes below.

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

## Results and error analysis

| Version | Change | Train score (41 q) | Leaderboard |
|---|---|---|---|
| v1 | Hybrid retrieval, top 8 excerpts, strict "answer only from the excerpts" prompt | **0.754** (30 fully correct) | not submitted |
| v2 | Fewer refusals, exact table values preferred, wider retry when refused | **0.768** (refusals 9 → 6) | not submitted |
| v3 | Question type stated to the model, 3-significant-figure rounding for calculated values, robust citation parsing | 0.763 as run; **0.826** after one more fix (below) | pending |

**v1 error analysis (11 questions lost points):**

| Cause | Questions | What happened |
|---|---|---|
| **Refused an answerable question** | 7 | 9 refusals, only 2 correct. The prompt was too strict: one refusal was "the excerpt doesn't say the year was 2020"; others needed a second fact (e.g. a household's yearly kWh from another paper) that ranked just below the top 8 |
| **Rounded text instead of exact table value** | 4 | "5.4 million liters" in text vs 5.439 in Table 1; "seven times" vs 48.66 ÷ 5.58 = 8.72 from a table; computed 33B × 2 bytes = 66 GB where Table 3 says 64.7 GB |
| Figure-only answer | 1 | A count read off a bar chart; text extraction can't see it (accepted for now) |

**v2 fixes:** the prompt now answers whenever the facts are present (directly or by arithmetic) and refuses only when the subject or a needed number is missing; it prefers exact table values over rounded text and never recomputes a value a document states. A refused question is retried once with 20 excerpts instead of 8. The trade-off is deliberate: only ~5% of training questions are truly unanswerable, so a wrong refusal costs more on average than a wrong answer.

**v2 error analysis:** the v1 fixes worked (5 refused questions now answered, 3 of them fully right), but exposed three bugs of my own:

| Bug | Questions | Fix in v3 |
|---|---|---|
| Name answers came back as `1` ("ML.ENERGY Benchmark" → 1): the True/False rule ("unit is_blank → 1/0") also matched name questions, which have no unit either | 2 | The prompt states the question type (number, true/false, name); post-processing rejects a 1/0 for a name question |
| Calculated values too precise: 13.02 vs 13, 8.73 vs 8.7204 (tolerance is ±0.1%) | 2 | Calculated values rounded to 3 significant figures; stated values copied exactly. Fitted on 6 training calculations (5 of 6 match), so it is a heuristic, not a rule from the organisers |
| A correct value (5,439,000 L) discarded because the citation wasn't an exact document ID | 1 | Citations matched leniently ("li2025b, page 5" → li2025b); if none is usable, the top retrieved document is cited rather than losing the answer |

**v3 result:** the targeted fixes landed (q003 name, q054 now reads 64.7 GB from Table 3, q280 rounds to 13), but the model then left `answer_value` **empty** on three name questions ("GPT-3", "water consumption"), which the cleanup didn't treat as missing. Treating an empty value as missing and falling back to the answer text recovers all three: **0.826** on the same saved replies, no new LLM calls.

**Run-to-run variation:** even at temperature 0, three answers changed between runs with identical inputs except the prompt (q075, q124, q272). With 41 questions, one question is worth ~2.4 points, so differences under ~0.03 between versions are within noise.

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
