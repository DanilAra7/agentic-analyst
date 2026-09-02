# agentic-analyst

A RAG + SQL analyst over e-commerce data, built to answer one question honestly:
**what does retrieval quality actually cost?**

Every architectural decision here is measured, not asserted. Decisions that did
not work are kept in the repo with the numbers that killed them.

---

## Results

Two question sets over the same corpus:

- **easy** — questions phrased in the document's own vocabulary
- **hard** — the same facts asked the way a user would ask them

The gap between them is the price of lexical mismatch. **Every decision is
judged on the hard set.**

### Retrieval, hard set (51 questions)

| Configuration | recall@1 | recall@5 | MRR | rerank p50 |
|---|---|---|---|---|
| Dense, bge-m3, 400/60 chunks | 0.510 | 0.765 | 0.623 | — |
| + cross-encoder reranker, window 20 | 0.471 | 0.882 | 0.657 | 2.8 s |
| **+ document title prepended to chunk** | **0.765** | 0.863 | **0.820** | — |
| + title + reranker, window 20 | 0.667 | 0.941 | 0.787 | 2.5 s |
| + title + reranker, window 50 | 0.667 | **0.961** | 0.791 | 4.6 s |
| + title + section + table header | 0.647 | 0.843 | 0.755 | — |
| hierarchical retrieval (doc → chunk) | 0.765 | 0.863 | — | — |

Easy set for reference: recall@5 goes 0.879 → 0.989.

Reranker: `bge-reranker-v2-m3`, float16, Apple M4 GPU. Latency is the median of
**single** queries, not batched: in production a request arrives alone.

### Three findings worth the repo

**1. There is no single best configuration.**
The reranker raises recall@5 from 0.863 to 0.961 and *lowers* recall@1 from
0.765 to 0.667. It is better at assembling the right five and worse at picking
the single best one. Which row wins depends on who consumes the output: top-5
into an LLM context, or one answer shown to a user. Turning the reranker on is
a per-product decision, not a universal improvement.
Root cause: the cross-encoder rewards topical coherence of the pair. A regional
handbook is coherent prose about one state; a row of a shipping-rate table is
not. See decisions 11 and 13.

**2. An aggregate metric hid a regression.**
Mean recall@5 improved and looked like a clean win. Looking at the distribution
of rank changes instead showed 10 questions entering the top-3 and 4 leaving it.
The trade was favourable, but it was a trade, and only visible when measured as
one.

**3. Checking a hypothesis before building it saved two days.**
Contextual retrieval was dropped after a check showed its target failure class
was an artifact of bad labels (decision 10). Hierarchical retrieval was dropped
after a 20-minute measurement showed it matches flat search to three decimals
(decision 14). BM25 is postponed because 0 of 51 hard questions contain an exact
code, so the current question set cannot measure it (decision 12).

### Generation and agent

Not built yet. Tables are empty on purpose rather than filled with guesses.

| Configuration | faithfulness | answer relevancy | steps/task | p95 | $/query |
|---|---|---|---|---|---|
| RAG, no agent | — | — | 1 | — | — |
| Agent, naive flow | — | — | — | — | — |
| Agent, optimised flow | — | — | — | — | — |

---

## Status

| | | |
|---|---|---|
| M0 | infrastructure: LLM providers with an on-disk cache, Olist → DuckDB | **done** |
| M1 | corpus, chunking, embeddings, dense search, two golden sets, baseline | **done** |
| M2 | retrieval ablation: reranker, chunk enrichment, hierarchy, BM25 check | **done** |
| M3 | agent: tool calling, SQL tool, document search tool | not started |
| M4 | Langfuse: traces, per-step latency and token breakdown | not started |
| M5 | agent flow optimisation, before/after measurement | not started |

`src/agent/` and `src/tools/` are empty packages. Nothing in this README claims
otherwise.

---

## What is measured, and how

**No labelled data exists for this corpus, so the golden set is synthesised**:
a model reads a random chunk and writes a question answerable from it, giving
(question → gold chunk) pairs. This is cheap and it is also the weakest link in
the whole project, so it gets its own scrutiny:

- **The judge was broken and it took a control run to notice.** Auditing 18
  retrieval misses returned "0 false misses" — a convenient answer. A control on
  18 known-good chunks showed the judge said NO half the time. Cause: the word
  `completely` in the prompt. Fixed with a FULL/PARTIAL/NO scale and reasoning
  before verdict; the control then returned 0 rejections. Decision 8.
- **The question set contained unanswerable questions.** The generator was told
  to name a state for disambiguation and invented one (almost always São Paulo)
  where the source chunk had none. 13 easy and 8 hard questions removed.
  Decision 9.
- **The judge is never the same model as the worker.** A model systematically
  over-scores its own output. This is asserted at startup and raises if the two
  providers match (`src/config.py`).

Metrics: recall@{1,3,5,10,20} and MRR, in two strictness levels — **strict**
(the exact gold chunk) and **lenient** (any chunk of the right document). The
gap between them separates a chunking problem from a retrieval problem, and it
is what showed that the structural enrichment variant finds the right document
but picks the wrong chunk inside it.

**Retrieval evaluation makes zero LLM calls**, so the whole ablation is free and
can be re-run as often as needed.

### Known gaps in the measurement

Written down rather than hidden — see `BACKLOG.md` for all 27 items.

- The judge has not been validated against human labels yet (item 14).
- The hard set is small (51 questions) and contains no keyword-style queries,
  which is exactly why BM25 cannot be judged on it (items 24, 25).
- Reranker latency is measured on a laptop GPU, not production hardware (item 22).

---

## Data

- **Structured**: the public [Olist Brazilian e-commerce dataset](https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce)
  (~100k orders), loaded into DuckDB as two views at **different grains** —
  `order_facts` (one row per item) and `order_summary` (one row per order).
  This matters: `AVG(review_score)` over the item grain returns 4.0170, over the
  order grain 4.0868. The first number is wrong, because orders with more items
  are counted more times *and* score lower.
- **Unstructured**: 373 synthetic policy documents (~52k words) generated
  deterministically by `src/ingest/build_docs.py` on top of the real dataset's
  facts — shipping-rate tables, regional handbooks, superseded and in-force
  policy versions, and 250 near-identical bulletins acting as distractors. The
  retrieval difficulties in this corpus are planted on purpose.

---

## Stack

Nothing to deploy: everything runs locally from a virtualenv.

| Layer | Choice | Why |
|---|---|---|
| SQL | DuckDB | reads CSV directly, real window functions, no server |
| Embeddings | bge-m3, local | multilingual, no rate limit, zero cost |
| Reranker | bge-reranker-v2-m3, local | cross-encoder over the candidate window |
| Vector search | numpy exact | 422 chunks; ANN is a later measured trade-off |
| LLM | Mistral (worker + separate judge) | OpenAI-compatible, free tier |
| Tracing | Langfuse | planned, M4 |

The **LLM call cache** (`src/llm/cache.py`) is load-bearing, not an
optimisation: an ablation run is thousands of requests, and repeat runs have to
be free and deterministic.

---

## Running it

```bash
uv sync
cp .env.example .env      # add a Mistral key
make corpus               # generate the 373-document corpus
make chunks               # split into chunks
make index                # embed (about 30 s on CPU)
make eval                 # retrieval metrics on both golden sets
make ablation             # reranker quality/latency curve (about 12 min on GPU)
```

`make data` additionally builds the DuckDB database and needs the Kaggle
dataset; it is only required for M3.

`make golden` regenerates the golden sets and is the only step that spends LLM
tokens. The generated sets are committed, so everything above runs without an
API key.

---

## Repository map

| Path | What |
|---|---|
| `DECISIONS.md` | **14 architectural decisions**, each with alternatives, cost and the number that verifies it |
| `BACKLOG.md` | 27 open items, each tied to observed evidence |
| `src/rag/` | chunking, embedding index, dense search, reranker |
| `src/eval/` | golden set synthesis, retrieval metrics, judge audit, ablations |
| `src/ingest/` | Olist → DuckDB, synthetic corpus generator |
| `src/llm/` | provider abstraction, retries, on-disk cache |
| `evals/` | golden sets, chunk index, all measurement outputs |

`DECISIONS.md` and `BACKLOG.md` are written in Russian; the code and this README
are in English.

---

## License

MIT
