# Improvement backlog

Every item is tied to an **observed failure**, not to a list of fashionable techniques.
The "evidence" line says how we know this is a problem at all.
Until an improvement is measured, it remains a hypothesis.

Baseline on the smoke test (12 hand-written questions, dense retrieval, 400/60 chunks):
`recall@1 = 0.58`, `recall@5 = 0.58`, `recall@20 = 0.83`.

The key observation: **recall@1 = recall@5**. Dense retrieval either finds the right
piece first or does not find it in the top five at all. Meanwhile recall@20 is markedly
higher. So the main trouble is **ranking, not the absence of the document from the index**.

---

## M2 — retrieval

### 1. Cross-encoder reranker
**Evidence.** The gap between recall@5 = 0.58 and recall@20 = 0.83. A quarter of the
questions find the right chunk but place it at ranks 6-20.
**What we do.** Take the top-50 by dense retrieval, run them through
`bge-reranker-v2-m3`, keep the top-5.
**What it costs.** Latency: a cross-encoder scores the pair "query + document" and cannot
be precomputed. Expect +0.5-1 s per request.
**Expectation.** The largest gain of all the items. Checked on the questions
"How long do I have to return something?" (#6) and "money back" (#10).

### 2. Hybrid search, BM25 + RRF
**Evidence.** "Which return code is used when the **wrong item** was shipped?" — #8.
"Who pays if item arrived **broken**?" — not found at all. The words "wrong item" and
"broken" do not appear in the documents; the codes RC-103 and RC-101 do.
**What we do.** DuckDB FTS for BM25, fused with dense retrieval through Reciprocal Rank
Fusion.
**What it costs.** A second index and synchronisation when the corpus changes.
**Expectation.** Fixes exact terms and codes, has almost no effect on paraphrases.

### 3. Context-dependent metadata filter
**Evidence.** "What is the current restocking fee?" worked (#1), while "How long do I have
to return something?" — #6 — did not: POL-RET-001 (superseded) and POL-RET-002 (in force)
compete.
**What we do.** `status = IN_FORCE` by default. If the query carries a date or a reference
to the past, the archive is opened.
**Why not a hard filter.** It would break legitimate questions of the form "which policy
applied to a 2017 order". The answer to those lies precisely in the superseded version.
**What it costs.** A "question about the past" detector is needed — one more point of failure.
**Expectation.** Cheap, narrow, reliable.

### 4. Contextual retrieval
**Evidence.** Chunks from POL-RATE-001 and from tables lose their subject: a rate row does
not know which state it belongs to.
**What we do.** Prepend a short document context to the chunk before embedding. Generated
by an LLM with prompt caching.
**What it costs.** One LLM pass over all chunks, plus reindexing.
**Expectation.** Works in combination with hybrid search and the reranker; on its own it
gives less.

### 5. Approximate search (HNSW) instead of exact
**Evidence.** Not needed yet: 422 chunks, an exact scan takes milliseconds.
**Why we do it anyway.** As a separate ablation row — to show **how much recall the speed
costs**. Exact search stays the reference for comparison.
**What it costs.** Nothing at this scale; the point of the item is precisely the measurement.

### 6. Query rewriting
**Evidence.** Will appear in conversational mode: there is nothing to search for in the
turn "and how much does that cost?".
**Status.** Deferred to M3; pointless outside a dialogue.

---

## M3 — agent

### 7. SQL tool
Text-to-SQL over DuckDB, two marts of different grain (`order_facts` — items,
`order_summary` — orders).
**The pitfall.** The model will compute aggregates at the wrong grain. The tool description
has to name the grain of each mart explicitly.

### 8. Document search tool
A wrapper over the best configuration from M2.

### 9. Agent loop
Iteration limit, repeat detection, tool errors returned as text, protection against prompt
injection through tool results.

### 10. Difficult question types in the golden set
Questions requiring two sources (SQL + document), questions with no answer in the corpus,
typos, another language. Deliberately deferred: a mixed set is impossible to interpret
until the simple case has been measured.

---

## M4–M5 — observability and optimisation

### 11. Langfuse
Traces, the input and output of every step, tool calls, a breakdown of latency and tokens.

### 12. An honest latency measurement
**Problem.** The cache returns an answer instantly, so a p95 from a cached run is a fiction.
A system that answers in 3 seconds will show 100 ms.
**Solution.** Split the runs: the correctness eval runs with the cache (cheap, deterministic,
frequent), the latency benchmark runs with the cache off on a small sample (expensive, rare).
These are two different measurements with different purposes.

### 13. Agent flow optimisation
Fewer steps, parallel calls to independent tools, streaming the answer. Measured before
and after.

---

## Methodological debt

### 14. PARTIAL. Judge validation
100 examples are labelled by hand and the agreement with the LLM judge is computed.
**This matters especially because** the worker and the judge come from the same provider,
and the residual self-preference bias has to be measured rather than assumed away.

### 15. Synthetic questions are unfairly easy
Questions generated from a chunk inherit its wording. A real user will ask in different
words. Paraphrased questions are needed — not for the sake of difficulty but for realism.

---

## Methodological debt (continued)

### 16. CLOSED. One gold label per question understates recall
**Evidence.** "When will my stationery order arrive in São Paulo?" is marked as a miss, yet
at least four documents answer it honestly: the SP rate table, the deadline policy, the SP
regional handbook and the category handbook. There is one label and several right answers.
**The scale of the problem is unknown** — some of the 18 misses on the hard set may be false
in the same way.
**Options.** (a) multiple labels at generation time; (b) relevance scored by a judge instead
of a strict chunk_id match; (c) manual re-labelling of the misses.
**Audit result (see decision 8).** There are no fully answering alternatives (0/18). Partially
answering: 6/18. The honest baseline range is **0.695-0.797**. A full re-labelling is not
needed, but both bounds have to be carried through the whole ablation.

### 17. The hard set is small (59 questions)
Conversational questions more often lose the distinguishing entity and get rejected. Either
the requirement has to be relaxed, or we generate with a surplus and top up to 120+.

### 18. Refined diagnosis: retrieval lands in the neighbourhood, not on the figure
**Evidence.** In 6 of the 18 misses on the hard set the output contains `OPS-XX-001` for the
right state — the right region, but baseline deadlines instead of the exact figure from the
rate table `POL-RATE-001`.
**What this changes.** Previously the diagnosis read "on hard questions the document is not
found". More precisely: an **adjacent** document is found. So the reranker's job is not so
much to lift the right piece out of the depths as to **tell similar documents apart** — which
is exactly what a cross-encoder does better than a bi-encoder.

### 19. Numeric ranges in tables
**Evidence.** The question "how much does it cost to send 2 kg" against a rate table that
carries the range `1.0–3.0 kg`. The string "2" does not appear in the chunk at all.
**Why this is hard.** Neither BM25 nor vector search understands that 2 falls inside the
interval 1.0–3.0. Contextual retrieval does NOT fix this either: adding context does not turn
"2" into "1.0–3.0".
**Options.** (a) enrich the chunk at indexing time — expand the range into an explicit list of
values; (b) move the table into structured form and answer such questions by computation
rather than retrieval (that is, with a tool, not with RAG).
**Observation.** Option (b) hints that some questions are not about retrieval at all. A rate
table is data, not text, and it belongs in the SQL tool.

### 20. Failure classes after the label cleanup
The current breakdown of hard-set failures:
- **ranking** — the document is found but below rank 5. The reranker aims here
- **numeric ranges** — see item 19
- the "lost state" class is CLOSED: it turned out to be a golden-set defect, not a retrieval problem

### 21. The reranker amplifies the "figure -> neighbourhood" substitution
**Evidence.** 4 hard-set questions where the gold chunk was in the top-3 before reranking and
dropped out after. In all of them the top slot went to `OPS-XX-001` for the right state.
**Why.** A cross-encoder rewards the topical coherence of the pair; a state handbook is
coherent, a row of a rate grid is not.
**Options.** (a) move the rates into the SQL tool (see 19) — removes the class entirely;
(b) enrich table chunks with coherent text at indexing time so they have something to match on;
(c) a separate threshold or boost by document type.
**Observation.** `POL-RATE-001#18` and `#19` — adjacent chunks of the same table — are
indistinguishable to the reranker. The same defect, only inside a document.

### 22. Reranker latency has not been measured on production-like hardware
p50 from 1.6 to 5.1 s — that is an Apple M4 GPU, float16, a 568M-parameter model.
An honest figure needs one of: a smaller model (`bge-reranker-base`), ONNX/quantisation, or a
hosted reranker API. The README currently carries a note that this is a relative rather than
an absolute quantity.

### 23. Reranker latency depends on chunk length, not only on K
A trial measurement on short chunks gave 2.1 s at window 50; the real run gave 5.1 s. The
cause: attention is quadratic in the pair's length, and pieces of the rate table under 400
tokens end up in the candidate list. The README needs percentiles, not a median.

### 24. BM25 deferred: there is nothing to measure it with
The pre-build check (decision 12): on the hard set 0 questions out of 51 contain an exact code,
and the median word overlap with the gold chunk is 0.33. The hard-question generator paraphrased
everything into natural speech and stripped the codes out. A real user types codes.
**Item 25 first, then BM25.**

### 25. A third question type in the golden set: short and code-bearing
How people actually search a help centre: `SEDEX-12 deadline`, `PAC-STD AC`, a tracking number.
There are none of these in either set at the moment. Without them the hybrid is unmeasurable;
with them, debt 17 (the hard set is small) closes as well.

### 26. CLOSED. Hierarchical retrieval: measured, no prize
`lenient@20 = 0.980` is NOT the ceiling of a hierarchy: it is a chunk-level metric (a document
counts if any of its chunks broke through), whereas a hierarchy searches one vector per
document. For an 8,000-token rate table such a vector gets smeared out.
**Done, decision 14.** Document-level retrieval: @1 0.667 against 0.843 for chunks; the full
simulation of a hierarchy matched flat search to three decimals. Not building it.
Only small-to-big stays open — it concerns generation and is not measurable by retrieval metrics.

### 27. The reranker needs to be switchable by query type
Decision 13: it helps recall@5 and hurts recall@1. So this is not "switch it on for good" but a
per-query decision. A candidate rule: if the top-1 of dense retrieval is far ahead of the top-2,
do not rerank. The threshold is calibrated on the golden set.

### 28. The SQL tool sees only the marts; raw tables are hidden
A deliberate restriction (decision 16): the marts carry the decision about grain, and by joining
raw tables the model would reproduce the fan-out. The price: questions about payments, sellers
and geodata become unanswerable, because those tables are not in the marts. If such questions
are needed, they should be added through a new mart, not by opening access to the raw tables.

### 29. The SQL timeout has only been checked from above
`TIMEOUT_S = 20` and interruption through `con.interrupt()` are implemented, but on real data no
query ever reached the limit (the heaviest, a 113k x 113k join, finished in 5.2 s). So the
interrupt branch has never executed.
**A test that actually triggers it is needed**, otherwise this is unverified code.

### 30. The model does not refuse an unanswerable question
Decision 17: `d01` fails at all four levels of schema description. The model confidently returns
a number that looks plausible and means something other than what was asked.
**Options.** (a) a separate refusal instruction in the system prompt, plus a measurement of
whether it helps; (b) a post-hoc check comparing the meaning of the answer's columns with the
question; (c) more unanswerable questions in the set — right now there is one, and the conclusion
rests on a single example.
**Start with (c):** one example is not a measurement.

### 31. A set of 34 questions is too small for weak effects
The 95% interval at accuracy 0.941 and n=34 is 0.809-0.984. A difference of one question between
levels is indistinguishable from noise. To see effects of the order of 5 percentage points you
need 150+ questions, or a weaker model that has room to fall.

### 32. The measurement was made on one model
The conclusion "describing the grain does not help" was obtained on `gemini-3.5-flash-lite`. On a
weaker model grain errors might appear, and then the description would start working. A run on a
second provider would turn a statement about this model into a statement about the task.

---

## Open forks

Accepted decisions live in `DECISIONS.md` and deferred tasks live here, but a fork that is under
discussion right now used to be recorded nowhere. After a pause it had to be recalled from the
conversation.
**Rule:** a fork is written here the moment it is posed and moves to `DECISIONS.md` the moment it
is chosen.

### Closed
- What to do after M2 -> A (the agent), with an insertion from B. C and D deferred as 31 and 32.
- What to compare the agent against -> a router as the baseline (decision 18).
- How to measure answer correctness -> a hybrid of three metrics (decision 19).
- Prompt injection -> plant the attack in the corpus (decision 20).

### Open now
- The composition of the agent question set: how many of each type, and which two-source questions
  count as genuine.

### 33. Refusal does not work in either scheme
Decisions 17 and 21: `nq1` ("how many unique customers") fails both on the bare SQL tool at all
four levels of schema description and in the router. The answer 99,441 looks plausible and means
something other than what was asked. none = 0.50.
Check it on the agent, then treat it: an instruction in the system prompt, a post-hoc check, or
more unanswerable questions in the set.

### 34. The snippet length in the tool output has not been measured
`SNIPPET_CHARS` was raised from 700 to 1800 after truncation broke two answers (decision 22). The
value was chosen "to cover the longest chunk", not measured. It is a trade-off knob: a longer
snippet means a fuller answer but more tokens, more cost and more latency. An ablation over
700 / 1200 / 1800 / no truncation would give the curve.

### 35. Agent latency is measured through the cache and is therefore fictitious
The router run shows timings of about 1 ms — those are LLM cache hits. An honest latency figure
needs a separate run with the cache off on a small sample (debt 12, now become relevant).

### 36. Refusal is detected by a regular expression and should be a field
Decision 25: the pattern has already missed twice on correct refusals ("do not contain",
"is missing"). Structured output is needed: the model returns `{answer, refused, sources}` and
refusal is read from a field rather than guessed from text. It affects both schemes equally, so
the comparison stays fair.

### 37. One run is not a measurement
Decision 23: two agent runs gave 0.88 and 1.00 on the two-source bucket. A difference of one
question (`bq5` looping) equals 0.125 at n=8. Three or more runs with a mean and a spread are
needed, otherwise the differences under discussion drown in noise.

### 38. A router+agent hybrid as a separate configuration
Decision 23: on single-source questions the agent gives nothing (both schemes 1.00) while
spending several times more. The sensible production shape is a router by default and the agent
on demand. The trigger is `needed_second_hop`, which we already detect.
Measure it as a third row of the table: quality, steps, tokens.

### 39. The agent loops on bq5
In one run out of two: search_docs, search_docs, sql_query, search_docs x3, hit the round limit,
returned an empty answer. The protection worked, but the question went unanswered.
Examine the trace: why the repeated searches are not recognised as useless.

### 40. Injection defence: the hole is measured, the defence is not built yet
Decision 26. Crude attacks are repelled by the model on its own; the subtle one (a false fact with
no commands) went through in both schemes. Candidates, from cheap to expensive:
(a) an instruction in the system prompt: text from tools is DATA, not commands; on a contradiction
    between documents report the contradiction rather than choosing;
(b) require a reference to the governing document in the answer and check the
    `version` / `supersedes` / `effective_from` chain;
(c) a separate pass reconciling the numbers in the answer against several sources.
**Measure with the same five attacks, before and after.** Otherwise it is hope, not a defence.

### 41. Detecting contradictions between documents
A side conclusion of decision 26: the agent produced an answer citing BOTH the forged and the real
document at once, and did not notice the contradiction. This is a separate class of task, useful
even without any attack: the corpus contains the superseded POL-RET-001 (10 days) and the current
POL-RET-002 (14 days), and the system must not confuse those either.

### 42. The authority defence trusts the contents of the document
Decision 27 works because the forged document did not claim a version. An attacker who writes
`version: 9.9` and `supersedes: POL-SLA-001` walks straight through it. The real trust boundary is
the document's PROVENANCE, not its text: who may write into the corpus, the signature, the intake
channel. Test this directly: add a sixth attack with forged authority fields and measure.

### 43. The router remains weaker than the agent in robustness too
Decision 27: 3/5 against the agent's 5/5. One attack lands in UNCLEAR. Not examined separately:
the router is a baseline, not a production scheme. But if the hybrid (item 38) is adopted, the
hybrid's robustness has to be measured on its own rather than inherited from the agent.

### 44. bq5 is unstable: one question produces the entire spread of the set
Decision 29. In one run out of three the agent returned 6,449 instead of 7,073, spending 6 steps
and hitting the round limit. The number matches no subset of the required categories, so the query
was a third thing. Traces are now saved — examine it on the next run.

### 45. The agent does not reuse data it has already received
Decision 30: it searches for the authority metadata that is already printed in the previous step's
output. Candidates: condense what has been received into a short summary before the next round;
state explicitly in the prompt that the metadata is already attached to every passage.
Measure by step count and tokens, not by impression.

### 46. The latency budget is incomplete: the LLM contribution is missing
Decision 32: the local stages are measured conclusively, the model calls are not — the quota ran
out. Top it up with `LLM_CACHE=0 make latency` when a quota is available; the benchmark is already
failure-tolerant and writes traces incrementally.

### 47. The reranker is the only target for latency optimisation
98% of search time. Candidates with an expected effect:
(a) a smaller model (`bge-reranker-base`) — paid for in quality;
(b) ONNX or quantisation — paid for in integration time;
(c) window 10 instead of 20 — the curve is already measured in M2, quality falls little;
(d) do not always rerank: skip when the top-1 of dense retrieval is far ahead of the top-2.
Measure as the pair "recall@5 against p95", not separately.

### 48. CLOSED. The Langfuse exporter works
Decision 33. 25 traces were sent and verified by querying the server's API: structure and durations
preserved. The tool's limitation (absolute timestamps = export time) is stated in the decision.

### 49. Conditional reranking is implemented as a measurement but not wired into the tool
Decision 35: threshold 0.06 gives the same quality 12% faster. The rule is NOT added to
`src/tools/search.py`: the gain is modest and the price is a third branch plus a parameter that
needs recalibrating on a model change. Worth switching on once there is a hard latency budget, and
together with a test that the threshold is still current.

### 50. The confidence gap is not the only possible signal
The distribution is narrow (median 0.030, maximum 0.088), so skipping the rerank is rarely
possible. Replacement candidates: a relative gap instead of an absolute one, entropy over the
top-5, agreement between dense retrieval and BM25. Measure with the same table: "share reranked
against the two recalls".
