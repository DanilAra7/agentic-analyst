.PHONY: setup data corpus chunks index golden eval ablation doc-level \
	sql-ablation router agent injection latency latency-local langfuse clean help

help:           ## list the commands
	@grep -E '^[a-z-]+:.*?##' $(MAKEFILE_LIST) | awk 'BEGIN{FS=":.*?## "};{printf "  \033[36m%-12s\033[0m %s\n", $$1, $$2}'

setup:          ## install dependencies
	uv sync --extra dev

data:           ## download Olist and build the DuckDB database
	uv run python -m src.ingest.download
	uv run python -m src.ingest.build_db

corpus:         ## generate the document corpus (deterministic)
	uv run python -m src.ingest.build_docs

chunks:         ## split the corpus into chunks
	uv run python -m src.rag.chunking

index:          ## compute chunk embeddings
	uv run python -m src.rag.index

golden:         ## synthesise the golden set (needs an LLM key)
	uv run python -m src.eval.golden

eval:           ## retrieval metrics on both golden sets
	uv run python -m src.eval.retrieval

ablation:       ## reranker ablation: quality against latency
	uv run python -m src.eval.ablation_rerank

doc-level:      ## hierarchical retrieval measurement (decision 14)
	uv run python -m src.eval.ablation_doc_level

sql-ablation:   ## text-to-SQL across four schema description levels
	uv run python -m src.eval.ablation_sql

router:         ## agent baseline: one round of tool calls
	uv run python -m src.eval.ablation_agent router

agent:          ## the agent loop, with chained calls
	uv run python -m src.eval.ablation_agent agent

injection:      ## five planted attacks; AGENT_DEFENSE=0 turns the defence off
	uv run python -m src.eval.injection

latency:        ## per-stage latency budget (cache must be off)
	LLM_CACHE=0 uv run python -m src.eval.latency

latency-local:  ## local stages only: no provider quota needed, writes traces
	LAT_LOCAL_ONLY=1 uv run python -m src.eval.latency

langfuse:       ## send traces to Langfuse; DRY_RUN=1 shows them without sending
	uv run python -m src.obs.langfuse_export

clean:
	rm -rf data/*.duckdb data/*.npy llm_cache.sqlite evals/rerank_scores_*.json
