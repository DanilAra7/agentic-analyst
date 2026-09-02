.PHONY: setup data chunks index golden eval ablation clean help

help:           ## показать список команд
	@grep -E '^[a-z-]+:.*?##' $(MAKEFILE_LIST) | awk 'BEGIN{FS=":.*?## "};{printf "  \033[36m%-12s\033[0m %s\n", $$1, $$2}'

setup:          ## поставить зависимости
	uv sync --extra dev

data:           ## скачать Olist и собрать DuckDB
	uv run python -m src.ingest.download
	uv run python -m src.ingest.build_db

corpus:         ## сгенерировать корпус документов (373 шт., детерминированно)
	uv run python -m src.ingest.build_docs

chunks:         ## нарезать корпус на чанки
	uv run python -m src.rag.chunking

index:          ## посчитать эмбеддинги чанков
	uv run python -m src.rag.index

golden:         ## синтезировать golden set (требует ключа LLM)
	uv run python -m src.eval.golden

eval:           ## метрики поиска на обоих golden set
	uv run python -m src.eval.retrieval

ablation:       ## ablation по реранкеру: качество против латентности
	uv run python -m src.eval.ablation_rerank

doc-level:      ## замер иерархического поиска (решение №14)
	uv run python -m src.eval.ablation_doc_level

clean:
	rm -rf data/*.duckdb data/*.npy llm_cache.sqlite evals/rerank_scores_*.json
