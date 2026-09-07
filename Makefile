.PHONY: setup data corpus chunks index golden eval ablation doc-level \
	sql-ablation router agent injection latency latency-local langfuse clean help

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

sql-ablation:   ## text-to-SQL на четырёх уровнях описания схемы
	uv run python -m src.eval.ablation_sql

router:         ## бейзлайн агента: один раунд вызовов инструментов
	uv run python -m src.eval.ablation_agent router

agent:          ## цикл агента со сцеплением вызовов
	uv run python -m src.eval.ablation_agent agent

injection:      ## пять заложенных атак; AGENT_DEFENSE=0 отключает защиту
	uv run python -m src.eval.injection

latency:        ## бюджет латентности по этапам (только без кеша)
	LLM_CACHE=0 uv run python -m src.eval.latency

latency-local:  ## только локальные этапы: без квоты провайдера, пишет трассы
	LAT_LOCAL_ONLY=1 uv run python -m src.eval.latency

langfuse:       ## отправить трассы в Langfuse; DRY_RUN=1 показать без отправки
	uv run python -m src.obs.langfuse_export

clean:
	rm -rf data/*.duckdb data/*.npy llm_cache.sqlite evals/rerank_scores_*.json
