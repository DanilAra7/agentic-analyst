.PHONY: setup data index golden eval agent clean

setup:          ## поставить зависимости
	uv sync --extra dev

data:           ## скачать Olist и собрать DuckDB
	uv run python -m src.ingest.download
	uv run python -m src.ingest.build_db

index:          ## посчитать эмбеддинги и построить индексы
	uv run python -m src.rag.index

golden:         ## синтезировать golden set
	uv run python -m src.eval.golden

eval:           ## прогнать ablation по всем конфигурациям
	uv run python -m src.eval.run

agent:          ## интерактивный запуск агента
	uv run python -m src.agent.loop

clean:
	rm -rf data/*.duckdb data/*.npy llm_cache.sqlite
