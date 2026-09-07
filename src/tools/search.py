"""Инструмент поиска по документам: обёртка над лучшей конфигурацией M2.

Конфигурация не выбирается здесь заново - она измерена и записана в решениях
№11-13. Кратко, почему именно такая:

  плотный поиск (bge-m3) достаёт CANDIDATES кандидатов
    -> cross-encoder переупорядочивает первые RERANK_WINDOW
      -> наверх идут TOP_K

Реранкер ВКЛЮЧЁН, потому что потребитель выдачи - агент, который читает
несколько документов сразу: для него важен recall@5 (0.863 -> 0.941), а не
recall@1. Если бы мы показывали пользователю один готовый ответ, реранкер надо
было бы ВЫКЛЮЧИТЬ: он роняет recall@1 с 0.765 до 0.667. Единственной лучшей
конфигурации не существует, и этот файл - место, где сделан выбор под задачу.

Окно 20, а не 50: переход к 50 покупает +0.020 recall за +2.1 секунды. Колено
кривой на 20.
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

CANDIDATES = 50        # сколько достаёт плотный поиск
RERANK_WINDOW = 20     # сколько из них переупорядочивает cross-encoder
TOP_K = 5              # сколько уходит в контекст модели
SNIPPET_CHARS = 1800   # обрезка текста куска: размен контекста на полноту
# Было 700 и это ломало ответы: обрезка приходилась на середину таблицы окон
# возврата по категориям, и строки telephony, auto, musical_instruments модель
# не видела вовсе. Она отвечала честно по тому, что ей дали, и промахивалась.
# 1800 покрывает самый длинный чанк (400 токенов). Это ручка размена
# «токены против полноты», её стоит померить отдельно - бэклог №34.


# Поля, по которым определяется, какой документ формально главнее. Они лежали
# в шапке документа с самого M1 и никогда не доходили до модели: в текст чанка
# дописывался только title. Из-за этого агент, получив правило «предпочитай
# документ с версией и цепочкой отмены», честно ответил «ни у одного из них
# версии нет» - хотя у настоящего регламента version 1.4, а у подложного ничего.
# Правило без доказательств не работает. Решение №27.
AUTHORITY = ("version", "effective_from", "status", "supersedes")


@dataclass
class Passage:
    chunk_id: str
    doc_id: str
    title: str
    text: str
    score: float
    meta: dict | None = None


class DocSearchTool:
    def __init__(self, use_reranker: bool = True) -> None:
        self.use_reranker = use_reranker

    @property
    def _retriever(self):
        from src.rag.retrieve import get_retriever
        return get_retriever()

    @property
    def _reranker(self):
        from src.rag.rerank import get_reranker
        return get_reranker()

    def search(self, query: str, top_k: int = TOP_K) -> list[Passage]:
        hits = self._retriever.search(query, k=CANDIDATES if self.use_reranker else top_k)
        if self.use_reranker:
            head = hits[:RERANK_WINDOW]
            scores = self._reranker.score(query, [h.text for h in head])
            order = sorted(range(len(head)), key=lambda i: -float(scores[i]))
            picked = [(head[i], float(scores[i])) for i in order[:top_k]]
        else:
            picked = [(h, h.score) for h in hits[:top_k]]

        return [Passage(h.chunk_id, h.chunk_id.split("#")[0],
                        str(h.meta.get("title", "")), h.text, s, h.meta)
                for h, s in picked]

    def as_text(self, query: str, top_k: int = TOP_K) -> str:
        """Как результат выглядит для модели.

        Каждый фрагмент подписан идентификатором документа: без этого модель
        не сможет сослаться на источник, а ответ без ссылки на источник в
        справочной системе бесполезен - его нечем проверить.
        """
        ps = self.search(query, top_k)
        if not ps:
            return "No matching passages found."
        out = []
        for i, p in enumerate(ps, 1):
            body = p.text[:SNIPPET_CHARS] + ("..." if len(p.text) > SNIPPET_CHARS else "")
            m = p.meta or {}
            auth = ", ".join(f"{k}={m[k]}" for k in AUTHORITY if m.get(k))
            head = f"[{i}] {p.doc_id} - {p.title}"
            if auth:
                head += f"\n    ({auth})"
            else:
                head += "\n    (no version, no effective date, no supersedes chain)"
            out.append(f"{head}\n{body}")
        return "\n\n".join(out)


@lru_cache(maxsize=2)
def get_search_tool(use_reranker: bool = True) -> DocSearchTool:
    return DocSearchTool(use_reranker)


TOOL_SPEC = {
    "type": "function",
    "function": {
        "name": "search_docs",
        "description": (
            "Search the company's policy and operations documents: shipping rates and "
            "deadlines, return and refund policy, regional handbooks, category rules, "
            "help-centre entries. Use it for rules, procedures and published figures. "
            "Returns the most relevant passages with their document ids."),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string",
                          "description": "A natural-language search query."}},
            "required": ["query"],
        },
    },
}


def main() -> None:
    tool = get_search_tool()
    for q in ("How much does it cost to ship 2 kg to Bahia?",
              "When does Saturday count as a working day?"):
        print("=" * 74)
        print("Q:", q)
        for p in tool.search(q):
            print(f"   {p.score:>7.2f}  {p.chunk_id:<20} {p.title[:46]}")


if __name__ == "__main__":
    main()
