"""Document search tool: a wrapper around the best configuration from M2.

The configuration is not chosen anew here - it was measured and recorded in
decisions #11-13. Briefly, why it looks like this:

  dense search (bge-m3) pulls CANDIDATES candidates
    -> a cross-encoder reorders the first RERANK_WINDOW of them
      -> TOP_K go upstream

The reranker is ON, because the consumer of the output is an agent that reads
several documents at once: what matters for it is recall@5 (0.863 -> 0.941), not
recall@1. If we were showing a user one finished answer, the reranker would have
to be OFF: it drops recall@1 from 0.765 to 0.667. There is no single best
configuration, and this file is where the choice is made for this task.

Window 20, not 50: going to 50 buys +0.020 recall for +2.1 seconds. The knee of
the curve is at 20.
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

from src.obs import trace as obs

CANDIDATES = 50        # how many dense search pulls
RERANK_WINDOW = 20     # how many of them the cross-encoder reorders
TOP_K = 5              # how many go into the model context
SNIPPET_CHARS = 1800   # chunk text cutoff: context traded against completeness
# It used to be 700 and that broke answers: the cutoff landed in the middle of
# the per-category return window table, and the model never saw the telephony,
# auto and musical_instruments rows at all. It answered honestly from what it was
# given, and got it wrong. 1800 covers the longest chunk (400 tokens). This is a
# "tokens against completeness" knob and deserves its own measurement - backlog #34.


# The fields that decide which document is formally the more authoritative one.
# They had been in the document header since M1 and never reached the model: only
# the title was prepended to the chunk text. Because of that the agent, given the
# rule "prefer the document with a version and a supersedes chain", honestly
# answered "neither of them has a version" - although the real policy has
# version 1.4 and the planted one has nothing. A rule without evidence does not
# work. Decision #27.
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
        # The two stages are measured SEPARATELY: dense search takes
        # milliseconds, the cross-encoder takes seconds. A single "search took
        # 2.5 s" is true and useless: it does not say what to cut.
        with obs.span("dense", "retrieval", k=CANDIDATES):
            hits = self._retriever.search(query, k=CANDIDATES if self.use_reranker else top_k)
        if self.use_reranker:
            head = hits[:RERANK_WINDOW]
            with obs.span("rerank", "rerank", window=RERANK_WINDOW):
                scores = self._reranker.score(query, [h.text for h in head])
            order = sorted(range(len(head)), key=lambda i: -float(scores[i]))
            picked = [(head[i], float(scores[i])) for i in order[:top_k]]
        else:
            picked = [(h, h.score) for h in hits[:top_k]]

        return [Passage(h.chunk_id, h.chunk_id.split("#")[0],
                        str(h.meta.get("title", "")), h.text, s, h.meta)
                for h, s in picked]

    def as_text(self, query: str, top_k: int = TOP_K) -> str:
        """What the result looks like to the model.

        Every passage is labelled with its document id: without it the model
        cannot cite a source, and in a reference system an answer with no source
        is useless - there is no way to check it.
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
