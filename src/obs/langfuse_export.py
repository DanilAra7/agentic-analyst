"""Exporting our own traces to Langfuse.

Why an exporter rather than instrumenting with Langfuse directly. The numbers in
the README must reproduce for anyone who clones the repository, WITHOUT signing
up for an external service (decision #31). So the source of truth is
`evals/traces.jsonl`, and Langfuse gets a copy. A side benefit: exporting can be
done after the fact, including traces recorded before the account existed.

How the concepts map:
    our Trace           -> a root observation of type agent
    span kind=llm       -> generation (Langfuse counts tokens and cost from it)
    span kind=tool      -> tool
    span kind=retrieval -> retriever
    span kind=rerank    -> span
Nesting is reconstructed from the `depth` field: spans lie in a flat list in the
order they were opened, so a stack of depths determines the tree unambiguously.

Times come from the trace, not from the moment of export: otherwise Langfuse
would receive the duration of the export itself rather than the measured one.
"""
from __future__ import annotations

import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path

from src.config import EVALS

KIND_TO_TYPE = {"llm": "generation", "tool": "tool",
                "retrieval": "retriever", "rerank": "span"}


def _host() -> str:
    return (os.getenv("LANGFUSE_HOST") or os.getenv("LANGFUSE_BASE_URL")
            or "https://cloud.langfuse.com")


def _client():
    from langfuse import Langfuse

    pk, sk = os.getenv("LANGFUSE_PUBLIC_KEY"), os.getenv("LANGFUSE_SECRET_KEY")
    if not (pk and sk):
        raise SystemExit(
            "LANGFUSE_PUBLIC_KEY / LANGFUSE_SECRET_KEY are missing from .env.\n"
            "Keys are created inside a project on cloud.langfuse.com "
            "(Settings -> API keys).\n"
            "To see what would be sent without sending it: DRY_RUN=1 make langfuse")
    # Langfuse separates regions by address (eu / us), and the SDK reads
    # LANGFUSE_HOST. In .env the variable is often called LANGFUSE_BASE_URL -
    # we accept both names, otherwise traces silently go to the wrong region.
    c = Langfuse(public_key=pk, secret_key=sk, host=_host())
    if not c.auth_check():
        raise SystemExit("Langfuse rejected the keys: check the public/secret pair and host.")
    return c


def load(path: Path) -> list[dict]:
    if not path.exists():
        raise SystemExit(f"{path} is missing. Collect traces first: LLM_CACHE=0 make latency")
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def _tree(spans: list[dict]) -> list[tuple[dict, int]]:
    """(span, parent index). The parent is the nearest preceding span with a
    smaller depth; -1 means the root."""
    out, stack = [], []
    for i, sp in enumerate(spans):
        while stack and spans[stack[-1]]["depth"] >= sp["depth"]:
            stack.pop()
        out.append((sp, stack[-1] if stack else -1))
        stack.append(i)
    return out


def export_one(client, tr: dict) -> None:
    """Send one trace.

    A LIMITATION that has to be said out loud. In Langfuse v4 an observation
    starts "now": `start_observation` does not take a start time, only `end_time`
    can be set. So the absolute timestamps in the UI are the time of the EXPORT,
    not of the measurement. DURATIONS are preserved exactly - and those are what
    the analysis needs. The real moment of measurement goes into the metadata so
    that it is not lost.

    Closing order matters: children are closed BEFORE their parents, otherwise
    the nesting in the UI falls apart. So we create in forward order and close in
    reverse.
    """
    root = client.start_observation(
        name=f"{tr['scheme']}:{tr['label']}", as_type="agent",
        input={"label": tr["label"]},
        metadata={"scheme": tr["scheme"], "total_ms": round(tr["total_ms"]),
                  "measured_at": datetime.fromtimestamp(tr["started_at"],
                                                        tz=timezone.utc).isoformat(),
                  "trace_id_local": tr["trace_id"],
                  **{k: v for k, v in (tr.get("outcome") or {}).items()}})

    made: list = []
    for sp, parent in _tree(tr["spans"]):
        attrs = sp.get("attrs") or {}
        kind = KIND_TO_TYPE.get(sp["kind"], "span")
        owner = made[parent][0] if parent >= 0 else root
        kw = {"name": sp["name"], "as_type": kind, "metadata": attrs}
        if kind == "generation":
            kw["model"] = attrs.get("model")
            kw["usage_details"] = {"input": attrs.get("prompt_tokens", 0),
                                   "output": attrs.get("completion_tokens", 0)}
        made.append((owner.start_observation(**kw), sp["ms"]))

    for o, ms in reversed(made):
        o.end(end_time=time.time_ns() + int(ms * 1e6))
    root.end(end_time=time.time_ns() + int(tr["total_ms"] * 1e6))


def dry_run(traces: list[dict]) -> None:
    """Show what would be sent without sending anything. Needed to check the
    tree assembly and the type mapping without an account."""
    print("DRY RUN: nothing is sent\n")
    for tr in traces[:3]:
        print(f"agent  {tr['scheme']}:{tr['label']}   {tr['total_ms']:.0f} ms")
        for sp, parent in _tree(tr["spans"]):
            pad = "  " * (sp["depth"] + 1)
            t = KIND_TO_TYPE.get(sp["kind"], "span")
            print(f"{pad}{t:<11}{sp['name']:<14}{sp['ms']:>8.0f} ms"
                  f"   parent: {'root' if parent < 0 else tr['spans'][parent]['name']}")
        print()
    kinds: dict[str, int] = {}
    for tr in traces:
        for sp in tr["spans"]:
            k = KIND_TO_TYPE.get(sp["kind"], "span")
            kinds[k] = kinds.get(k, 0) + 1
    print(f"traces in total: {len(traces)}   observations by type: {kinds}")


def main() -> None:
    path = EVALS / "traces.jsonl"
    traces = load(path)
    if os.getenv("DRY_RUN") == "1":
        dry_run(traces)
        return

    client = _client()
    for tr in traces:
        export_one(client, tr)
    client.flush()
    print(f"traces sent: {len(traces)}")
    print(f"view at: {_host()}")


if __name__ == "__main__":
    main()
