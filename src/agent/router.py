"""Router: the baseline for the agent. Exactly ONE trip to the tools.

Why it exists (decision #18). If the router is enough, then the agent is being
built for its own sake, and that has to be found out BEFORE a week of work on the
loop. The router is a test of the project's premise, not a warm-up.

Exactly one thing separates it from the agent: EXACTLY ONE ROUND of tool calls.
The system prompt, the tool set and response parsing are identical, otherwise the
comparison would measure their difference instead.

An important clarification, found on the very first run. "One round" does not mean
"one tool": the model is free to call both at once, in parallel, and it does.
So the router differs from the agent not in the number of tools but in the
impossibility of CHAINING: the second call cannot depend on the first one's
result. Chaining is exactly what agency buys, and chaining is what we measure.

A second observation from the same run: having received a document result, the
model tries to call SQL even when no tools were passed to it - returning empty
content and empty tool_calls. That is the router's structural limit in pure form,
and we count it as a separate quantity, `needed_second_hop`: the share of
questions where one round was not enough. It measures the need for agency
DIRECTLY, not through answer quality.

The expectation, fixed in advance:
  sql / docs   the router should do as well as the agent
  both         the router should FAIL: one trip is not enough
  none         should refuse
"""
from __future__ import annotations

from src.agent.base import (FINAL_SPEC, Step, Trace, assistant_msg, execute,
                            run_step, system_prompt)


def answer(llm, question: str) -> Trace:
    tr = Trace(question=question)
    messages = [{"role": "system", "content": system_prompt()},
                {"role": "user", "content": question}]

    r = run_step(llm, messages, tr=tr)

    if not r.tool_calls:
        tr.answer = r.text
        tr.stop_reason = "no_tool_call"
        return tr

    messages.append(assistant_msg(r))
    for call in r.tool_calls:                 # the router executes what was asked for in one go
        if call.name == "final_answer":       # answered without looking at any source
            tr.take_final(call.arguments)
            tr.stop_reason = "final_answer"
            return tr
        out, ok = execute(call.name, call.arguments)
        tr.steps.append(Step(call.name, call.arguments, out, ok, 0.0))
        messages.append({"role": "tool", "tool_call_id": call.id,
                         "name": call.name, "content": out})

    # A second round is forbidden: we pass no tools.
    messages.append({"role": "user", "content":
                     "Now answer the original question using only what you already have. "
                     "If it is not enough, say plainly what is missing."})
    # A second round of FETCHING is forbidden, but the exit must use the same
    # typed action as the agent: otherwise the schemes would also differ in the
    # shape of their answer.
    r2 = run_step(llm, messages, tools=[FINAL_SPEC], tr=tr)

    for call in r2.tool_calls:
        if call.name == "final_answer":
            tr.take_final(call.arguments)
            tr.stop_reason = "answered_after_one_hop"
            return tr

    if r2.text.strip():
        tr.answer = r2.text
        tr.stop_reason = "answered_after_one_hop"
        return tr

    # An empty reply with an attempted tool call = one round was not enough.
    # Give a last chance to say so in words: otherwise the failure would be
    # scored as muteness of the implementation rather than a limit of the scheme.
    tr.stop_reason = "needed_second_hop"
    messages.append({"role": "user", "content":
                     "You have no tools left and cannot call any. Answer in plain text: "
                     "give the answer if you have it, otherwise state exactly what "
                     "additional information you would need."})
    r3 = run_step(llm, messages, tools=[FINAL_SPEC], tr=tr)
    for call in r3.tool_calls:
        if call.name == "final_answer":
            tr.take_final(call.arguments)
            return tr
    tr.answer = r3.text
    return tr


def main() -> None:
    from src.config import settings
    from src.llm.providers import OpenAICompatProvider

    llm = OpenAICompatProvider(settings.llm_provider, settings.llm_model)
    for q in ("How many orders have the status 'delivered'?",
              "Under the delivery delay compensation policy, how many orders qualify "
              "for the highest compensation tier?",
              "How many orders were paid by credit card?"):
        tr = answer(llm, q)
        print("=" * 78)
        print("Q:", q)
        print(f"tools: {tr.tools_used or '-'}   llm calls: {tr.llm_calls}   "
              f"{tr.ms:.0f} ms   stop: {tr.stop_reason}")
        print("A:", (tr.answer or "").strip()[:400])


if __name__ == "__main__":
    main()
