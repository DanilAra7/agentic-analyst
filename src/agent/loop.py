"""Agent loop: several rounds of tool calls, with chaining.

Exactly one thing separates it from the router, and that is deliberate: the next
call may depend on the result of the previous one. The system prompt, the tool
set, response parsing and metric accounting are identical. Otherwise the
comparison would measure a difference in prompts rather than agency.

Three guards, without which the loop must not ship (backlog #9):

1. ROUND LIMIT. The model may never stop on its own. A two-source question needs
   two rounds; six leaves room to recover from mistakes while still bounding the
   worst case.

2. REPEAT DETECTION. The most common way to spin forever is to repeat the same
   call and get the same result. A repeat is not executed: instead of a result
   the model gets text explaining that this call was already made. That is
   cheaper than the round limit and does not burn calls.

3. ERRORS AS TEXT, not as exceptions (in src/agent/base.py). The line
   "Binder Error: column X does not exist" teaches the model; an exception kills
   the loop.

What is NOT done, recorded honestly: there is no defence here against prompt
injection through tool results. Per decision #20 the attack is first planted in
the corpus and the agent measured against it, and only then is the defence
written. A defence nobody tried to break is not a defence.
"""
from __future__ import annotations

import json

from src.agent.base import (FINAL_SPEC, Step, Trace, assistant_msg, execute,
                            run_step, system_prompt)

MAX_ROUNDS = 6          # a round = one model reply, which may contain several calls
MAX_TOOL_CALLS = 10     # overall ceiling per question


def answer(llm, question: str) -> Trace:
    tr = Trace(question=question)
    messages = [{"role": "system", "content": system_prompt()},
                {"role": "user", "content": question}]
    seen: set[str] = set()

    for _ in range(MAX_ROUNDS):
        r = run_step(llm, messages, tr=tr)

        if not r.tool_calls:
            # The model answered with text instead of final_answer. This is the
            # fallback path: we accept the answer but tag it, so such cases stay
            # visible in the run instead of dissolving into overall accuracy.
            tr.answer = r.text
            tr.stop_reason = "plain_text_fallback"
            return tr

        messages.append(assistant_msg(r))
        for call in r.tool_calls:
            if call.name == "final_answer":
                tr.take_final(call.arguments)
                tr.steps.append(Step(call.name, call.arguments, "", True, 0.0))
                tr.stop_reason = "final_answer"
                return tr
            key = f"{call.name}:{json.dumps(call.arguments, sort_keys=True, ensure_ascii=False)}"
            if key in seen:
                out, ok = ("TOOL ERROR: you already made this exact call and received the "
                           "result above. Use it, or make a different call, or answer."), False
            elif len(tr.steps) >= MAX_TOOL_CALLS:
                out, ok = ("TOOL ERROR: tool call budget exhausted. Answer with what you "
                           "have, or state what is missing."), False
            else:
                seen.add(key)
                out, ok = execute(call.name, call.arguments)
            tr.steps.append(Step(call.name, call.arguments, out, ok, 0.0))
            messages.append({"role": "tool", "tool_call_id": call.id,
                             "name": call.name, "content": out})

    # Rounds are exhausted. Ask for an answer from what is already there: silence
    # would be a failure of the implementation rather than a limit of the scheme,
    # and we would not be able to tell the two apart.
    tr.stop_reason = "max_rounds"
    messages.append({"role": "user", "content":
                     "You have no more tool calls. Answer in plain text with what you "
                     "have, or state exactly what is missing."})
    r = run_step(llm, messages, tools=[FINAL_SPEC], tr=tr)
    for call in r.tool_calls:
        if call.name == "final_answer":
            tr.take_final(call.arguments)
            return tr
    tr.answer = r.text
    return tr


def main() -> None:
    from src.config import settings
    from src.llm.providers import OpenAICompatProvider

    llm = OpenAICompatProvider(settings.llm_provider, settings.llm_model)
    for q in ("Under the delivery delay compensation policy, how many orders qualify "
              "for the highest compensation tier?",
              "For the state with the most orders, what is the designated sorting hub?",
              "How many unique customers placed at least one order?"):
        tr = answer(llm, q)
        print("=" * 78)
        print("Q:", q)
        for i, s in enumerate(tr.steps, 1):
            print(f"  step {i}: {s.tool}({str(s.arguments)[:86]})")
            print(f"          -> {s.result[:96].replace(chr(10),' ')}")
        print(f"llm calls: {tr.llm_calls}   steps: {len(tr.steps)}   stop: {tr.stop_reason}")
        print("A:", (tr.answer or "").strip()[:300])


if __name__ == "__main__":
    main()
