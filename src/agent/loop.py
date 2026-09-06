"""Цикл агента: несколько раундов вызовов инструментов со сцеплением.

Отличие от роутера ровно одно и оно намеренно единственное: следующий вызов
может зависеть от результата предыдущего. Системный промпт, набор инструментов,
разбор ответа и подсчёт метрик - те же самые. Иначе сравнение мерило бы разницу
промптов, а не агентность.

Три защиты, без которых цикл нельзя выпускать (бэклог №9):

1. ЛИМИТ РАУНДОВ. Модель может не остановиться никогда. Двухисточниковый вопрос
   требует двух раундов; шесть даёт запас на исправление ошибок и при этом
   ограничивает худший случай.

2. ДЕТЕКЦИЯ ПОВТОРОВ. Самый частый способ зациклиться - повторять один и тот же
   запрос, получая один и тот же ответ. Повтор не выполняется: вместо результата
   возвращается текст, объясняющий, что этот вызов уже был. Это дешевле лимита
   раундов и не тратит вызовы впустую.

3. ОШИБКИ ТЕКСТОМ, а не исключением (в src/agent/base.py). Строка
   «Binder Error: column X does not exist» учит модель; исключение убивает цикл.

Что НЕ сделано и записано честно: защиты от prompt injection через результаты
инструментов здесь нет. По решению №20 сначала атака закладывается в корпус и
измеряется, поддаётся ли агент, и только потом пишется защита. Защита, которую
не пробовали пробить, защитой не является.
"""
from __future__ import annotations

import json

from src.agent.base import Step, Trace, assistant_msg, execute, run_step, system_prompt

MAX_ROUNDS = 6          # раунд = один ответ модели, в нём может быть несколько вызовов
MAX_TOOL_CALLS = 10     # общий потолок на вопрос


def answer(llm, question: str) -> Trace:
    tr = Trace(question=question)
    messages = [{"role": "system", "content": system_prompt()},
                {"role": "user", "content": question}]
    seen: set[str] = set()

    for _ in range(MAX_ROUNDS):
        r = run_step(llm, messages, tr=tr)

        if not r.tool_calls:
            tr.answer = r.text
            tr.stop_reason = "answered"
            return tr

        messages.append(assistant_msg(r))
        for call in r.tool_calls:
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

    # Раунды кончились. Просим ответить тем, что есть: молчание было бы провалом
    # реализации, а не пределом схемы, и мы бы не отличили одно от другого.
    tr.stop_reason = "max_rounds"
    messages.append({"role": "user", "content":
                     "You have no more tool calls. Answer in plain text with what you "
                     "have, or state exactly what is missing."})
    r = run_step(llm, messages, tools=None, tr=tr)
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
            print(f"  шаг {i}: {s.tool}({str(s.arguments)[:86]})")
            print(f"          -> {s.result[:96].replace(chr(10),' ')}")
        print(f"вызовов LLM: {tr.llm_calls}   шагов: {len(tr.steps)}   стоп: {tr.stop_reason}")
        print("A:", (tr.answer or "").strip()[:300])


if __name__ == "__main__":
    main()
