"""Роутер: бейзлайн для агента. Ровно ОДИН заход в инструменты.

Зачем он существует (решение №18). Если роутера достаточно, то агент строится
ради агента, и узнать это надо ДО недели работы над циклом. Роутер - это
проверка посылки проекта, а не разминка.

Отличие от агента ровно одно: РОВНО ОДИН РАУНД вызовов инструментов. Системный
промпт, набор инструментов и разбор ответа - те же самые, иначе сравнение мерило
бы их разницу.

Важное уточнение, найденное на первом же прогоне. «Один раунд» не значит «один
инструмент»: модель вправе вызвать оба сразу, параллельно, и она так делает.
Значит роутер отличается от агента не числом инструментов, а невозможностью
СЦЕПЛЕНИЯ: второй вызов не может зависеть от результата первого.
Именно сцепление и есть то, что покупается агентностью, и именно его мы меряем.

Второе наблюдение оттуда же: получив результат документа, модель пытается
вызвать SQL даже когда инструменты ей не передали - возвращает пустой content и
tool_calls. Это структурный предел роутера в чистом виде, и мы его считаем
отдельной величиной `needed_second_hop`: доля вопросов, где одного раунда не
хватило. Она измеряет потребность в агентности НАПРЯМУЮ, не через качество ответа.

Ожидание, назначенное заранее:
  sql / docs   роутер должен справляться наравне с агентом
  both         роутер должен ПРОВАЛИТЬСЯ: одного захода не хватает
  none         должен отказаться
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
    for call in r.tool_calls:                 # роутер выполняет то, что запросили за один раз
        if call.name == "final_answer":       # ответил, не заглядывая в источники
            tr.take_final(call.arguments)
            tr.stop_reason = "final_answer"
            return tr
        out, ok = execute(call.name, call.arguments)
        tr.steps.append(Step(call.name, call.arguments, out, ok, 0.0))
        messages.append({"role": "tool", "tool_call_id": call.id,
                         "name": call.name, "content": out})

    # Второй раунд запрещён: инструменты не передаём.
    messages.append({"role": "user", "content":
                     "Now answer the original question using only what you already have. "
                     "If it is not enough, say plainly what is missing."})
    # Второй раунд ДОБЫЧИ запрещён, но выйти надо тем же типизированным
    # действием, что и у агента: иначе схемы отличались бы ещё и формой ответа.
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

    # Пустой ответ с попыткой вызвать инструмент = одного раунда не хватило.
    # Даём последний шанс сказать это словами: иначе провал был бы засчитан
    # за немоту реализации, а не за предел схемы.
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
        print(f"инструменты: {tr.tools_used or '—'}   вызовов LLM: {tr.llm_calls}   "
              f"{tr.ms:.0f} мс   стоп: {tr.stop_reason}")
        print("A:", (tr.answer or "").strip()[:400])


if __name__ == "__main__":
    main()
