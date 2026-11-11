"use strict";

// Only observable API data is rendered. No invented reasoning or inferred
// pairing of repeated tool names with unrelated execution events.
window.TraceView = (() => {
  const tools = {
    report_search: "Поиск в отчётах", read_report: "Чтение страницы PDF",
    web_search: "Веб-поиск и котировки", price_forecast: "Расчёт прогноза",
    finish: "Перейти к ответу",
  };
  const statuses = {
    ok: "Ответ получен", partial: "Неполный ответ", clarification: "Нужно уточнение",
    outside: "Вне специализации", unavailable: "Сервис недоступен",
  };
  const stops = {
    completed: "Завершён", budget_exhausted: "Достигнут лимит шагов",
    llm_unavailable: "LLM недоступна или ответ не прошёл схему",
  };
  const intents = { research: "Анализ источников", forecast: "Прогноз", outside: "Вне специализации" };
  const node = (tag, text, cls) => {
    const el = document.createElement(tag);
    if (text !== undefined) el.textContent = text;
    if (cls) el.className = cls;
    return el;
  };
  const json = value => node("pre", JSON.stringify(value, null, 2), "trace-json");
  const detail = (label, child, cls = "inspector-detail") => {
    const el = node("details", undefined, cls);
    el.append(node("summary", label), child);
    return el;
  };
  const duration = ms => Number.isFinite(ms) ? (ms < 1000 ? `${ms} мс` : `${(ms / 1000).toFixed(1)} с`) : "—";
  function safeURL(value) {
    try {
      const url = new URL(value, location.origin);
      return ["http:", "https:"].includes(url.protocol) ? url.href : null;
    } catch { return null; }
  }
  function errorText(value) {
    if (!value) return "";
    if (value.startsWith("Duplicate action rejected")) return "Повтор того же вызова отклонён. Агенту предложено изменить запрос или открыть другую страницу.";
    if (value.startsWith("Action rejected by policy")) return "Действие отклонено правилами доступа к инструментам.";
    if (value.startsWith("Use a registered report ID")) return "Отчёт не найден среди доступных источников. Вызов отклонён.";
    return value;
  }
  function planView(plan) {
    const box = node("div", undefined, "plan-content");
    if (!plan) { box.append(node("p", "План не передан сервером.", "small")); return box; }
    const rows = [
      ["Задача", intents[plan.intent] || plan.intent],
      ["Поисковый запрос", plan.rag_query || "—"],
      ["Контекст из источников", plan.wants_context ? "Запрошен" : "Не запрошен"],
      ["Только отчёты", plan.report_only ? "Да" : "Нет"],
      ["Актуальные данные", plan.fresh ? "Нужны" : "Не запрошены"],
    ];
    if (plan.quotes?.length) rows.push(["Котировки", plan.quotes.join(", ")]);
    if (plan.intent === "forecast") {
      for (const [key, label] of [["benchmark", "Сорт"], ["months", "Горизонт, мес."], ["discount", "Дисконт, USD/барр."], ["cut", "Сокращение, млн б/с"], ["sensitivity", "Чувствительность"]]) {
        if (plan[key] !== null && plan[key] !== undefined) rows.push([label, plan[key]]);
      }
    }
    if (plan.clarify) rows.push(["Уточнение", plan.clarify]);
    const dl = node("dl", undefined, "trace-fields");
    for (const [label, value] of rows) dl.append(node("dt", label), node("dd", String(value)));
    box.append(dl, detail("Полный план JSON", json(plan)));
    return box;
  }
  function stepView(decision, execution, index) {
    const li = node("li", undefined, "trace-step");
    const title = node("div", undefined, "step-heading");
    title.append(node("span", String(index), "step-number"), node("strong", tools[decision.action] || decision.action));
    const state = execution?.status;
    const badgeText = state === "rejected" ? "Отклонён" : state === "error" ? "Ошибка" : state === "ok" ? duration(execution.ms) : decision.action === "finish" ? "Решение" : "Нет результата";
    title.append(node("span", badgeText, "step-state " + (state || "decision")));
    li.append(title);
    if (decision.purpose) li.append(node("p", "Цель по плану: " + decision.purpose, "step-purpose"));
    if (execution) {
      if (execution.error) li.append(node("p", errorText(execution.error), "step-error"));
      if (execution.result_count !== null && execution.result_count !== undefined) {
        li.append(node("p", `Найдено фрагментов: ${execution.result_count}${execution.source_ids?.length ? " · " + execution.source_ids.join(", ") : ""}`, "step-result"));
      }
      if (execution.result) li.append(node("p", `${execution.result.benchmark || ""} · история по ${execution.result.as_of || "—"} · последняя цена ${execution.result.last ?? "—"} USD/барр.`, "step-result"));
      for (const notice of execution.notices || []) li.append(node("p", notice, "step-notice"));
      li.append(detail("Аргументы и результат шага", json(execution), "step-detail"));
    } else if (decision.action !== "finish") {
      li.append(node("p", "Подробный результат этого шага не передан. Решение не означает, что инструмент был выполнен.", "small"));
    }
    return li;
  }
  function sourcesView(sources) {
    const box = node("div", undefined, "retrieved-list");
    if (!sources.length) box.append(node("p", "В диагностике нет найденных фрагментов.", "small"));
    for (const s of sources) {
      const content = node("div");
      content.append(node("p", [s.date || "Дата не указана", s.page ? `стр. PDF ${s.page}` : null, Number.isFinite(s.score) ? `score ${s.score.toFixed(4)}` : null].filter(Boolean).join(" · "), "small"));
      content.append(node("p", s.used_in_answer ? "Использован в ответе" : "Найден, но не указан в ответе", "source-use"));
      content.append(node("pre", s.text, "source-preview"));
      if (s.text_truncated) content.append(node("p", "Показаны первые 4000 символов. Полный текст доступен в документе.", "small"));
      const href = safeURL(s.local_url || s.url);
      if (href) {
        const a = node("a", "Открыть источник ↗");
        a.href = href; a.target = "_blank"; a.rel = "noopener noreferrer";
        content.append(a);
      }
      box.append(detail(`[${s.id}] ${s.title}`, content, "retrieved-source"));
    }
    return box;
  }
  function render(data, elapsed) {
    const box = node("aside", undefined, "trace-panel");
    box.setAttribute("aria-label", "Ход запроса");
    const heading = node("div", undefined, "trace-heading");
    heading.append(node("h3", "Ход запроса"), node("span", "TRACE", "trace-kicker"));
    box.append(heading, node("p", "Фактические действия и краткие пояснения выбора, переданные моделью.", "small"));
    const metrics = node("div", undefined, "trace-metrics");
    for (const [value, label] of [[duration(elapsed), "запрос в браузере"], [(data.trace || []).length, "вызовов инструментов"], [(data.decisions || []).length, "решений модели"]]) {
      const metric = node("div"); metric.append(node("strong", String(value)), node("span", label)); metrics.append(metric);
    }
    box.append(metrics);
    const diagnostics = data.diagnostics;
    box.append(detail("Как понят запрос", planView(diagnostics?.plan)));
    const timeline = node("ol", undefined, "trace-timeline");
    const decisions = data.decisions || [];
    const byStep = new Map((diagnostics?.steps || []).map(s => [s.step, s]));
    decisions.forEach((d, i) => timeline.append(stepView(d, byStep.get(i + 1), i + 1)));
    if (decisions.length) box.append(timeline);
    else box.append(node("p", (data.trace || []).length ? "Решения модели не переданы сервером." : "Вызовов инструментов нет. Причину смотрите в ответе и плане.", "small empty-trace"));
    if (!diagnostics) {
      box.append(node("p", "Подробная диагностика недоступна в этой версии ответа. Показаны только переданные сервером данные.", "trace-legacy small"));
      if (data.trace?.length) {
        const list = node("ul", undefined, "legacy-tools");
        for (const e of data.trace) list.append(node("li", `${tools[e.tool] || e.tool} · ${e.status} · ${duration(e.ms)}`));
        box.append(detail("Выполненные инструменты", list));
      }
    }
    const stop = node("div", undefined, "trace-stop");
    stop.append(node("span", "Состояние: "), node("strong", statuses[data.status] || data.status), node("p", "Завершение: " + (stops[data.stop_reason] || data.stop_reason || "не указано")));
    if (diagnostics?.failure) stop.append(node("p", diagnostics.failure, "step-error"));
    if (diagnostics?.answer_feedback) stop.append(detail("Замечание проверки ответа", node("p", diagnostics.answer_feedback)));
    box.append(stop);
    if (diagnostics) box.append(detail(`Найденные фрагменты · ${diagnostics.retrieved_sources?.length || 0}`, sourcesView(diagnostics.retrieved_sources || [])));
    box.append(node("p", "Пояснение модели и статус выполнения не подтверждают правильность вывода. Сверяйте ответ с источниками.", "trace-footnote"));
    if (data.trace_id) {
      const id = node("div", undefined, "trace-id");
      id.append(node("span", "Trace ID"), node("code", data.trace_id));
      const copy = node("button", "Копировать ID", "text-button"); copy.type = "button";
      copy.addEventListener("click", async () => {
        try { await navigator.clipboard.writeText(data.trace_id); copy.textContent = "Скопировано"; }
        catch { copy.textContent = "Выделите ID для копирования"; }
      });
      id.append(copy); box.append(id);
    }
    const raw = detail("Полный ответ API · JSON", json(data));
    const download = node("button", "Скачать JSON", "download-button"); download.type = "button";
    download.addEventListener("click", () => {
      const url = URL.createObjectURL(new Blob([JSON.stringify(data, null, 2)], { type: "application/json" }));
      const a = node("a"); a.href = url; a.download = `agent-trace-${String(data.trace_id || Date.now()).replace(/[^a-zA-Z0-9_-]/g, "")}.json`;
      a.click(); setTimeout(() => URL.revokeObjectURL(url), 1000);
    });
    raw.append(download); box.append(raw);
    return box;
  }
  return { render, safeURL, statusLabel: value => statuses[value] || value };
})();
