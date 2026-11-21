"use strict";

// Only observable API data is rendered. No invented reasoning or inferred
// pairing of repeated tool names with unrelated execution events.
window.TraceView = (() => {
  const tools = {
    report_search: "Поиск в отчётах", read_report: "Чтение страницы PDF",
    web_search: "Веб-поиск и котировки", read_web: "Чтение веб-страницы", price_forecast: "Расчёт прогноза",
    finish: "Перейти к ответу",
  };
  const statuses = {
    ok: "Ответ получен", partial: "Неполный ответ", clarification: "Нужно уточнение",
    outside: "Вне специализации", unavailable: "Сервис недоступен",
  };
  const stops = {
    completed: "Завершён", budget_exhausted: "Достигнут лимит шагов",
    llm_unavailable: "LLM недоступна или ответ не прошёл схему",
    no_progress: "Повторные попытки не добавили данных",
    time_budget_exhausted: "Достигнут общий лимит времени",
    research_budget_exhausted: "Поиск завершён с резервом времени на ответ",
    review_disabled: "Кандидат ответа получен; проверка смысла и полноты отключена",
  };
  const intents = { research: "Анализ источников", forecast: "Прогноз", capabilities: "Вопрос о возможностях агента", outside: "Вне специализации" };
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
    if (value.startsWith("Duplicate action rejected")) return "Повтор того же вызова отклонён. Предыдущий результат доступен; можно использовать его, изменить параметры или перейти к ответу.";
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
    if (plan.quotes?.length && plan.context_request) {
      rows.push(["Отдельный вопрос к источникам", plan.context_request]);
      if (typeof plan.context_fresh === "boolean") rows.push(["Свежие данные для этой части", plan.context_fresh ? "Нужны" : "Можно использовать отчёт за подходящий период"]);
    }
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
    if (decision.origin === "plan_policy") {
      const message = decision.quote_lookup ? "Модель выбрала котировку в плане. Агент запросил дневной ряд EIA/FRED для указанного сорта." : "Агент взял поисковый запрос из плана и вызвал инструмент без отдельного вызова LLM.";
      li.append(node("p", message, "small step-origin"));
    }
    if (decision.purpose) li.append(node("p", "Цель по плану: " + decision.purpose, "step-purpose"));
    if (decision.controller_note) li.append(node("p", `Модель отметила, что данных достаточно. Контроллер перевёл запрос к ответу вместо действия «${tools[decision.model_action] || decision.model_action}».`, "small"));
    if (decision.missing_facts?.length || decision.evidence_checks?.length) {
      const assessment = node("div");
      const list = node("ul");
      for (const check of decision.evidence_checks || []) list.append(node("li", `${check.fact} · ${(check.source_ids || []).join(", ") || "источник не указан"}`));
      assessment.append(list);
      if (decision.missing_facts?.length) assessment.append(node("p", "Не хватает: " + decision.missing_facts.join("; ")));
      if (decision.unknown_source_ids?.length) assessment.append(node("p", "Неизвестные ссылки: " + decision.unknown_source_ids.join(", "), "step-error"));
      assessment.append(node("p", "Это оценка модели. Наличие ссылки не подтверждает правильность толкования источника.", "small"));
      li.append(detail("Какие сведения найдены и чего не хватает", assessment, "step-detail"));
    }
    if (execution) {
      if (execution.argument_repair?.mode === "plan_rag_query") li.append(node("p", "Модель оставила поисковую строку пустой. Агент взял из плана фразу «" + execution.argument_repair.after.query + "».", "small"));
      if (execution.error) li.append(node("p", errorText(execution.error), "step-error"));
      if (execution.duplicate_of) li.append(node("p", `Повтор шага ${execution.duplicate_of}. Ранее получено: ${execution.reused_source_ids?.join(", ") || "фрагментов нет"}.`, "step-result"));
      if (typeof execution.new_evidence === "boolean") li.append(node("p", `Новые данные: ${execution.new_evidence ? "да" : "нет"}. Попыток без новых данных подряд: ${execution.no_progress_steps ?? 0}.`, "small"));
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
      if (s.provider) content.append(node("p", `Получение: ${s.provider}${s.cache_hit === true ? " · из кэша" : ""}.`, "small"));
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
  function routeView(data) {
    const box = node("section", undefined, "route-summary");
    box.setAttribute("aria-label", "Источники и маршрут ответа");
    box.append(node("h4", "Откуда взят ответ"));
    const labels = { report: "база знаний", web: "интернет", model: "расчёт" };
    const counts = new Map();
    for (const source of data.sources || []) {
      const label = labels[source.kind] || "другие источники";
      counts.set(label, (counts.get(label) || 0) + 1);
    }
    box.append(node("p", counts.size ? "Источники ответа: " + [...counts].map(([label, count]) => `${label} (${count})`).join("; ") + "." : "Подтверждённые источники ответа не указаны.", "route-used-sources"));
    const trace = data.trace || [];
    const successful = [...new Set(trace.filter(step => step.status === "ok").map(step => step.tool))];
    box.append(node("p", successful.length ? "Успешно выполнено: " + successful.map(name => tools[name] || name).join("; ") + "." : trace.length ? "Успешных вызовов инструментов нет." : "Вызовов инструментов нет.", "route-executed"));
    const steps = data.diagnostics?.steps || [];
    const rejected = steps.filter(step => step.status === "rejected").length;
    const failed = trace.filter(step => step.status === "error").length;
    if (rejected || failed) box.append(node("p", `Отклонённых попыток: ${rejected}. Вызовов с ошибкой: ${failed}.`, "small route-attempts"));
    // Match by the observed step number and tool, never by a repeated tool name.
    const first = steps.find(step => step.status === "ok");
    const decision = first && (data.decisions || [])[first.step - 1];
    if (first && decision && decision.action === first.tool && decision.purpose) {
      const label = decision.origin === "plan_policy" ? "Цель шага контроллера: " : "Цель модели: ";
      box.append(node("p", label + decision.purpose, "small route-purpose"));
    }
    const plan = data.diagnostics?.plan;
    if (plan?.report_only) {
      box.append(node("p", "Область поиска: только база знаний. Интернет запрещён для этого запроса.", "small route-policy"));
    } else if ((data.decisions || []).some(d => d.quote_lookup)) {
      const context = plan?.wants_context ? " Для аналитической части модель выбирает источники отдельно." : "";
      box.append(node("p", "Для запрошенной спотовой цены агент обращается к дневному ряду EIA/FRED." + context, "small route-policy"));
    } else if (data.agent_mode === "loop" && plan && plan.intent !== "outside") {
      box.append(node("p", "Источник выбирает модель по задаче: содержание отчётов — в базе знаний, внешние и свежие сведения — в интернете. Обязательной очереди нет; при нехватке данных можно сменить источник.", "small route-policy"));
    }
    const lastCheck = data.diagnostics?.answer_checks?.at(-1);
    if (data.diagnostics?.semantic_review_enabled === false && (plan?.wants_context || data.diagnostics?.answer_checks?.length)) {
      box.append(node("p", "Проверка смысла и полноты ответа отключена в настройках.", "small route-caveat"));
    }
    if (data.status === "ok" && lastCheck?.complete === true) {
      box.append(node("p", "Ответ завершён: модель сочла найденные сведения достаточными. Это оценка модели, а не гарантия полноты.", "small route-completion"));
    }
    for (const source of data.sources || []) {
      if (source.kind !== "web") continue;
      const caveats = [];
      if (!source.date) caveats.push("дата публикации неизвестна");
      if (source.content_type === "snippet") caveats.push("цитата из поисковой выдачи, это не полный текст страницы");
      else if (source.content_scope === "query_chunks") caveats.push("использованы выбранные фрагменты страницы");
      if (source.content_truncated) caveats.push("переданный текст ограничен по длине");
      if (caveats.length) box.append(node("p", `[${source.id}] ${caveats.join("; ")}.`, "small route-caveat"));
    }
    return box;
  }
  function render(data, elapsed) {
    const box = node("aside", undefined, "trace-panel");
    box.setAttribute("aria-label", "Ход запроса");
    const heading = node("div", undefined, "trace-heading");
    heading.append(node("h2", "Ход запроса"));
    box.append(heading);
    const directHelp = data.intent === "capabilities" && data.status === "ok" &&
      !data.trace?.length && !data.sources?.length && !data.decisions?.length;
    if (directHelp) {
      box.append(node("p", "Агент показал встроенную справку о возможностях стенда.", "small"));
    } else {
      box.append(node("p", "Фактические действия, пояснения модели и шаги по правилам агента.", "small"), routeView(data));
    }
    const metrics = node("div", undefined, "trace-metrics");
    for (const [value, label] of [[duration(elapsed), "запрос в браузере"], [(data.trace || []).length, "вызовов инструментов"], [(data.decisions || []).filter(d => d.origin !== "plan_policy").length, "выборов LLM"]]) {
      const metric = node("div"); metric.append(node("strong", String(value)), node("span", label)); metrics.append(metric);
    }
    box.append(metrics);
    const diagnostics = data.diagnostics;
    if (directHelp) {
      if (diagnostics?.plan) box.append(detail("Как понят запрос", planView(diagnostics.plan)));
      return appendTraceExport(box, data);
    }
    const clock = diagnostics?.time_context;
    if (clock?.today) {
      const time = node("div", undefined, "trace-time");
      time.append(node("strong", `Дата запроса: ${clock.today}`), node("p", `Часовой пояс: ${clock.timezone || "не указан"}. Эта дата передана модели.`, "small"));
      time.append(detail("Время запроса", json(clock)));
      box.append(time);
    }
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
    if (diagnostics?.timeout_seconds) stop.append(node("p", `Бюджет обработки: ${diagnostics.timeout_seconds} с. Порог отсутствия прогресса: ${diagnostics.no_progress_limit} попытки.`, "small"));
    if (diagnostics?.answer_reserve_seconds) stop.append(node("p", `Резерв на ответ внутри общего бюджета: ${diagnostics.answer_reserve_seconds} с.`, "small"));
    if (diagnostics?.answer_checks?.length) {
      const handoffs = node("div");
      for (const check of diagnostics.answer_checks) {
        const item = node("div", undefined, "answer-check");
        item.append(node("p", `После шага ${check.after_step}: переданы источники ${(check.source_ids || []).join(", ") || "отсутствуют"}.`));
        item.append(node("p", `Подготовка и проверка: ${duration(check.ms)}. ${check.status === "error" ? "Ошибка: " + check.error : `Утверждений после проверки: ${check.claims}. Полнота кандидата по оценке модели: ${check.complete ? "полный ответ" : "неполный ответ"}.`}`, "small"));
        if (check.feedback) item.append(node("p", check.feedback, "small"));
        if (Number.isInteger(check.validation?.candidate_claims)) {
          const v = check.validation;
          item.append(node("p", `Модель предложила утверждений: ${v.candidate_claims}. Отклонено по ссылкам: ${v.citation_rejected || 0}, по числам: ${v.numeric_rejected || 0}, по смыслу: ${v.semantic_rejected || 0}.`, "small"));
        }
        if (check.validation?.citation_repair_attempted) {
          item.append(node("p", "Модели повторно переданы те же фрагменты для исправления ссылок.", "small"));
        }
        const coverageLabels = { sufficient: "достаточно сведений", partial: "частичное", unverified: "не проверено" };
        if (check.validation?.coverage) item.append(node("p", "Покрытие вопроса: " + (coverageLabels[check.validation.coverage] || check.validation.coverage) + ".", "small"));
        if (check.validation?.missing_aspects?.length) item.append(node("p", "Не хватает: " + check.validation.missing_aspects.join("; "), "small"));
        handoffs.append(item);
      }
      box.append(detail("Как источники переданы в ответ", handoffs));
    }
    if (diagnostics?.answer_feedback) stop.append(detail("Замечание проверки ответа", node("p", diagnostics.answer_feedback)));
    box.append(stop);
    if (diagnostics) box.append(detail(`Найденные фрагменты · ${diagnostics.retrieved_sources?.length || 0}`, sourcesView(diagnostics.retrieved_sources || [])));
    box.append(node("p", "Пояснение модели и статус выполнения не подтверждают правильность вывода. Сверяйте ответ с источниками.", "trace-footnote"));
    return appendTraceExport(box, data);
  }
  function appendTraceExport(box, data) {
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
