"use strict";

window.RequestJourney = (() => {
  const tools = {
    report_search: "Поиск в отчётах", read_report: "Чтение документа", web_search: "Веб-поиск",
    read_web: "Чтение веб-страницы", price_forecast: "Расчёт прогноза", finish: "Переход к ответу",
  };
  const intents = { research: "Анализ источников", forecast: "Прогноз", outside: "Вне специализации", capabilities: "Возможности агента" };
  const statuses = { ok: "Ответ готов", partial: "Частичный ответ", clarification: "Запрошено уточнение", outside: "Вне специализации", unavailable: "Ответ недоступен" };
  const stops = {
    completed: "Обработка завершена", budget_exhausted: "Исчерпан лимит шагов", no_progress: "Повторные попытки не добавили данных",
    llm_unavailable: "Сбой на этапе обращения к модели", time_budget_exhausted: "Истёк лимит времени",
    research_budget_exhausted: "Поиск остановлен, чтобы осталось время на ответ",
    review_disabled: "Смысловая проверка отключена", research_clarification: "По результатам поиска потребовалось уточнение",
    chronology_format_incomplete: "Не удалось завершить сопоставление дат",
  };
  const reasons = {
    complete: "Проверка подтвердила полноту", coverage_incomplete: "Часть вопроса осталась без подтверждения",
    coverage_unverified: "Полноту ответа подтвердить не удалось", incomplete: "Ответ остался неполным",
    semantic_validation_rejected: "Смысловая проверка отклонила утверждения", no_candidate_claims: "Модель не предложила утверждений",
    no_evidence: "Нет фрагментов для подготовки ответа", local_validation_rejected: "Утверждения не прошли проверку ссылок или чисел",
    semantic_review_disabled: "Смысловая проверка отключена", invalid_review_indexes: "Проверяющая модель вернула некорректные номера утверждений",
    no_relevant_event_sources: "Публикации посвящены другим событиям", no_confirmed_event: "Дата состоявшегося события не подтверждена",
    event_chronology_mismatch: "Выводы не согласуются с хронологией", chronology_format_incomplete: "Не удалось разобрать хронологию",
    pending: "Итог проверки не зафиксирован",
  };
  const phases = { evidence: "Подбор фрагментов", answer_generation: "Подготовка ответа моделью", local_validation: "Проверка ссылок и чисел",
    semantic_review: "Смысловая проверка", event_chronology: "Проверка дат", event_relevance: "Проверка события", citation_repair: "Исправление ссылок", finished: "Завершение проверки" };
  const outcomes = { passed: "Пройдена", failed: "Отклонено", not_checked: "Не проводилась", event_context: "Числа из контекста события",
    invalid_review: "Некорректный результат проверки" };
  const list = value => Array.isArray(value) ? value : [];
  const yesNo = value => typeof value === "boolean" ? (value ? "Да" : "Нет") : "";
  const label = (labels, key) => Object.hasOwn(labels, key) ? labels[key] : key;
  const join = value => list(value).join(", ");
  const duration = ms => Number.isFinite(ms) ? (ms < 1000 ? Math.round(ms) + " мс" : (ms / 1000).toLocaleString("ru-RU", {maximumFractionDigits: 2}) + " с") : "";
  function el(tag, text, cls) {
    const node = document.createElement(tag);
    if (cls) node.className = cls;
    if (text !== undefined) node.textContent = text;
    return node;
  }
  function facts(parent, rows) {
    const dl = el("dl", undefined, "journey-facts");
    for (const [name, value] of rows) {
      if (value !== null && value !== undefined && value !== "") dl.append(el("dt", name), el("dd", value));
    }
    if (dl.childElementCount) parent.append(dl);
  }
  function disclosure(parent, title, cls) {
    const details = el("details", undefined, cls);
    details.append(el("summary", title));
    parent.append(details);
    return details;
  }
  function stage(flow, kind, title, badge, tone, attributes = {}) {
    const item = el("li", undefined, "journey-stage");
    item.dataset.kind = kind;
    Object.assign(item.dataset, attributes);
    const details = el("details", undefined, "journey-node");
    const heading = el("summary", undefined, "journey-node-heading");
    const sign = el("span", badge, "journey-badge");
    sign.dataset.tone = tone;
    heading.append(el("span", title, "journey-node-title"), sign);
    const body = el("div", undefined, "journey-body");
    details.append(heading, body);
    item.append(details);
    flow.append(item);
    return body;
  }

  function sourceLink(parent, raw, caption) {
    if (typeof raw !== "string" || !raw.trim()) return;
    try {
      const url = new URL(raw, location.origin);
      if (!["https:", "http:"].includes(url.protocol)) return;
      const link = el("a", caption);
      link.href = url.href;
      link.target = "_blank";
      link.rel = "noopener noreferrer";
      parent.append(link);
    } catch { /* Some traces only carry a source ID. */ }
  }
  function sourceCard(parent, source, citation) {
    const id = citation?.evidence_id || source.id;
    const title = citation?.title || source.title;
    const box = disclosure(parent, [id, title].filter(Boolean).join(" · "), "journey-source");
    box.dataset.sourceId = source.id;
    const record = citation ? {...source, ...citation} : source;
    const page = record.page ?? record.locator?.page;
    const where = window.Workbench?.locatorLabel?.(record.locator, page) || record.locator?.label || (page ? "Страница PDF " + page : "");
    facts(box, [
      ["Место в документе", where], ["Дата источника", record.date || (record.kind === "web" ? "Не указана" : "")],
      ["Установление даты", record.date_basis === "provider_estimate" ? "По оценке поискового провайдера" : ""],
      ["Материал", record.content_type === "snippet" ? "Фрагмент поисковой выдачи" : record.content_scope === "query_chunks" ? "Выбранные фрагменты страницы" : ""],
      ["Поставщик", record.provider],
      ["В итоговом ответе", yesNo(record.used_in_answer)],
    ]);
    const text = citation ? citation.quote : source.text;
    if (text) {
      box.append(el("p", citation ? "Фрагмент, указанный в утверждении" : "Содержимое источника", "journey-caption"), el("blockquote", text));
      if (!citation && (source.text_truncated || source.content_truncated)) box.append(el("p", "Показана часть текста."));
    }
    sourceLink(box, record.local_url, "Открыть документ");
    if (record.url !== record.local_url) sourceLink(box, record.url, "Открыть источник");
  }
  function sources(parent, ids, catalog) {
    for (const id of new Set(list(ids))) {
      const record = catalog.get(id);
      if (record) sourceCard(parent, record);
      else parent.append(el("p", id + " · описание источника не передано"));
    }
  }
  function planStage(flow, data, diag) {
    const plan = diag.plan;
    if (!plan) return;
    const body = stage(flow, "plan", "План и границы поиска", label(intents, plan.intent) || "План получен", "neutral");
    facts(body, [
      ["Запрос", data.question], ["Модель", data.mode], ["Дата запроса", diag.time_context?.today], ["Часовой пояс", diag.time_context?.timezone],
      ["Источники по плану", typeof plan.report_only === "boolean" ? (plan.report_only ? "Только загруженные отчёты" : "Разрешены отчёты и интернет") : ""],
      ["Нужна проверка на текущую дату", yesNo(plan.fresh)], ["Требуется контекст из источников", yesNo(plan.wants_context)],
      ["Последнее состоявшееся событие", plan.latest_completed_event ? "Да" : ""], ["Фраза для поиска", plan.rag_query],
      ["Котировки", join(plan.quotes)], ["Отдельный вопрос к источникам", plan.context_request],
      ["Сорт для расчёта", plan.benchmark], ["Горизонт, мес.", plan.months], ["Дисконт, USD/барр.", plan.discount],
      ["Сокращение, млн барр./сут.", plan.cut], ["Чувствительность", plan.sensitivity], ["Уточнение", plan.clarify],
      ["Этап в коде", "src/agent.py · build_graph → plan"],
    ]);
  }

  const argLabels = { query: "Поисковая фраза", search: "Поиск публикаций", quotes: "Котировки", fresh: "Актуальные сведения",
    quote_fallback: "Поиск альтернативной котировки", report: "Документ", page: "Страница", block_id: "Блок", version_id: "Версия",
    source_id: "Источник", publisher_domain: "Сайт издателя", benchmark: "Сорт", months: "Горизонт, мес.", discount: "Дисконт",
    cut: "Сокращение добычи", sensitivity: "Чувствительность", start_date: "Начало периода", end_date: "Конец периода", url: "Адрес страницы", topic: "Тема", days: "Период, дней" };
  function argumentsView(body, args) {
    if (!args || !Object.keys(args).length) return;
    const rows = Object.entries(args).map(([key, value]) => [label(argLabels, key),
      typeof value === "boolean" ? yesNo(value) : Array.isArray(value) ? join(value) : value && typeof value === "object" ? JSON.stringify(value) : value]);
    facts(disclosure(body, "Параметры вызова", "journey-section"), rows);
  }
  function actionName(action, decision, execution) {
    if (action === "web_search" && (decision?.quote_lookup || (execution?.arguments?.search === false && execution.arguments.quotes?.length))) return "Получение котировки";
    return label(tools, action) || "Действие";
  }
  function publicationPeriod(args) {
    if (args?.days != null) return `Последние ${args.days} дн.`;
    return [args?.start_date && "с " + args.start_date, args?.end_date && "по " + args.end_date].filter(Boolean).join(" ") || "Без заданного периода";
  }
  function actionStage(flow, number, decision, run, catalog) {
    const action = run?.tool || decision?.action;
    const state = run?.status;
    const badge = { ok: "Выполнено", rejected: "Отклонено", error: "Ошибка" };
    const title = (number ? "Шаг " + number + " · " : "") + actionName(action, decision, run);
    const body = stage(flow, "action", title, label(badge, state) || (action === "finish" ? "Решение" : "Исполнение не зафиксировано"),
      state === "rejected" ? "warning" : state === "error" ? "error" : state === "ok" ? "success" : "neutral", number ? {step: number} : {});
    const repair = run?.argument_repair;
    const periodChanged = repair?.mode === "event_search" && publicationPeriod(repair.before) !== publicationPeriod(repair.after);
    facts(body, [
      ["Выбор действия", label({plan_policy: "Контроллер по плану модели", llm: "Языковая модель", event_recovery: "Контроллер после повторного предложения модели"}, decision?.origin)],
      ["Цель", decision?.purpose || run?.purpose],
      ["Предложение модели", decision?.model_action && actionName(decision.model_action)],
      ["Причина перехода", decision?.controller_note],
      ["Оценка модели перед действием", label({need_evidence: "Нужны источники", need_detail: "Нужны подробности", enough: "Сведений достаточно", unavailable: "Сведения недоступны"}, decision?.assessment)],
      ["Модель запрашивает", join(decision?.missing_facts)], ["Неизвестные ссылки", join(decision?.unknown_source_ids)],
      ["Время исполнения", duration(run?.ms)], ["Получено фрагментов", run?.result_count],
      ["Появились новые данные", yesNo(run?.new_evidence)], ["Новые источники", join(run?.new_source_ids)],
      ["Попыток подряд без новых данных", run?.no_progress_steps], ["Повтор шага", run?.duplicate_of], ["Ранее найдено", join(run?.reused_source_ids)],
      ["Пересобранный запрос", run?.argument_repair?.after?.query], ["Причина отклонения или ошибки", localError(run?.error)],
      ["Период публикаций, предложенный моделью", periodChanged ? publicationPeriod(repair.before) : ""],
      ["Период в поиске", periodChanged ? publicationPeriod(repair.after) : ""],
      ["Этап в коде", run && decision ? "src/agent.py · decide → act" : run ? "src/agent.py · act" : "src/agent.py · decide"],
      ["Расчёт", run?.result?.benchmark], ["История по", run?.result?.as_of], ["Последняя цена в ряду", run?.result?.last],
    ]);
    if (!decision && run && !number) body.append(el("p", "Номер шага не передан."));
    if (decision && !run && action !== "finish") body.append(el("p", "Зафиксирован выбор модели или контроллера. Результат исполнения отсутствует."));
    for (const item of list(decision?.evidence_checks)) {
      body.append(el("p", [item.fact, join(item.source_ids)].filter(Boolean).join(" · ")));
    }
    if (run) {
      argumentsView(body, run.arguments);
      for (const notice of list(run.notices)) body.append(el("p", notice));
      sources(body, run.source_ids, catalog);
    }
  }
  function localError(message) {
    if (!message) return "";
    if (typeof message !== "string") return "";
    if (message.startsWith("Duplicate action rejected")) return "Этот вызов повторяет предыдущий с теми же параметрами.";
    if (message.startsWith("Action rejected by policy")) return "Правила доступа запрещают выбранное действие.";
    if (message.startsWith("Use a registered report ID")) return "Документа нет среди найденных источников.";
    return message;
  }

  function checkStage(flow, check, linked) {
    const v = check.validation || {};
    const hasDraftCount = Number.isInteger(v.candidate_claims);
    let badge = "Результат не указан", tone = "neutral";
    if (check.status === "error") { badge = "Ошибка"; tone = "error"; }
    else if (v.reason === "semantic_review_disabled") { badge = "Проверка отключена"; tone = "warning"; }
    else if (check.complete === true) { badge = "Ответ полный"; tone = "success"; }
    else if (check.complete === false) { badge = check.claims === 0 ? "Нет принятых утверждений" : "Ответ неполный"; tone = "warning"; }
    const title = linked ? "Проверка ответа после шага " + check.after_step : "Проверка ответа без привязки к действию";
    const body = stage(flow, "check", title, badge, tone, Number.isInteger(check.after_step) ? {afterStep: check.after_step} : {});
    facts(body, [
      ["Указанный номер шага", !linked ? check.after_step : ""], ["Этап проверки", label(phases, v.stage)],
      ["Результат проверки", label(reasons, v.reason)], ["Ошибка", check.error], ["Время проверки", duration(check.ms)],
      ["Переданы для подготовки ответа", join(check.supplied_source_ids)], ["Источники проверки", join(check.source_ids)],
      ["Фрагменты проверки", join(check.evidence_ids)], ["Предложено утверждений", v.candidate_claims],
      ["Отклонено по ссылкам", hasDraftCount ? v.citation_rejected : null], ["Отклонено по числам", hasDraftCount ? v.numeric_rejected : null],
      ["Отклонено по смыслу", hasDraftCount && v.review_executed !== false ? v.semantic_rejected : null], ["Принято утверждений", check.claims ?? (hasDraftCount ? v.accepted_claims : null)],
      ["Смысловая проверка выполнена", yesNo(v.review_executed)],
      ["Полнота по оценке проверки", label({sufficient: "Достаточно сведений", partial: "Частично", unverified: "Не проверена"}, v.coverage)],
      ["Неподтверждённые части вопроса", join(v.missing_aspects)], ["Замечание проверки", check.feedback],
      ["Повторная проверка ссылок", v.citation_repair_attempted ? label({pending: "Запрошена", empty: "Модель не предложила утверждений", ids_valid: "Ссылки исправлены", invalid_ids: "Ссылки остались неверными", partially_valid_ids: "Часть ссылок исправлена"}, v.citation_repair_outcome) || "Запрошена" : ""],
      ["Исключены публикации о других событиях", join(v.event_relevance?.excluded_source_ids)],
      ["Этап в коде", "src/agent.py · answer → src/evidence.py · generate_answer"],
    ]);
    if (v.event_chronology) {
      const c = v.event_chronology;
      const box = disclosure(body, "Проверка дат события", "journey-section");
      facts(box, [
        ["Дата сравнения", c.as_of], ["Последнее подтверждённое событие", c.latest_held_date || "Дата не установлена"],
        ["Подтверждающие фрагменты", join(c.latest_evidence_ids)], ["Фрагменты с невыясненной датой", join(c.unresolved_evidence_ids)],
        ["Прошедшие даты анонсов", join(c.overdue_scheduled_dates)], ["Пропуски в контексте", join(c.missing_requested_context)],
        ["Фрагменты с ошибкой разбора", join(c.extraction?.unparsed_evidence_ids)],
      ]);
    }
    for (const row of list(check.claim_results)) {
      const accepted = row.accepted === true ? "принято" : row.accepted === false ? "не принято" : "итог не указан";
      const claim = disclosure(body, "Утверждение " + row.index + " · " + accepted, "journey-claim");
      facts(claim, [
        ["Проверка ссылок", label(outcomes, row.citation)], ["Проверка чисел", label(outcomes, row.numeric)],
        ["Смысловая проверка", label(outcomes, row.semantic)], ["Известные блоки источников", join(row.evidence_ids)],
        ["Неподтверждённых чисел", row.missing_number_count],
      ]);
    }
  }

  function finishStage(flow, data, diag, catalog) {
    const status = label(statuses, data.status) || "Статус не передан";
    const body = stage(flow, "finish", "Итог запроса", status, data.status === "unavailable" ? "error" : data.status === "partial" ? "warning" : data.status === "ok" ? "success" : "neutral");
    const stop = label(stops, data.stop_reason);
    const claims = list(data.claims);
    facts(body, [["Остановка", stop], ["Сообщение об ошибке", diag.failure || data.failure], ["ID запроса", data.trace_id],
      ["Источники итогового ответа", list(data.sources).map(s => s.id).join(", ")], ["Этап в коде", data.agent_mode === "loop" ? "src/agent.py · finish" : ""]]);
    if (data.status === "clarification" || data.status === "outside" || data.status === "unavailable" || data.intent === "capabilities") {
      if (data.answer) body.append(el("p", data.answer));
    }
    if (data.forecast) {
      const f = data.forecast;
      const end = list(f.rows).at(-1);
      const calc = disclosure(body, "Результат расчётного модуля", "journey-section");
      facts(calc, [["Сорт", f.benchmark], ["Данные по", f.as_of], ["Единица", f.unit], ["Последняя цена", f.last], ["Последний месяц прогноза", end?.month],
        ["Оценка цены", end?.point], ["Нижняя граница", end?.lower], ["Верхняя граница", end?.upper],
        ["Уровень интервала", Number.isFinite(f.level) ? (f.level * 100) + "%" : ""], ["Историческая MAE", end?.mae], ["MAE прогноза по последней цене", end?.naive_mae]]);
      sourceLink(calc, f.source_url, "Открыть ряд данных");
    }
    for (const claim of claims) {
      const box = disclosure(body, claim.text, "journey-claim");
      if (list(claim.citations).length) {
        for (const c of claim.citations) sourceCard(box, catalog.get(c.source) || {id: c.source}, c);
      } else {
        const source = catalog.get(claim.source) || {id: claim.source};
        if (claim.quote) sourceCard(box, source, {quote: claim.quote});
        else sources(box, [claim.source].filter(Boolean), catalog);
      }
    }
    for (const s of list(data.sources)) {
      if (s.price !== null && s.price !== undefined) sourceCard(body, s);
    }
  }

  function render(data = {}) {
    const root = el("details", undefined, "request-journey");
    const diag = data.diagnostics || {}, runs = list(diag.steps), decisions = list(data.decisions);
    const summary = el("summary", undefined, "journey-summary");
    const outcome = label(statuses, data.status) || "Итог не указан";
    const why = data.stop_reason && data.stop_reason !== "completed" ? label(stops, data.stop_reason) : "";
    summary.append(el("span", "Как прошёл запрос", "journey-title"), el("span", [outcome, why].filter(Boolean).join(" · "), "journey-outcome"));
    const flow = el("ol", undefined, "journey-timeline");
    flow.setAttribute("role", "list");
    root.append(summary, flow);
    const catalog = new Map();
    for (const s of [...list(data.sources), ...list(diag.retrieved_sources)]) catalog.set(s.id, {...catalog.get(s.id), ...s});
    planStage(flow, data, diag);
    const indexed = new Map();
    decisions.forEach((decision, i) => indexed.set(i + 1, {decision, runs: []}));
    const unindexed = [];
    for (const run of runs) {
      if (!Number.isInteger(run.step) || run.step < 1) { unindexed.push(run); continue; }
      if (!indexed.has(run.step)) indexed.set(run.step, {runs: []});
      indexed.get(run.step).runs.push(run);
    }
    const checks = list(diag.answer_checks);
    for (const n of [...indexed.keys()].sort((a, b) => a - b)) {
      const {decision, runs: sameStep} = indexed.get(n);
      if (decision && !sameStep.some(run => run.tool === decision.action)) actionStage(flow, n, decision, null, catalog);
      for (const run of sameStep) actionStage(flow, n, decision?.action === run.tool ? decision : null, run, catalog);
      for (const check of checks) if (check.after_step === n) checkStage(flow, check, true);
    }
    for (const run of unindexed) actionStage(flow, null, null, run, catalog);
    if (!runs.length && list(data.trace).length) {
      const body = stage(flow, "legacy", "Журнал вызовов инструментов", "Без привязки к решению", "neutral");
      const entries = el("ol");
      for (const event of data.trace) entries.append(el("li", [actionName(event.tool), label({ok: "Выполнено", error: "Ошибка", rejected: "Отклонено"}, event.status), duration(event.ms)].filter(Boolean).join(" · ")));
      body.append(entries);
    } else if (!diag.plan && !indexed.size && !runs.length && !checks.length) {
      const body = stage(flow, "legacy", "Данные о маршруте", "Нет подробностей", "neutral");
      body.append(el("p", "В ответе сервера отсутствуют события выполнения."));
    }
    for (const check of checks) if (!indexed.has(check.after_step)) checkStage(flow, check, false);
    finishStage(flow, data, diag, catalog);
    return root;
  }
  return {render};
})();
