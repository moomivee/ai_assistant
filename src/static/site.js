"use strict";
const $ = (id) => document.getElementById(id);
let previous = null;
let dialogueToken = null;
let dialogueScope = null;
const defaultQuestionHint = $("question").placeholder;
const capabilities = [
  {
    kind: "report",
    icon: "M6 3h8l4 4v14H6z M14 3v5h4 M9 12h6 M9 16h6",
    label: "Найти в отчётах",
    example: "Чем отличаются Urals и ESPO по плотности и содержанию серы? Используй только загруженные отчёты.",
  },
  {
    kind: "web",
    icon: "M21 12a9 9 0 1 1-18 0 9 9 0 0 1 18 0z M3 12h18 M12 3c4 4 4 14 0 18-4-4-4-14 0-18z",
    label: "Проверить свежие данные",
    example: "Какая последняя опубликованная спотовая цена Brent? Укажи дату наблюдения и источник.",
  },
  {
    kind: "compare",
    icon: "M3 8h17l-4-4 M21 16H4l4 4",
    label: "Сопоставить источники",
    example: "Что в отчёте EIA сказано об экспорте российской нефти в Индию? Сравни со свежими публикациями.",
  },
  {
    kind: "forecast",
    icon: "M3 18l6-7 4 3 7-9 M14 5h6v6",
    label: "Спрогнозировать цену",
    example: "Спрогнозируй среднемесячную цену Brent на 3 месяца.",
  },
  {
    kind: "scenario",
    icon: "M4 6h7m4 0h5 M4 12h2m4 0h10 M4 18h9m4 0h3 M11 4v4h4V4z M6 10v4h4v-4z M13 16v4h4v-4z",
    label: "Рассчитать сценарий",
    example: "Спрогнозируй Urals на 3 месяца при дисконте 12 долларов за баррель к Brent.",
  },
  {
    kind: "upload",
    icon: "M12 15V3m-5 5 5-5 5 5 M4 15v5h16v-5",
    label: "Разобрать свой документ",
  },
];
function element(tag, text, cls) {
  const node = document.createElement(tag);
  if (text !== undefined) node.textContent = text;
  if (cls) node.className = cls;
  return node;
}
function link(label, url) {
  const a = element("a", label);
  const safe = typeof url === "string" && url.trim() ? TraceView.safeURL(url) : null;
  if (safe) a.href = safe;
  a.target = "_blank";
  a.rel = "noreferrer";
  return a;
}
function showTab(name, focus = false) {
  for (const tab of document.querySelectorAll('.workspace-tabs [role="tab"]')) {
    const active = tab.id === "tab-" + name;
    tab.setAttribute("aria-selected", String(active));
    tab.tabIndex = active ? 0 : -1;
    $(tab.getAttribute("aria-controls")).hidden = !active;
    if (active && focus) tab.focus();
  }
}
for (const tab of document.querySelectorAll('.workspace-tabs [role="tab"]')) {
  tab.addEventListener("click", () => showTab(tab.id.slice(4)));
  tab.addEventListener("keydown", event => {
    const tabs = [...document.querySelectorAll('.workspace-tabs [role="tab"]')];
    const index = tabs.indexOf(tab);
    const next = event.key === "ArrowRight" ? (index + 1) % tabs.length : event.key === "ArrowLeft" ? (index + tabs.length - 1) % tabs.length : event.key === "Home" ? 0 : event.key === "End" ? tabs.length - 1 : null;
    if (next !== null) { event.preventDefault(); showTab(tabs[next].id.slice(4), true); }
  });
}
function resetClarification() {
  previous = null;
  dialogueToken = null;
  dialogueScope = null;
  $("newquestion").hidden = true;
  $("question").placeholder = defaultQuestionHint;
}
function requestScope() {
  return {
    document_ids: (window.KnowledgeUI?.selectedIds() || []).sort(),
    report_only: $("report-only").checked,
    note_id: window.ResearchNotes?.selectedId() || null,
  };
}
function checkDialogueScope() {
  if (dialogueToken && dialogueScope !== JSON.stringify(requestScope())) {
    resetClarification();
    $("result").replaceChildren(element("p", "Источники для запроса изменились. Для нового прогноза укажите его параметры заново."));
  }
}
function selectQuery(question) {
  if ($("question").disabled) return;
  resetClarification();
  $("question").value = question;
  counter();
  showTab("test");
  $("result").hidden = true;
  $("question").focus();
}
function updateScopeHint() {
  const onlyReports = $("report-only").checked;
  const count = window.KnowledgeUI?.selectedIds().length || 0;
  const active = $("active-scope");
  active.hidden = !onlyReports && !count;
  active.textContent = [onlyReports ? "Только база знаний" : "", count ? "Выбрано документов: " + count : ""].filter(Boolean).join(" · ");
  $("scope-hint").textContent = onlyReports
    ? "Для этого вопроса веб-поиск выключен."
    : "Агент может использовать документы и интернет.";
}
updateScopeHint();
function locatorLabel(locator = {}, page = null) {
  if (locator?.label) return locator.label;
  const where = locator || {}, parts = [];
  if (page || where.page) parts.push("Страница PDF " + (page || where.page));
  if (where.sheet) parts.push("Лист «" + where.sheet + "»");
  if (where.cell_range || where.cells) parts.push("Ячейки " + (where.cell_range || where.cells));
  if (where.section) parts.push("Раздел " + where.section);
  if (where.part) parts.push(({ header: "Верхний колонтитул", footer: "Нижний колонтитул" })[where.part] || "Изображение " + String(where.part).split("/").pop());
  if (where.paragraph) parts.push("Абзац " + where.paragraph);
  if (where.table) parts.push("Таблица " + where.table);
  if (where.row_start && !where.cell_range) parts.push("Строки " + where.row_start + (where.row_end && where.row_end !== where.row_start ? "–" + where.row_end : ""));
  if (where.region === "whole_image" && !where.part) parts.push("Изображение целиком");
  return parts.join(" · ");
}
updateScopeHint();
$("report-only").addEventListener("change", () => { updateScopeHint(); checkDialogueScope(); });
$("remember-forecast").addEventListener("change", () => {
  if (dialogueToken) {
    resetClarification();
    $("result").replaceChildren(element("p", "Запоминание уточнений выключено. Задайте вопрос о прогнозе целиком."));
  }
});
$("choose-documents").addEventListener("click", () => {
  showTab("knowledge"); $("document-filter").focus();
});
window.Workbench = { showTab, selectQuery, updateScopeHint, checkDialogueScope, locatorLabel, partialNotice };
function counter() {
  $("counter").textContent = $("question").value.length + " / 2000";
}
$("question").addEventListener("input", counter);
$("question").addEventListener("keydown", (event) => {
  if (event.key === "Enter" && (event.ctrlKey || event.metaKey)) {
    event.preventDefault();
    $("ask").requestSubmit();
  }
});

for (const task of capabilities) {
  const button = element("button", undefined, "capability-query");
  button.type = "button";
  const icon = element("span", undefined, "cap-icon");
  icon.dataset.kind = task.kind;
  icon.setAttribute("aria-hidden", "true");
  const drawing = document.createElementNS("http://www.w3.org/2000/svg", "svg");
  drawing.setAttribute("viewBox", "0 0 24 24");
  drawing.setAttribute("width", "18");
  drawing.setAttribute("height", "18");
  drawing.setAttribute("fill", "none");
  drawing.setAttribute("stroke", "currentColor");
  drawing.setAttribute("stroke-width", "1.6");
  drawing.setAttribute("stroke-linecap", "round");
  drawing.setAttribute("stroke-linejoin", "round");
  const path = document.createElementNS(drawing.namespaceURI, "path");
  path.setAttribute("d", task.icon);
  drawing.append(path);
  icon.append(drawing);
  const arrow = element("span", "↗", "cap-arrow");
  arrow.setAttribute("aria-hidden", "true");
  button.append(icon, element("span", task.label, "cap-label"), arrow);
  if (task.example) {
    button.title = task.example;
    button.addEventListener("click", () => selectQuery(task.example));
  } else {
    button.title = "Открыть загрузку в базе знаний";
    button.addEventListener("click", () => {
      showTab("knowledge");
      $("document-file").focus();
    });
  }
  $("capabilities").append(button);
}

fetch("/api/rag/status").then(response => {
  if (!response.ok) throw new Error("rag");
  return response.json();
}).then(data => {
  const lines = [data.model ? "Эмбеддер: " + data.model : null, data.model_repo ? "Модель: " + data.model_repo : null, data.search_mode || data.mode ? "Режим: " + (data.search_mode || data.mode) : null, data.top_k ? "Фрагментов за поиск: " + data.top_k : null, data.status ? "Индекс: " + (data.status === "ready" ? "готов" : "ожидает построения при запросе") : null];
  $("rag-settings").textContent = lines.join(" · ") || "Настройки эмбеддера сервером не переданы.";
}).catch(() => { $("rag-settings").textContent = "Настройки поиска сейчас недоступны. Метрики здесь не измерялись."; });
function source(row) {
  const detail = element("details", undefined, "source"),
    labels = { model: "Расчёт", report: "Отчёт", web: "Веб" };
  detail.append(element("summary", "[" + row.id + "] " + row.title));
  detail.append(
    element(
      "small",
      [
        labels[row.kind],
        row.date || "Дата публикации не указана",
        locatorLabel(row.locator, row.page),
      ]
        .filter(Boolean)
        .join(" · "),
    ),
  );
  if (row.kind === "web") detail.append(element("small", ["full_text", "page", "extracted"].includes(row.content_type) ? "Извлечённый текст открытой страницы" : "Фрагмент веб-источника; полнота страницы не подтверждена"));
  detail.append(element("p", row.text));
  const links = element("div", undefined, "links");
  if (row.local_url) links.append(link(row.page ? "Открыть страницу документа ↗" : "Открыть документ ↗", row.local_url));
  if (row.url) links.append(link("Первоисточник ↗", row.url));
  detail.append(links);
  return detail;
}
function citationData(data, paragraph, sourceId) {
  const clean = text => String(text || "").replace(/\s+/g, " ").trim();
  const source = (data.sources || []).find(row => row.id === sourceId);
  const preliminary = (data.preliminary_evidence || []).filter(item =>
    item.source_id === sourceId && clean(item.quote) && clean(paragraph).includes(clean(item.quote)));
  if (preliminary.length) return { source, rows: preliminary, preliminary: true };
  const matching = (data.claims || []).filter(claim => {
    const text = clean(claim.text);
    return text && clean(paragraph).includes(text);
  });
  const rows = [];
  for (const claim of matching) {
    for (const citation of claim.citations || []) {
      if (citation.source === sourceId) rows.push(citation);
    }
    if (!claim.citations?.length && claim.source === sourceId) {
      const source = (data.sources || []).find(row => row.id === sourceId);
      const quotes = (claim.evidence_ids || []).map(id => ({ evidence_id: id, quote: source?.evidence?.[id] })).filter(row => row.quote);
      rows.push(...(quotes.length ? quotes : claim.quote ? [{ quote: claim.quote }] : []));
    }
  }
  return { source, rows };
}
let citationNumber = 0;
function appendAnswerParagraph(container, text, data) {
  const paragraph = element("p", undefined, "answer-text");
  const excerpts = [];
  const expression = /\[([RWM]\d+)\]/g;
  let start = 0;
  for (const match of text.matchAll(expression)) {
    paragraph.append(document.createTextNode(text.slice(start, match.index)));
    const id = match[1], { source, rows, preliminary } = citationData(data, text, id);
    if (!source && !rows.length) {
      paragraph.append(document.createTextNode(match[0]));
      start = match.index + match[0].length;
      continue;
    }
    const button = element("button", match[0], "citation-button");
    const box = element("aside", undefined, "citation-excerpt");
    button.type = "button";
    button.setAttribute("aria-label", (preliminary ? "Показать цитату из источника " : "Показать подтверждение ") + id);
    box.id = "citation-" + (++citationNumber);
    button.setAttribute("aria-controls", box.id);
    button.setAttribute("aria-expanded", "false");
    box.hidden = true;
    box.append(element("strong", source?.title || id));
    const caption = preliminary ? "Цитата из источника · предварительные сведения" : rows.length ? "Фрагмент, подтверждающий утверждение" : source?.kind === "model" ? "Описание расчёта" : "Найденный фрагмент источника; точная привязка к утверждению отсутствует";
    box.append(element("span", caption, "citation-kind"));
    for (const citation of rows.length ? rows : [source]) {
      const record = { ...source, ...citation };
      const where = locatorLabel(record.locator, record.page);
      if (where) box.append(element("small", where));
      box.append(element("blockquote", citation.quote || citation.text || "Текст подтверждения не передан."));
      const links = element("div", undefined, "links");
      if (record.local_url) links.append(link(record.page ? "Открыть страницу " + record.page + " ↗" : "Открыть оригинал ↗", record.local_url));
      if (record.url) links.append(link("Первоисточник ↗", record.url));
      box.append(links);
    }
    button.addEventListener("click", () => {
      box.hidden = !box.hidden;
      button.setAttribute("aria-expanded", String(!box.hidden));
    });
    paragraph.append(button); excerpts.push(box);
    start = match.index + match[0].length;
  }
  paragraph.append(document.createTextNode(text.slice(start)));
  container.append(paragraph, ...excerpts);
}
function forecastTable(calc) {
  const box = element("div", undefined, "tablewrap"),
    table = element("table"),
    head = element("thead"),
    top = element("tr");
  for (const label of ["Месяц", "ARIMA", "Сглаживание", "Прогноз", "90%-й интервал"])
    top.append(element("th", label));
  head.append(top);
  table.append(head);
  const body = element("tbody");
  for (const r of calc.rows) {
    const tr = element("tr");
    for (const value of [
      r.month,
      r.arima.toFixed(2),
      r.ses.toFixed(2),
      r.point.toFixed(2),
      r.lower.toFixed(2) + " – " + r.upper.toFixed(2),
    ])
      tr.append(element("td", value));
    body.append(tr);
  }
  table.append(body);
  box.append(table);
  return box;
}
function partialNotice(data) {
  if (data.status !== "partial") return "";
  if (data.preliminary_evidence?.length) return "Предварительные сведения. Агент может ошибаться в выводе. Цитаты можно сверить с оригиналами в источниках.";
  if (data.claims?.length || data.sources?.length || data.forecast) return "Частичный ответ. Найденных данных хватило только на часть вопроса.";
  return "Недостаточно сведений для ответа по существу.";
}
function render(q, data, elapsed) {
  const out = $("result");
  const heading = element("div", undefined, "result-heading");
  const help = data.intent === "capabilities" && data.status === "ok";
  const title = help ? "Возможности агента" : "Ответ аналитика";
  heading.append(element("h2", title));
  const notice = partialNotice(data);
  if (!notice) heading.append(element("span", TraceView.statusLabel(data.status), "response-status " + data.status));
  const answer = element("section", undefined, "answer-panel");
  answer.setAttribute("aria-label", title);
  answer.append(heading, element("div", q, "answer-q"));
  if (notice) {
    const banner = element("p", notice, "answer-caution");
    banner.setAttribute("role", "note");
    answer.append(banner);
  }
  for (const p of (data.answer || "").split("\n\n")) appendAnswerParagraph(answer, p, data);
  if (data.forecast) {
    const detail = element("details", undefined, "forecast-details");
    detail.append(element("summary", "Посмотреть расчёт по месяцам"), forecastTable(data.forecast));
    answer.append(detail);
  }
  if (data.sources?.length) {
    answer.append(element("h3", "Источники ответа", "sectiontitle"));
    for (const row of data.sources) answer.append(source(row));
  }
  if (window.ResearchNotes && !help && !data.clarification && ["ok", "partial"].includes(data.status)) {
    const save = element("button", "Сохранить в заметки", "secondary-button save-note");
    save.type = "button";
    const feedback = element("p", undefined, "small"); feedback.setAttribute("role", "status");
    save.addEventListener("click", () => window.ResearchNotes.save(q, data, save, feedback));
    answer.append(save, feedback);
  }
  const grid = element("div", undefined, "result-grid");
  grid.append(answer, RequestJourney.render(data), TraceView.render(data, elapsed));
  out.replaceChildren(grid);
}
function requestError(status, detail) {
  if (status === 409 && dialogueToken) {
    if (detail?.code === "dialogue_busy") return new Error("Предыдущая реплика ещё обрабатывается. Повторите уточнение позже.");
    resetClarification();
    return new Error("Это уточнение уже завершено или устарело. Задайте вопрос о прогнозе целиком.");
  }
  return new Error("Сервис вернул ошибку " + status + ".");
}

async function askWithProgress(payload, onQuoteFallback) {
  const response = await fetch("/api/ask/stream", {
    method: "POST",
    headers: { "Content-Type": "application/json", Accept: "application/x-ndjson" },
    body: JSON.stringify(payload),
  });
  if (!response.ok) {
    const error = await response.json().catch(() => ({}));
    throw requestError(response.status, error.detail);
  }
  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  try {
    while (true) {
      const { value, done } = await reader.read();
      buffer += decoder.decode(value, { stream: !done });
      const lines = buffer.split("\n");
      buffer = done ? "" : lines.pop();
      for (const line of lines) {
        if (!line.trim()) continue;
        const event = JSON.parse(line);
        if (event.type === "result") return event.result;
        if (event.type === "error") throw requestError(event.status, event.detail);
        if (event.type === "progress" && event.stage === "quote_fallback") onQuoteFallback();
      }
      if (done) throw new Error("Соединение закрылось до получения ответа. Попробуйте ещё раз.");
    }
  } finally {
    await reader.cancel().catch(() => {});
    reader.releaseLock();
  }
}

$("ask").addEventListener("submit", async (e) => {
  e.preventDefault();
  const q = $("question").value.trim(),
    button = $("ask").querySelector(".submit");
  if (!q || button.disabled) return;
  const submitLabel = button.querySelector(".submit-label");
  const idleLabel = submitLabel.textContent;
  checkDialogueScope();
  const scope = requestScope();
  const remember = $("remember-forecast").checked;
  button.disabled = true;
  $("question").disabled = true;
  const controls = document.querySelectorAll(".capability-query, #newquestion, #report-only, #choose-documents, #remember-forecast");
  controls.forEach(b => b.disabled = true);
  button.setAttribute("aria-busy", "true");
  submitLabel.textContent = "Обработка…";
  $("result").setAttribute("aria-busy", "true");
  const target = $("result");
  target.hidden = false;
  target.replaceChildren();
  const status = $("ask-status");
  status.hidden = false;
  status.textContent = "Агент готовит ответ…";
  const started = performance.now();
  try {
    const payload = { question: q, ...scope, remember_clarification: remember, ...(dialogueToken ? { dialogue_token: dialogueToken } : { previous }) };
    // The continuation token grants access to the dialogue. Keep it out of shared traces.
    const { dialogue_token: nextToken, ...data } = await askWithProgress(payload, () => {
      status.textContent = "Не удалось получить котировку через FRED. Проверяю другие источники…";
    });
    render(q, data, Math.round(performance.now() - started));
    if (remember && nextToken && (data.clarification || data.status === "unavailable")) {
      dialogueToken = nextToken;
      dialogueScope = JSON.stringify(scope);
      previous = null;
      checkDialogueScope();
    } else if (!data.clarification) {
      resetClarification();
    } else if (data.intent === "research") {
      dialogueToken = null;
      dialogueScope = null;
      // Keep the question that listed the options for replies like “the second one”.
      const clarification = "\nУточнение агента: " + data.answer.slice(0, 500);
      const userTurn = (previous ? "\nОтвет пользователя: " : "") + data.question;
      const available = 2000 - clarification.length;
      const retained = (previous || "").slice(0, Math.max(0, available - userTurn.length));
      previous = retained + userTurn.slice(0, available) + clarification;
    } else {
      dialogueToken = null;
      dialogueScope = null;
      previous = data.question.slice(0, 2000);
    }
    $("newquestion").hidden = !(previous || dialogueToken);
    if (previous || dialogueToken) {
      if (data.clarification) $("question").value = "";
      $("question").placeholder = data.intent === "research" ? "Ответьте на уточняющий вопрос…" :
        dialogueToken ? "Продолжите текущий прогноз…" : "Уточните параметры предыдущего запроса…";
      counter();
    } else {
      $("question").placeholder = defaultQuestionHint;
    }
  } catch (err) {
    target.replaceChildren(element("p", "Не удалось получить ответ. " + err.message));
  } finally {
    status.hidden = true;
    status.textContent = "";
    button.disabled = false;
    $("question").disabled = false;
    controls.forEach(b => b.disabled = false);
    button.removeAttribute("aria-busy");
    submitLabel.textContent = idleLabel;
    target.setAttribute("aria-busy", "false");
    if (previous || dialogueToken) $("question").focus();
  }
});
function showModel(settings) {
  const badge = $("model-badge");
  const configured = settings.llm_configured ?? Boolean(settings.model);
  const gigachat = settings.provider === "gigachat" || (!settings.provider && /^gigachat/i.test(settings.model || ""));
  badge.dataset.provider = configured ? (gigachat ? "gigachat" : "other") : "unconfigured";
  if (configured) {
    $("mode").textContent = settings.model || (gigachat ? "GigaChat" : "Настроена");
    badge.title = "Модель, выбранная в настройках сервера. Доступ к API проверяется командой check-llm.";
  } else {
    $("mode").textContent = gigachat ? "GigaChat · нет ключа" : "Не настроена";
    badge.title = "Укажите ключ модели в .env для работы с агентом.";
  }
}

fetch("/api/status")
  .then(response => {
    if (!response.ok) throw new Error("status");
    return response.json();
  })
  .then(settings => {
    showModel(settings);
    const routing = { loop: "Агентский цикл", workflow: "Фиксированный маршрут" }[settings.agent_mode];
    const stepLimit = settings.agent_mode === "loop" && settings.agent_max_steps
      ? `до ${settings.agent_max_steps} шагов` : "";
    const search = settings.tavily ? "Поиск через Tavily" : "Публикации из RSS";
    $("status-details").textContent = [routing, stepLimit, search, "Котировки FRED"].filter(Boolean).join(" · ");
    const reports = settings.reports || [];
    $("library").replaceChildren(...reports.map(report => link(report.title, report.url)));
    if (!reports.length) $("library").textContent = "Исходных отчётов пока нет.";
  })
  .catch(() => {
    $("model-badge").dataset.provider = "unavailable";
    $("model-badge").title = "Сервер не вернул настройки модели.";
    $("mode").textContent = "Статус неизвестен";
    $("status-details").textContent = "Не удалось загрузить настройки стенда.";
    $("library").textContent = "Не удалось загрузить список отчётов.";
  });
$("newquestion").addEventListener("click", () => selectQuery(""));
