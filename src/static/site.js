"use strict";
const $ = (id) => document.getElementById(id);
let previous = null;
let historyRows = [];
let chartRendered = false;
const scenarios = [
  ["Сравнить Urals и ESPO", "Чем отличаются Urals и ESPO по плотности и содержанию серы?"],
  ["Последняя цена Brent", "Какая последняя опубликованная спотовая цена Brent?"],
  [
    "Сопоставить отчёт и рынок",
    "Чем по качеству отличаются Urals и ESPO и какова последняя цена Brent?",
  ],
  ["Спрогнозировать Brent", "Спрогнозируй среднемесячную цену Brent на 3 месяца."],

  ["Рассчитать сценарий Urals", "Спрогнозируй Urals на 3 месяца при дисконте 12 долларов к Brent."],
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
function selectQuery(question, keepScope = false) {
  if ($("question").disabled) return;
  previous = null;
  $("newquestion").hidden = true;
  $("question").placeholder = "Например, чем отличаются Urals и ESPO?";
  $("question").value = question;
  if (!keepScope) {
    window.KnowledgeUI?.clearSelection();
    window.ResearchNotes?.clearContext();
    $("report-only").checked = false;
    updateScopeHint();
  }
  counter();
  showTab("test");
  $("question").focus();
}
function updateScopeHint() {
  $("scope-hint").textContent = $("report-only").checked
    ? "Используются только документы базы знаний. Веб-поиск отключён для этого вопроса."
    : "Агент выбирает документы или веб по задаче. Для актуальных событий может сразу обратиться к интернету.";
}
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
$("report-only").addEventListener("change", updateScopeHint);
$("choose-documents").addEventListener("click", () => {
  showTab("knowledge"); $("document-filter").focus();
});
window.Workbench = { showTab, selectQuery, updateScopeHint, locatorLabel };
function counter() {
  $("counter").textContent = $("question").value.length + " / 2000";
}
for (const [label, question] of scenarios) {
  const button = element("button", label, "sample");
  button.type = "button";
  button.addEventListener("click", () => selectQuery(question));
  $("samples").append(button);
}
$("question").addEventListener("input", counter);
$("question").addEventListener("keydown", (e) => {
  if (e.key === "Enter" && (e.ctrlKey || e.metaKey)) {
    $("ask").requestSubmit();
  }
});
function renderCapabilities(settings) {
  const configured = settings?.llm_configured ?? Boolean(settings?.model);
  const tasks = [
    { label: "Найти ответ в документах", text: "Показатели, определения и сравнения из загруженных отчётов — со ссылкой на конкретный фрагмент.", query: scenarios[0][1], note: "Приоритет базы знаний" },
    { label: "Проверить свежие сведения", text: "Новости, события и последние опубликованные цены. В ответе веб-источники отделены от отчётов.", query: scenarios[1][1], note: settings?.tavily ? "Tavily настроен · доступ ещё не проверен" : settings ? "Без Tavily: ограниченные RSS и FRED" : "Настройки веб-поиска неизвестны" },
    { label: "Сопоставить несколько источников", text: "Объединить сведения из документов и интернета, показать подтверждения и недостающие данные.", query: scenarios[2][1], note: "Документы + веб-источники" },
    { label: "Рассчитать прогноз", text: "Brent и WTI по историческим ценам; Urals — через заданный дисконт к Brent. Методы, интервал и историческая ошибка видны в результате.", query: scenarios[3][1], note: "ARIMA + экспоненциальное сглаживание" },
    { label: "Проверить сценарий", text: "Оценить сценарий с заданными допущениями. Дисконт и чувствительность задаются явно, а не угадываются моделью.", query: scenarios[4][1], note: "Условный расчёт по вашим параметрам" },
    { label: "Разобрать собственный документ", text: "Добавить PDF, изображение, DOCX или таблицу. Проверить результат распознавания и ограничить вопрос выбранными документами.", query: null, note: "Локальная обработка и OCR" },
  ];
  $("capabilities").replaceChildren();
  for (const [index, task] of tasks.entries()) {
    const card = element("article", undefined, "capability-card");
    card.append(element("span", String(index + 1).padStart(2, "0"), "capability-number"), element("h3", task.label), element("p", task.text), element("span", task.note, "capability-note"));
    const button = element("button", task.query ? "Подставить пример ↗" : "Открыть базу знаний ↗", "plain-button capability-query");
    button.type = "button";
    button.addEventListener("click", () => task.query ? selectQuery(task.query) : showTab("knowledge"));
    card.append(button);
    if (task.query && !configured) card.append(element("span", settings ? "Для ответа нужен настроенный LLM-клиент" : "Состояние модели неизвестно", "small"));
    $("capabilities").append(card);
  }
}
renderCapabilities(null);
fetch("/api/rag/status").then(response => {
  if (!response.ok) throw new Error("rag");
  return response.json();
}).then(data => {
  const lines = [data.model ? "Эмбеддер: " + data.model : null, data.model_repo ? "Модель: " + data.model_repo : null, data.search_mode || data.mode ? "Режим: " + (data.search_mode || data.mode) : null, data.top_k ? "Фрагментов за поиск: " + data.top_k : null, data.status ? "Индекс: " + (data.status === "ready" ? "готов" : "ожидает построения при запросе") : null];
  $("rag-settings").textContent = lines.join(" · ") || "Настройки эмбеддера сервером не переданы.";
}).catch(() => { $("rag-settings").textContent = "Настройки поиска сейчас недоступны. Метрики здесь не измерялись."; });
function renderChart(rows) {
  const NS = "http://www.w3.org/2000/svg";
  const svg = document.createElementNS(NS, "svg"),
    width = 680,
    height = 160,
    left = 40,
    right = 16,
    top = 12,
    bottom = 27;
  svg.setAttribute("viewBox", "0 0 " + width + " " + height);
  svg.setAttribute("role", "img");
  svg.setAttribute("aria-label", "Среднемесячные цены Brent и WTI, доллары за баррель");
  const vals = rows.flatMap((r) => [r.Brent, r.WTI]),
    low = Math.floor(Math.min(...vals) / 10) * 10,
    high = Math.max(low + 10, Math.ceil(Math.max(...vals) / 10) * 10);
  const x = (i) => left + ((width - left - right) * i) / Math.max(rows.length - 1, 1),
    y = (v) => top + ((high - v) * (height - top - bottom)) / (high - low);
  function add(tag, attrs, value) {
    const n = document.createElementNS(NS, tag);
    for (const [k, v] of Object.entries(attrs)) n.setAttribute(k, v);
    if (value !== undefined) n.textContent = value;
    svg.append(n);
  }
  for (const v of [low, (low + high) / 2, high]) {
    add("line", {
      x1: left,
      x2: width - right,
      y1: y(v),
      y2: y(v),
      stroke: "#dbe3d8",
      "stroke-dasharray": "3 4",
    });
    add("text", { x: left - 8, y: y(v) + 4, "text-anchor": "end" }, v);
  }
  for (const [key, color] of [
    ["Brent", "#236447"],
    ["WTI", "#a96e36"],
  ]) {
    add("polyline", {
      points: rows.map((r, i) => x(i) + "," + y(r[key])).join(" "),
      fill: "none",
      stroke: color,
      "stroke-width": "2.6",
      "stroke-linejoin": "round",
    });
  }
  for (const i of [0, Math.floor(rows.length / 2), rows.length - 1])
    add(
      "text",
      {
        x: x(i),
        y: height - 5,
        "text-anchor": i === 0 ? "start" : i === rows.length - 1 ? "end" : "middle",
      },
      rows[i].month,
    );
  $("historychart").replaceChildren(svg);
}
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
  const source = (data.sources || []).find(row => row.id === sourceId);
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
    const id = match[1], { source, rows } = citationData(data, text, id);
    if (!source && !rows.length) {
      paragraph.append(document.createTextNode(match[0]));
      start = match.index + match[0].length;
      continue;
    }
    const button = element("button", match[0], "citation-button");
    const box = element("aside", undefined, "citation-excerpt");
    button.type = "button";
    button.setAttribute("aria-label", "Показать подтверждение " + id);
    box.id = "citation-" + (++citationNumber);
    button.setAttribute("aria-controls", box.id);
    button.setAttribute("aria-expanded", "false");
    box.hidden = true;
    box.append(element("strong", source?.title || id));
    box.append(element("span", rows.length ? "Фрагмент, подтверждающий утверждение" : source?.kind === "model" ? "Описание расчёта" : "Найденный фрагмент источника; точная привязка к утверждению отсутствует", "citation-kind"));
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
function render(q, data, elapsed) {
  const out = $("result");
  const heading = element("div", undefined, "result-heading");
  heading.append(element("h2", "Результат проверки"), element("span", TraceView.statusLabel(data.status), "response-status " + data.status));
  const answer = element("section", undefined, "answer-panel");
  answer.setAttribute("aria-label", "Ответ аналитика");
  answer.append(element("h3", "Ответ аналитика"), element("div", q, "answer-q"));
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
  if (window.ResearchNotes && !data.clarification && ["ok", "partial"].includes(data.status)) {
    const save = element("button", "Сохранить в заметки", "secondary-button save-note");
    save.type = "button";
    const feedback = element("p", undefined, "small"); feedback.setAttribute("role", "status");
    save.addEventListener("click", () => window.ResearchNotes.save(q, data, save, feedback));
    answer.append(save, feedback);
  }
  const grid = element("div", undefined, "result-grid");
  grid.append(answer, TraceView.render(data, elapsed));
  out.replaceChildren(heading, grid);
}
$("ask").addEventListener("submit", async (e) => {
  e.preventDefault();
  const q = $("question").value.trim(),
    button = $("ask").querySelector(".submit");
  if (!q || button.disabled) return;
  button.disabled = true;
  $("question").disabled = true;
  document.querySelectorAll(".sample, .capability-query, #newquestion, #report-only, #choose-documents").forEach(b => b.disabled = true);
  button.textContent = "Готовим ответ…";
  $("result").setAttribute("aria-busy", "true");
  const target = $("result");
  target.hidden = false;
  target.replaceChildren(element("p", "Выполняется запрос. Ответ и шаги появятся после завершения.", "busy"));
  const started = performance.now();
  try {
    const r = await fetch("/api/ask", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ question: q, previous, document_ids: window.KnowledgeUI?.selectedIds() || [], report_only: $("report-only").checked, note_id: window.ResearchNotes?.selectedId() || null }),
    });
    if (!r.ok) throw new Error("Сервис вернул ошибку " + r.status + ".");
    const data = await r.json();
    render(q, data, Math.round(performance.now() - started));
    if (!data.clarification) {
      previous = null;
    } else if (data.intent === "research") {
      // Keep the question that listed the options for replies like “the second one”.
      const clarification = "\nУточнение агента: " + data.answer.slice(0, 500);
      const userTurn = (previous ? "\nОтвет пользователя: " : "") + data.question;
      const available = 2000 - clarification.length;
      const retained = (previous || "").slice(0, Math.max(0, available - userTurn.length));
      previous = retained + userTurn.slice(0, available) + clarification;
    } else {
      previous = data.question.slice(0, 2000);
    }
    $("newquestion").hidden = !previous;
    if (previous) {
      $("question").value = "";
      $("question").placeholder = "Уточните параметры предыдущего запроса…";
      counter();
    } else {
      $("question").placeholder = "Например, чем отличаются Urals и ESPO?";
    }
  } catch (err) {
    target.replaceChildren(element("p", "Не удалось получить ответ. " + err.message));
  } finally {
    button.disabled = false;
    $("question").disabled = false;
    document.querySelectorAll(".sample, .capability-query, #newquestion, #report-only, #choose-documents").forEach(b => b.disabled = false);
    button.textContent = "Получить ответ ↗";
    target.setAttribute("aria-busy", "false");
    if (previous) $("question").focus();
  }
});
fetch("/api/status")
  .then((r) => { if (!r.ok) throw new Error("status"); return r.json(); })
  .then((s) => {
    $("mode").textContent = s.model ? s.model + " · настроена" : "LLM не настроена";
    $("mode").title = s.model ? "Модель из настроек. Доступ к API проверяется отдельно командой check-llm." : "Добавьте ключ выбранного провайдера в .env.";
    $("environment").textContent = [s.agent_mode === "loop" ? "Агентский цикл" : s.agent_mode === "workflow" ? "Фиксированный workflow" : "Режим не указан", s.agent_max_steps ? `до ${s.agent_max_steps} решений` : null, s.tavily ? "Веб: Tavily + FRED" : "Веб: RSS + FRED"].filter(Boolean).join(" · ");
    $("library").replaceChildren();
    if (!s.reports?.length) $("library").textContent = "Отчёты пока не добавлены.";
    for (const r of s.reports || []) $("library").append(link(r.title, r.url));
    renderCapabilities(s);
    if (s.history?.length) {
      const first = s.history[0],
        last = s.history[s.history.length - 1];
      $("chartdates").textContent = first.month + " — " + last.month;
      historyRows = s.history;
      $("market").hidden = false;
    }
  })
  .catch(() => {
    $("mode").textContent = "Нет соединения";
    $("environment").textContent = "Не удалось получить конфигурацию сервера.";
    $("library").textContent = "Не удалось загрузить список. Попробуйте обновить страницу.";
    renderCapabilities(null);
  });

$("market").addEventListener("toggle", () => {
  if ($("market").open && !chartRendered && historyRows.length) {
    renderChart(historyRows);
    chartRendered = true;
  }
});
$("newquestion").addEventListener("click", () => {
  previous = null;
  $("question").value = "";
  $("question").placeholder = "Например, чем отличаются Urals и ESPO?";
  $("newquestion").hidden = true;
  counter();
  $("question").focus();
});
