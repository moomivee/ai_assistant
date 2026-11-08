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
  a.href = url;
  a.target = "_blank";
  a.rel = "noreferrer";
  return a;
}
function counter() {
  $("counter").textContent = $("question").value.length + " / 2000";
}
for (const [label, question] of scenarios) {
  const button = element("button", label, "sample");
  button.type = "button";
  button.addEventListener("click", () => {
    previous = null;
    $("newquestion").hidden = true;
    $("question").placeholder = "Например, чем отличаются Urals и ESPO?";
    $("question").value = question;
    counter();
    $("question").focus();
  });
  $("samples").append(button);
}
$("question").addEventListener("input", counter);
$("question").addEventListener("keydown", (e) => {
  if (e.key === "Enter" && (e.ctrlKey || e.metaKey)) {
    $("ask").requestSubmit();
  }
});
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
        row.page ? "стр. PDF " + row.page : null,
      ]
        .filter(Boolean)
        .join(" · "),
    ),
  );
  detail.append(element("p", row.text));
  const links = element("div", undefined, "links");
  if (row.local_url) links.append(link("Открыть страницу PDF ↗", row.local_url));
  links.append(link("Первоисточник ↗", row.url));
  detail.append(links);
  return detail;
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
function render(q, data) {
  const out = $("result");
  out.replaceChildren(element("h2", "Ответ аналитика"), element("div", q, "answer-q"));
  for (const p of data.answer.split("\n\n")) out.append(element("p", p));
  if (data.forecast) {
    const detail = element("details", undefined, "forecast-details");
    detail.append(element("summary", "Посмотреть расчёт по месяцам"), forecastTable(data.forecast));
    out.append(detail);
  }
  if (data.sources.length) {
    out.append(element("h3", "Источники", "sectiontitle"));
    for (const row of data.sources) out.append(source(row));
  }
  const details = element("details", undefined, "trace");
  details.append(element("summary", "Как подготовлен ответ"));
  details.append(
    element(
      "p",
      data.trace
        .map((e) => e.engine + " · " + e.tool + " · " + e.status + " · " + e.ms + " мс")
        .join("\n") || "Вызовы инструментов не потребовались.",
    ),
  );
  details.lastChild.style.whiteSpace = "pre-line";
  out.append(details);
}
$("ask").addEventListener("submit", async (e) => {
  e.preventDefault();
  const q = $("question").value.trim(),
    button = $("ask").querySelector(".submit");
  if (!q || button.disabled) return;
  button.disabled = true;
  $("question").disabled = true;
  document.querySelectorAll(".sample, #newquestion").forEach(b => b.disabled = true);
  button.textContent = "Готовим ответ…";
  $("result").setAttribute("aria-busy", "true");
  const target = $("result");
  target.hidden = false;
  target.replaceChildren(element("p", "Проверяю источники…", "busy"));
  try {
    const r = await fetch("/api/ask", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ question: q, previous }),
    });
    if (!r.ok) throw new Error("Сервис вернул ошибку " + r.status + ".");
    const data = await r.json();
    render(q, data);
    previous = data.clarification ? data.question.slice(0, 2000) : null;
    $("newquestion").hidden = !previous;
    if (previous) {
      $("question").value = "";
      $("question").placeholder = "Уточните параметры предыдущего запроса…";
      counter();
    }
  } catch (err) {
    target.replaceChildren(element("p", "Не удалось получить ответ. " + err.message));
  } finally {
    button.disabled = false;
    $("question").disabled = false;
    document.querySelectorAll(".sample, #newquestion").forEach(b => b.disabled = false);
    button.textContent = "Получить ответ ↗";
    target.setAttribute("aria-busy", "false");
    if (previous) $("question").focus();
  }
});
fetch("/api/status")
  .then((r) => { if (!r.ok) throw new Error("status"); return r.json(); })
  .then((s) => {
    $("mode").textContent = s.model ? "Анализ с AI" : "Базовый режим";
    $("library").replaceChildren();
    if (!s.reports.length) $("library").textContent = "Отчёты пока не добавлены.";
    for (const r of s.reports) $("library").append(link(r.title, r.url));
    if (s.history.length) {
      const first = s.history[0],
        last = s.history[s.history.length - 1];
      $("chartdates").textContent = first.month + " — " + last.month;
      historyRows = s.history;
      $("market").hidden = false;
    }
  })
  .catch(() => {
    $("mode").textContent = "Нет соединения";
    $("library").textContent = "Не удалось загрузить список. Попробуйте обновить страницу.";
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
