/* Local document catalog. All document content is rendered as text, never HTML. */
"use strict";
(() => {
  const $ = id => document.getElementById(id);
  const node = (tag, text, cls) => {
    const el = document.createElement(tag);
    if (text !== undefined) el.textContent = String(text);
    if (cls) el.className = cls;
    return el;
  };
  const selected = new Set();
  const updateCapabilities = new Map();
  let documents = [], loading = false, uploadDocumentId = null, previewDocument = null;
  let polling = null, pollsLeft = 0, previewSequence = 0;
  const statusLabels = { queued: "В очереди", processing: "Распознавание…", ready: "Текст готов", partial: "Готово с замечаниями", error: "Ошибка обработки", archived: "В архиве" };
  const idOf = doc => String(doc.document_id || doc.id);
  const versionOf = version => String(version.version_id || version.id);
  const usable = doc => !doc.archived && Boolean(doc.active_version_id);
  const path = id => "/api/knowledge/documents/" + encodeURIComponent(id);
  const versionPath = (id, version) => path(id) + "/versions/" + encodeURIComponent(version);
  const sizeLabel = size => size ? (size < 1024 * 1024 ? Math.ceil(size / 1024) + " КБ" : (size / 1024 / 1024).toFixed(1) + " МБ") : "";
  function anchor(text, url) {
    const el = node("a", text);
    const safe = typeof url === "string" && url.trim() ? TraceView.safeURL(url) : null;
    if (safe) el.href = safe;
    el.target = "_blank"; el.rel = "noreferrer";
    return el;
  }
  function button(text, fn, cls = "plain-button") {
    const el = node("button", text, cls);
    el.type = "button";
    el.addEventListener("click", fn);
    return el;
  }
  async function request(url, options = {}) {
    const response = await fetch(url, options);
    let data;
    try { data = await response.json(); } catch { data = {}; }
    if (!response.ok) {
      const detail = typeof data.detail === "string" ? data.detail : Array.isArray(data.detail) ? data.detail.map(item => item.msg).join("; ") : "Ошибка сервера " + response.status;
      throw new Error(detail);
    }
    return data;
  }
  function feedback(id, text, error = false) {
    $(id).textContent = text;
    $(id).classList.toggle("is-error", error);
  }
  function updateSelection() {
    const count = selected.size;
    $("scope-summary").textContent = count ? "Выбрано документов: " + count : "Все активные документы";
    $("selection-label").textContent = count ? "Для вопроса выбрано: " + count : "Документы для вопроса не выбраны";
    $("clear-selection").hidden = !count;
    $("ask-selected").hidden = !count;
  }
  function clearSelection() { selected.clear(); updateSelection(); renderCatalog(); }
  function askSelected(doc = null) {
    if ($("question").disabled) {
      feedback("knowledge-feedback", "Дождитесь завершения текущего вопроса."); return;
    }
    if (doc) { selected.clear(); selected.add(idOf(doc)); }
    updateSelection(); renderCatalog();
    $("report-only").checked = true;
    window.Workbench.updateScopeHint();
    window.Workbench.selectQuery(doc ? 'Какие основные выводы содержит документ «' + doc.title + '»?' : "", true);
  }
  window.KnowledgeUI = { selectedIds: () => [...selected], clearSelection };
  function renderCatalog() {
    const filter = $("document-filter").value.trim().toLocaleLowerCase("ru");
    const shown = documents.filter(doc => (!doc.archived || $("show-archived").checked) && [doc.title, doc.publisher, doc.filename].join(" ").toLocaleLowerCase("ru").includes(filter));
    $("document-list").replaceChildren();
    $("knowledge-count").textContent = String(documents.filter(doc => !doc.archived).length);
    if (!shown.length) {
      $("document-list").append(node("p", filter ? "Документов с таким названием не найдено." : "Здесь появятся документы базы знаний. Добавьте первый файл слева.", "empty-catalog"));
      return;
    }
    for (const doc of shown) {
      const id = idOf(doc);
      const card = node("article", undefined, "document-card");
      const heading = node("div", undefined, "document-heading");
      const check = node("input"); check.type = "checkbox";
      check.checked = selected.has(id); check.disabled = !usable(doc);
      check.setAttribute("aria-label", "Выбрать документ: " + doc.title);
      check.addEventListener("change", () => { if (check.checked) selected.add(id); else selected.delete(id); updateSelection(); });
      const title = node("div", undefined, "document-name");
      title.append(node("h3", doc.title || doc.filename || "Документ"), node("p", [doc.publisher, doc.published_date || "Дата не указана", doc.builtin ? "Из исходного проекта" : "Добавлен пользователем"].filter(Boolean).join(" · "), "small"));
      heading.append(check, title);
      const status = node("span", statusLabels[doc.status] || doc.status || "Статус неизвестен", "document-status " + doc.status);
      card.append(heading, status);
      card.append(node("p", [doc.filename, sizeLabel(doc.size_bytes), doc.versions_count ? "Версий: " + doc.versions_count : null, doc.block_count ? "Фрагментов: " + doc.block_count : null].filter(Boolean).join(" · "), "document-meta"));
      if (usable(doc)) {
        card.append(node("p", doc.current_version_id !== doc.active_version_id ? "В поиске используется предыдущая готовая версия." : doc.index_status === "indexed" ? "Доступен для поиска · индекс построен" : "Доступен для поиска · индекс строится при первом запросе", "document-ready"));
      }
      if (doc.warnings?.length) card.append(node("p", doc.warnings.join(" · "), "document-warning"));
      if (doc.error) card.append(node("p", doc.error, "document-warning is-error"));
      const origin = node("div", undefined, "document-origin");
      if (doc.source_url) {
        origin.append(anchor("Первоисточник ↗", doc.source_url));
        origin.append(node("span", updateCapabilities.get(id)?.supported ? "Можно проверить файл по прежнему URL. Новые выпуски по другим ссылкам не ищутся." : "Обновление: загрузите новую версию файла. Автоскачивание не включено.", "small"));
      } else origin.append(node("span", "Ссылка на первоисточник не указана. Новую версию можно добавить вручную.", "small"));
      card.append(origin);
      const actions = node("div", undefined, "document-actions");
      actions.append(button("Просмотр и версии", () => openPreview(doc)));
      if (usable(doc)) actions.append(button("Задать вопрос", () => askSelected(doc)));
      if (!doc.archived) {
        if (updateCapabilities.get(id)?.supported) actions.append(button("Проверить исходный файл", event => checkUpdate(doc, event.currentTarget)));
        actions.append(button("Новая версия", () => beginVersion(doc)));
        if (doc.status === "error") actions.append(button("Повторить обработку", event => retry(doc, event.currentTarget)));
        actions.append(button("В архив", event => archive(doc, event.currentTarget)));
      }
      card.append(actions); $("document-list").append(card);
    }
  }
  async function refresh({ poll = false } = {}) {
    if (loading) return;
    loading = true; $("refresh-knowledge").disabled = true;
    try {
      const data = await request("/api/knowledge/documents?include_archived=true");
      documents = Array.isArray(data.documents) ? data.documents : [];
      for (const id of selected) if (!documents.some(doc => idOf(doc) === id && usable(doc))) selected.delete(id);
      updateSelection(); renderCatalog();
      loadUpdateCapabilities();
      const busy = documents.some(doc => ["queued", "processing"].includes(doc.status));
      feedback("knowledge-feedback", busy ? "Файл обрабатывается. Список обновляется автоматически." : "Документов в базе: " + documents.filter(doc => !doc.archived).length);
      if (busy) {
        if (!poll) pollsLeft = 60;
        if (pollsLeft > 0) {
          pollsLeft--; clearTimeout(polling);
          polling = setTimeout(() => refresh({ poll: true }), 2000);
        } else feedback("knowledge-feedback", "Обработка ещё идёт. Автообновление приостановлено; нажмите «Обновить список» позже.");
      } else { clearTimeout(polling); polling = null; }
    } catch (error) {
      feedback("knowledge-feedback", "Не удалось загрузить базу знаний. " + error.message, true);
    } finally { loading = false; $("refresh-knowledge").disabled = false; }
  }
  async function loadUpdateCapabilities() {
    const pending = documents.filter(doc => doc.builtin && !doc.archived && !updateCapabilities.has(idOf(doc)));
    for (const doc of pending) updateCapabilities.set(idOf(doc), { supported: false });
    await Promise.all(pending.map(async doc => {
      try { updateCapabilities.set(idOf(doc), await request(path(idOf(doc)) + "/update-capability")); }
      catch { /* Manual upload remains available when an adapter is unavailable. */ }
    }));
    if (pending.length) renderCatalog();
  }
  async function checkUpdate(doc, actionButton) {
    actionButton.disabled = true;
    feedback("knowledge-feedback", "Проверяем PDF по прежнему адресу издателя…");
    try {
      const result = await request(path(idOf(doc)) + "/check-update", { method: "POST" });
      await refresh();
      feedback("knowledge-feedback", [result.status === "queued" ? "Издатель изменил файл. Новая версия добавлена на обработку." : result.status === "known_version" ? "Эта версия файла уже есть в базе." : "По прежнему адресу находится тот же файл.", ...(result.notices || [])].join(" "));
    } catch (error) { feedback("knowledge-feedback", "Не удалось проверить источник. " + error.message, true); actionButton.disabled = false; }
  }
  function beginVersion(doc) {
    uploadDocumentId = idOf(doc);
    $("upload-title").textContent = "Новая версия документа";
    $("version-notice").hidden = false;
    $("version-notice").textContent = doc.title + ". Предыдущие версии сохранятся.";
    $("document-title").value = doc.title || "";
    $("document-publisher").value = doc.publisher || "";
    $("document-url").value = doc.source_url || "";
    $("document-date").value = /^\d{4}-\d{2}-\d{2}$/.test(doc.published_date || "") ? doc.published_date : "";
    $("cancel-version").hidden = false;
    $("upload-button").textContent = "Добавить новую версию";
    $("document-file").value = "";
    feedback("upload-feedback", "Выберите файл новой версии.");
    $("document-file").focus();
  }
  function resetUpload() {
    uploadDocumentId = null; $("upload-document").reset();
    $("upload-title").textContent = "Добавить документ";
    $("version-notice").hidden = true; $("cancel-version").hidden = true;
    $("upload-button").textContent = "Добавить в базу";
  }
  $("upload-document").addEventListener("submit", async event => {
    event.preventDefault();
    const form = $("upload-document"), data = new FormData(form);
    if (uploadDocumentId) data.append("document_id", uploadDocumentId);
    $("upload-button").disabled = true;
    form.setAttribute("aria-busy", "true");
    feedback("upload-feedback", "Передаём файл… Не закрывайте страницу.");
    try {
      const result = await request("/api/knowledge/documents", { method: "POST", body: data });
      resetUpload();
      feedback("upload-feedback", "Файл принят: " + (result.document?.title || "документ") + ". Статус распознавания появится в списке.");
      await refresh();
    } catch (error) { feedback("upload-feedback", "Не удалось добавить файл. " + error.message, true); }
    finally { $("upload-button").disabled = false; form.setAttribute("aria-busy", "false"); }
  });
  async function archive(doc, actionButton) {
    actionButton.disabled = true;
    try {
      await request(path(idOf(doc)) + "/archive", { method: "POST" });
      selected.delete(idOf(doc)); await refresh();
    } catch (error) { feedback("knowledge-feedback", "Не удалось архивировать. " + error.message, true); actionButton.disabled = false; }
  }
  async function retry(doc, actionButton) {
    actionButton.disabled = true;
    try {
      await request(versionPath(idOf(doc), doc.current_version_id) + "/retry", { method: "POST" });
      await refresh();
    } catch (error) { feedback("knowledge-feedback", "Повторная обработка не запущена. " + error.message, true); actionButton.disabled = false; }
  }
  async function openPreview(doc) {
    previewDocument = doc; const sequence = ++previewSequence;
    $("preview-title").textContent = doc.title;
    $("preview-meta").textContent = [doc.publisher, doc.published_date].filter(Boolean).join(" · ");
    $("preview-blocks").replaceChildren(); $("preview-links").replaceChildren(); $("preview-version").replaceChildren();
    feedback("preview-feedback", "Загружаем версии…");
    if (!$("document-preview").open) $("document-preview").showModal();
    try {
      const detail = await request(path(idOf(doc)));
      if (sequence !== previewSequence) return;
      previewDocument = detail.document || detail;
      const versions = previewDocument.versions || detail.versions || [];
      for (const version of versions) {
        const id = versionOf(version), option = node("option", [id === String(doc.active_version_id) ? "В поиске" : "Версия", version.created_at?.slice(0, 16).replace("T", " ") || id.slice(0, 10), statusLabels[version.status] || version.status].filter(Boolean).join(" · "));
        option.value = id; $("preview-version").append(option);
      }
      if (!versions.length && doc.current_version_id) {
        const option = node("option", "Текущая версия"); option.value = doc.current_version_id; $("preview-version").append(option);
      }
      $("preview-version").value = doc.active_version_id || doc.current_version_id || "";
      if (!$("preview-version").value) $("preview-version").selectedIndex = 0;
      await loadBlocks();
    } catch (error) { if (sequence === previewSequence) feedback("preview-feedback", "Не удалось открыть документ. " + error.message, true); }
  }
  async function loadBlocks() {
    const doc = previewDocument, version = $("preview-version").value, sequence = ++previewSequence;
    $("preview-blocks").replaceChildren(); $("preview-links").replaceChildren();
    if (!doc || !version) { feedback("preview-feedback", "Версий для просмотра пока нет."); return; }
    const url = versionPath(idOf(doc), version);
    $("preview-links").append(anchor("Открыть оригинал ↗", url + "/file"));
    if (doc.source_url) $("preview-links").append(anchor("Первоисточник ↗", doc.source_url));
    feedback("preview-feedback", "Загружаем распознанный текст…");
    try {
      const data = await request(url + "/blocks");
      if (sequence !== previewSequence) return;
      const blocks = data.blocks || [];
      feedback("preview-feedback", [blocks.length ? "Фрагментов: " + blocks.length + ". Сверьте цифры и таблицы с оригиналом." : "Распознанных фрагментов пока нет. Проверьте статус обработки.", data.error, ...(data.warnings || [])].filter(Boolean).join(" "), Boolean(data.error));
      for (const block of blocks) {
        const section = node("section", undefined, "preview-block");
        const label = window.Workbench.locatorLabel(block.locator, block.page) || "Фрагмент документа";
        section.append(node("h3", label), node("span", block.method === "ocr" ? "OCR · распознанный текст" : "Текст документа", "small"), node("pre", block.text || ""));
        if (block.page) section.append(anchor("Открыть страницу " + block.page + " ↗", url + "/file#page=" + block.page));
        $("preview-blocks").append(section);
      }
    } catch (error) { if (sequence === previewSequence) feedback("preview-feedback", "Не удалось загрузить текст. " + error.message, true); }
  }
  $("close-preview").addEventListener("click", () => $("document-preview").close());
  $("document-preview").addEventListener("close", () => { previewSequence++; });
  $("preview-version").addEventListener("change", loadBlocks);
  $("cancel-version").addEventListener("click", () => { resetUpload(); feedback("upload-feedback", ""); });
  $("refresh-knowledge").addEventListener("click", () => refresh());
  $("document-filter").addEventListener("input", renderCatalog);
  $("show-archived").addEventListener("change", renderCatalog);
  $("clear-selection").addEventListener("click", clearSelection);
  $("ask-selected").addEventListener("click", () => askSelected());
  refresh();
})();

/* Research notes are user-saved drafts, never RAG evidence. */
(() => {
  const $ = id => document.getElementById(id);
  let noteId = null;
  const node = (tag, text, cls) => {
    const el = document.createElement(tag);
    if (text !== undefined) el.textContent = String(text);
    if (cls) el.className = cls;
    return el;
  };
  async function request(url, body, method = "POST") {
    const response = await fetch(url, body === undefined ? {} : { method, headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) });
    const data = await response.json();
    if (!response.ok) throw new Error(typeof data.detail === "string" ? data.detail : data.detail?.message || "Ошибка сервера " + response.status);
    return data.note || data;
  }
  function clearContext() {
    noteId = null; $("note-context").hidden = true;
    $("question").placeholder = "Например, чем отличаются Urals и ESPO?";
  }
  async function refresh() {
    $("notes-feedback").textContent = "Загружаем заметки…";
    try {
      const data = await request("/api/notes");
      $("notes-list").replaceChildren();
      const notes = Array.isArray(data.notes) ? data.notes : [];
      $("notes-feedback").textContent = notes.length ? "Заметок: " + notes.length : "Заметок пока нет. Сохраните результат после ответа агента.";
      for (const note of notes) {
        const detail = node("details", undefined, "research-note");
        detail.append(node("summary", note.title || note.question));
        detail.append(node("p", ["Черновик исследования", "Версия " + String(note.revision).slice(0, 8), note.updated_at?.slice(0, 10)].filter(Boolean).join(" · "), "small"));
        const body = node("div", "Загружаем текст…", "note-body"); detail.append(body);
        let loaded = false;
        detail.addEventListener("toggle", async () => {
          if (!detail.open || loaded) return;
          try {
            const full = await request("/api/notes/" + encodeURIComponent(note.note_id));
            body.replaceChildren(node("p", full.question, "small"), node("p", full.answer, "saved-answer"));
            if (full.source_refs?.length) {
              const refs = node("details", undefined, "note-references");
              refs.append(node("summary", "Ссылки, сохранённые с ответом"));
              for (const ref of full.source_refs) {
                const row = node("p", [ref.id, ref.title, ref.date, window.Workbench.locatorLabel(ref.locator, ref.page)].filter(Boolean).join(" · "), "small");
                const url = ref.local_url || ref.url;
                const safe = typeof url === "string" && url ? TraceView.safeURL(url) : null;
                if (safe) {
                  const anchor = node("a", " Открыть ↗"); anchor.href = safe; anchor.target = "_blank"; anchor.rel = "noreferrer"; row.append(anchor);
                }
                refs.append(row);
              }
              body.append(refs);
            }
            const actions = node("div", undefined, "document-actions");
            const use = node("button", "Продолжить исследование", "plain-button"); use.type = "button";
            use.addEventListener("click", () => {
              if ($("question").disabled) return;
              window.KnowledgeUI?.clearSelection();
              window.Workbench.selectQuery("", true);
              noteId = full.note_id;
              $("note-context-title").textContent = "Контекст заметки: " + full.title;
              $("note-context").hidden = false;
              $("question").placeholder = "Что уточнить или проверить в этом исследовании?";
            });
            const archive = node("button", "В архив", "plain-button"); archive.type = "button";
            archive.addEventListener("click", async () => {
              archive.disabled = true;
              try {
                await request("/api/notes/" + encodeURIComponent(full.note_id) + "/archive", { expected_revision: full.revision, reason: "Архивировано пользователем в интерфейсе" });
                if (noteId === full.note_id) clearContext();
                await refresh();
              } catch (error) { $("notes-feedback").textContent = error.message; archive.disabled = false; }
            });
            actions.append(use, archive); body.append(actions); loaded = true;
          } catch (error) { body.textContent = "Не удалось открыть заметку. " + error.message; }
        });
        $("notes-list").append(detail);
      }
    } catch (error) { $("notes-feedback").textContent = "Заметки недоступны. " + error.message; }
  }
  async function save(question, data, button, feedback) {
    button.disabled = true; feedback.textContent = "Сохраняем…";
    try {
      await request("/api/notes", { title: question.slice(0, 120), question, answer: data.answer, source_refs: data.sources || [], trace_id: data.trace_id || "", unresolved: data.status === "partial" ? ["Ответ отмечен как частичный. Требуется проверка полноты."] : [] });
      feedback.textContent = "Сохранено как непроверенная заметка исследования. Открыть её можно в разделе «Заметки исследования» ниже.";
      button.textContent = "Заметка сохранена";
      if ($("notes-panel").open) await refresh();
    } catch (error) { feedback.textContent = "Не удалось сохранить. " + error.message; button.disabled = false; }
  }
  window.ResearchNotes = { save, selectedId: () => noteId, clearContext };
  $("refresh-notes").addEventListener("click", refresh);
  $("notes-panel").addEventListener("toggle", () => { if ($("notes-panel").open) refresh(); });
  $("clear-note-context").addEventListener("click", clearContext);
})();
