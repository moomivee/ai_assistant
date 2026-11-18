"""Bounded, local-only extraction with honest, reproducible source locations.

OCR is Tesseract, not a call to an LLM. Office formulas/macros are never executed.
An extraction warning makes the knowledge-store version partial, not ready.
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
import logging
import math
import shutil
import subprocess
import tempfile
import threading
import unicodedata
import warnings as python_warnings
from pathlib import Path, PurePosixPath
from zipfile import BadZipFile, ZipFile

from PIL import Image, ImageOps

from ..config import MAX_UPLOAD_BYTES as MAX_FILE_BYTES
from ..config import MAX_UPLOAD_MIB

logger = logging.getLogger(__name__)

PARSER_VERSION = "local-documents/2"
MAX_PAGES = 500
MAX_IMAGE_PIXELS = 20_000_000
MAX_RENDER_EDGE = 20_000
MAX_ARCHIVE_BYTES = 100 * 1024 * 1024
MAX_ARCHIVE_ENTRIES = 2000
MAX_ROWS = 20_000
MAX_CELLS = 200_000
MAX_COLUMNS = 1000
MAX_TEXT_CHARS = 4_000_000
OCR_TIMEOUT_SECONDS = 45
TABLE_GROUP_ROWS = 40
_PDFIUM_LOCK = threading.Lock()
SUPPORTED_EXTENSIONS = {".pdf", ".png", ".jpg", ".jpeg", ".docx", ".xlsx", ".csv"}


class _Result:
    def __init__(self, digest: str):
        self.digest = digest
        self.blocks: list[dict] = []
        self.warnings: list[str] = []
        self.characters = 0

    def warn(self, message: str) -> None:
        if message not in self.warnings:
            self.warnings.append(message)

    def add(self, text: str, locator: dict, method: str = "native", **metadata) -> None:
        text = text.replace("\x00", "").strip()
        if not text:
            return
        self.characters += len(text)
        if self.characters > MAX_TEXT_CHARS:
            raise ValueError("Документ содержит слишком много текста. Разделите его на части.")
        identity = json.dumps(
            [self.digest, locator, method, text], ensure_ascii=False, sort_keys=True
        )
        self.blocks.append(
            {
                "block_id": hashlib.sha256(identity.encode()).hexdigest()[:24],
                "text": text,
                "page": locator.get("page"),
                "locator": locator,
                "method": method,
                **metadata,
            }
        )

    def finish(self) -> dict:
        if not self.blocks:
            detail = " ".join(self.warnings)
            raise ValueError("Не удалось извлечь текст из документа. " + detail)
        return {"blocks": self.blocks, "warnings": self.warnings, "parser_version": PARSER_VERSION}


def extract(path: Path, filename: str) -> dict:
    """Extract a stored object (its on-disk path need not have an extension)."""
    path = Path(path)
    suffix = Path(filename).suffix.lower()
    if suffix not in SUPPORTED_EXTENSIONS:
        raise ValueError("Поддерживаются PDF, PNG, JPEG, DOCX, XLSX и CSV.")
    if not path.is_file() or not 0 < path.stat().st_size <= MAX_FILE_BYTES:
        raise ValueError(f"Файл пуст, недоступен или превышает ограничение {MAX_UPLOAD_MIB} МБ.")
    with path.open("rb") as stream:
        result = _Result(hashlib.file_digest(stream, "sha256").hexdigest())
    logger.info(
        "document.extraction.started", extra={"format": suffix, "bytes": path.stat().st_size}
    )
    try:
        if suffix == ".pdf":
            _pdf(path, result)
        elif suffix in {".png", ".jpg", ".jpeg"}:
            _image_file(path, result)
        elif suffix in {".docx", ".xlsx"}:
            _validate_archive(path, suffix)
            (_docx if suffix == ".docx" else _xlsx)(path, result)
        else:
            _csv(path, result)
        output = result.finish()
    except ValueError:
        raise
    except Exception as exc:
        logger.warning(
            "document.extraction.failed", extra={"format": suffix, "error_type": type(exc).__name__}
        )
        raise ValueError(
            "Документ повреждён или его содержимое не соответствует формату файла."
        ) from exc
    logger.info(
        "document.extraction.finished",
        extra={
            "format": suffix,
            "blocks": len(result.blocks),
            "warnings_count": len(result.warnings),
        },
    )
    return output


def _validate_archive(path: Path, suffix: str) -> None:
    from defusedxml import ElementTree
    from defusedxml.common import DefusedXmlException

    try:
        with ZipFile(path) as archive:
            infos = archive.infolist()
            if (
                len(infos) > MAX_ARCHIVE_ENTRIES
                or sum(i.file_size for i in infos) > MAX_ARCHIVE_BYTES
            ):
                raise ValueError("Распакованный документ слишком велик. Разделите его на части.")
            names = {info.filename for info in infos}
            required = "word/document.xml" if suffix == ".docx" else "xl/workbook.xml"
            if required not in names or len(names) != len(infos):
                raise ValueError("Некорректный архив Office или повторяющиеся имена его частей.")
            for info in infos:
                parts = PurePosixPath(info.filename).parts
                if ".." in parts or info.filename.startswith(("/", "\\")) or info.flag_bits & 1:
                    raise ValueError(
                        "Зашифрованные или некорректные архивы Office не поддерживаются."
                    )
                if info.file_size > max(info.compress_size, 1) * 250:
                    raise ValueError(
                        "Слишком высокая степень сжатия документа. Разделите его на части."
                    )
                name = info.filename.lower()
                if "vbaproject" in name or "/embeddings/" in name:
                    raise ValueError(
                        "Документы с макросами или вложенными исполняемыми объектами не поддерживаются."
                    )
                if name.endswith((".xml", ".rels")):
                    # DefusedXML rejects entity expansion/DTDs. This also validates all parts
                    # before the Office libraries parse them; no relationship is dereferenced.
                    ElementTree.fromstring(archive.read(info), forbid_dtd=True)
    except DefusedXmlException as exc:
        raise ValueError(
            "Документ Office содержит запрещённые DTD или внешние сущности XML."
        ) from exc
    except BadZipFile as exc:
        raise ValueError("Файл Office повреждён или не является DOCX/XLSX.") from exc


def _checked_image(source) -> Image.Image:
    with python_warnings.catch_warnings():
        python_warnings.simplefilter("error", Image.DecompressionBombWarning)
        with Image.open(source) as image:
            width, height = image.size
            if width <= 0 or height <= 0 or width * height > MAX_IMAGE_PIXELS:
                raise ValueError("Изображение превышает ограничение 20 миллионов пикселей.")
            if image.format not in {"PNG", "JPEG", "TIFF", "BMP", "WEBP"}:
                raise ValueError("Этот формат встроенного изображения не поддерживается для OCR.")
            if getattr(image, "n_frames", 1) != 1:
                raise ValueError(
                    "Многостраничные изображения нужно предварительно преобразовать в PDF."
                )
            image.load()
            return ImageOps.exif_transpose(image).convert("RGB")


def _ocr(image: Image.Image) -> tuple[str, list[str]]:
    """Run only the local executable, with no shell and a hard process timeout."""
    binary = shutil.which("tesseract")
    if not binary:
        return "", ["OCR недоступен: установите Tesseract с языками rus и eng."]
    notices: list[str] = []
    try:
        languages = subprocess.run(
            [binary, "--list-langs"], capture_output=True, text=True, timeout=5, check=True
        ).stdout.splitlines()
        selected = [lang for lang in ("rus", "eng") if lang in languages]
        if not selected:
            return "", ["OCR недоступен: отсутствуют языки Tesseract rus и eng."]
        if len(selected) < 2:
            notices.append(
                "Для OCR установлен только язык "
                + selected[0]
                + "; другой язык может распознаваться неверно."
            )
        with tempfile.TemporaryDirectory(prefix="analyst-ocr-") as directory:
            input_path = Path(directory) / "page.png"
            output_base = Path(directory) / "recognized"
            image.save(input_path, format="PNG")
            subprocess.run(
                [binary, str(input_path), str(output_base), "-l", "+".join(selected), "--psm", "3"],
                capture_output=True,
                timeout=OCR_TIMEOUT_SECONDS,
                check=True,
            )
            output_path = output_base.with_suffix(".txt")
            if output_path.stat().st_size > MAX_TEXT_CHARS:
                raise ValueError("Результат OCR слишком велик.")
            text = output_path.read_text(encoding="utf-8").strip()
        return text, notices
    except subprocess.TimeoutExpired:
        return "", [
            f"OCR остановлен по тайм-ауту {OCR_TIMEOUT_SECONDS} с; текст изображения не извлечён."
        ]
    except (OSError, subprocess.CalledProcessError):
        return "", ["Локальный OCR завершился с ошибкой; текст изображения не извлечён."]


def _image_ocr(image: Image.Image, result: _Result, locator: dict, native: str = "") -> None:
    try:
        text, notices = _ocr(image)
        for notice in notices:
            result.warn(notice)
        if native:
            text = _additional_ocr_text(native, text)
        if text:
            result.add(
                text,
                {
                    **locator,
                    "region": "whole_image",
                    "image_width": image.width,
                    "image_height": image.height,
                },
                "ocr",
            )
            result.warn(
                "OCR может ошибаться в цифрах и структуре таблиц; сверяйте распознанный текст с оригиналом."
            )
        elif not native:
            result.warn("На изображении или отсканированной странице текст не распознан.")
    finally:
        image.close()


def _image_file(path: Path, result: _Result) -> None:
    _image_ocr(_checked_image(path), result, {"type": "image"})


def _additional_ocr_text(native: str, recognized: str) -> str:
    """Avoid duplicating a native text layer when OCR covers the whole page."""
    from difflib import SequenceMatcher

    normalized = " ".join(native.casefold().split())
    native_lines = [
        " ".join(line.casefold().split()) for line in native.splitlines() if line.strip()
    ]
    additional = []
    for line in recognized.splitlines():
        key = " ".join(line.casefold().split())
        if not key or key in normalized:
            continue
        if any(SequenceMatcher(None, key, existing).ratio() >= 0.90 for existing in native_lines):
            continue
        additional.append(line)
    return "\n".join(additional)


def _has_raster(page) -> bool:
    """Inspect resources without decoding embedded image payloads."""
    visited: set[int] = set()

    def inspect(resources, depth: int = 0) -> bool:
        if not resources or depth > 8:
            return False
        resources = resources.get_object()
        objects = resources.get("/XObject", {})
        objects = objects.get_object() if hasattr(objects, "get_object") else objects
        for value in objects.values():
            obj = value.get_object()
            if id(obj) in visited:
                continue
            visited.add(id(obj))
            if obj.get("/Subtype") == "/Image":
                return True
            if obj.get("/Subtype") == "/Form" and inspect(obj.get("/Resources"), depth + 1):
                return True
        return False

    return inspect(page.get("/Resources"))


def _render_scale(width: float, height: float) -> float:
    area = width * height
    if not math.isfinite(area) or area <= 0 or width <= 0 or height <= 0:
        raise ValueError("Некорректный размер страницы PDF.")
    scale = min(
        2.5,
        math.sqrt(MAX_IMAGE_PIXELS / area) * 0.99,
        MAX_RENDER_EDGE / max(width, height),
    )
    # PDFium rounds dimensions up. An area-only cap can still allocate billions
    # of pixels for a maliciously thin page once its smaller edge rounds to 1.
    if math.ceil(width * scale) * math.ceil(height * scale) > MAX_IMAGE_PIXELS:
        raise ValueError("Размер страницы PDF превышает ограничение растеризации.")
    return scale


def _render_page(path: Path, index: int) -> Image.Image:
    import pypdfium2 as pdfium

    # PDFium is not thread safe, even across distinct documents.
    with _PDFIUM_LOCK, pdfium.PdfDocument(str(path)) as document:
        page = document[index]
        try:
            width, height = page.get_size()
            bitmap = page.render(scale=_render_scale(width, height))
            try:
                return bitmap.to_pil().convert("RGB").copy()
            finally:
                bitmap.close()
        finally:
            page.close()


def _pdf(path: Path, result: _Result) -> None:
    from pypdf import PdfReader

    with path.open("rb") as stream:
        reader = PdfReader(stream)
        if reader.is_encrypted:
            raise ValueError("PDF защищён паролем. Загрузите незашифрованную копию.")
        if len(reader.pages) > MAX_PAGES:
            raise ValueError(f"PDF содержит более {MAX_PAGES} страниц. Разделите его на части.")
        for number, page in enumerate(reader.pages, 1):
            locator = {"type": "pdf", "page": number}
            try:
                native = (page.extract_text() or "").strip()
            except Exception:
                native = ""
                result.warn(
                    f"Страница {number}: текстовый слой PDF не прочитан, предпринята попытка OCR."
                )
            if (
                native
                and not any(char.isalnum() for char in native)
                and any(char == "\ufffd" or unicodedata.category(char) == "Co" for char in native)
            ):
                # Some embedded fonts render normally but map every glyph to a placeholder.
                native = ""
                result.warn(f"Страница {number}: повреждена кодировка текстового слоя PDF.")
            result.add(native, locator)
            if _has_raster(page) or (not native and page.get("/Contents")):
                try:
                    _image_ocr(_render_page(path, number - 1), result, locator, native)
                except Exception as exc:
                    logger.info(
                        "document.page.ocr_failed",
                        extra={"page": number, "error_type": type(exc).__name__},
                    )
                    result.warn(
                        f"Страница {number}: OCR недоступен или не смог обработать страницу."
                    )


def _table(
    result: _Result,
    rows: list[list[str]],
    locator: dict,
    row_numbers: list[int] | None = None,
    **metadata,
) -> None:
    if not rows:
        return
    if len(rows) > MAX_ROWS or max(map(len, rows)) > MAX_COLUMNS or sum(map(len, rows)) > MAX_CELLS:
        raise ValueError("Таблица слишком велика. Разделите её на части.")
    header = " | ".join(rows[0])
    numbers = row_numbers or list(range(1, len(rows) + 1))
    for start in range(0, len(rows), TABLE_GROUP_ROWS):
        group = rows[start : start + TABLE_GROUP_ROWS]
        group_numbers = numbers[start : start + TABLE_GROUP_ROWS]
        text_rows = ([rows[0]] if start else []) + group
        location = {**locator, "row_start": group_numbers[0], "row_end": group_numbers[-1]}
        if location["type"] == "xlsx":
            from openpyxl.utils import get_column_letter

            location["cell_range"] = (
                f"A{group_numbers[0]}:{get_column_letter(max(map(len, rows)))}{group_numbers[-1]}"
            )
        block_metadata = dict(metadata)
        if location["type"] == "xlsx":
            from openpyxl.utils.cell import coordinate_from_string

            for key in ("formulas", "cell_formats"):
                block_metadata[key] = {
                    coordinate: value
                    for coordinate, value in block_metadata.get(key, {}).items()
                    if coordinate_from_string(coordinate)[1] in group_numbers
                }
        result.add(
            "\n".join(" | ".join(row) for row in text_rows),
            location,
            table_header=header,
            table_header_inferred=True,
            table_rows=group,
            header_row=numbers[0],
            **block_metadata,
        )


def _office_images(path: Path, result: _Result, prefix: str) -> None:
    with ZipFile(path) as archive:
        for name in archive.namelist():
            if not name.startswith(prefix) or name.endswith("/"):
                continue
            try:
                image = _checked_image(io.BytesIO(archive.read(name)))
                _image_ocr(image, result, {"type": "office_image", "part": name})
            except (
                ValueError,
                OSError,
                Image.DecompressionBombError,
                Image.DecompressionBombWarning,
            ):
                result.warn(f"Встроенное изображение {PurePosixPath(name).name} не обработано OCR.")


def _docx(path: Path, result: _Result) -> None:
    from docx import Document
    from docx.table import Table
    from docx.text.paragraph import Paragraph

    with path.open("rb") as stream:
        document = Document(stream)
        paragraph_index = table_index = 0
        for item in document.iter_inner_content():
            if isinstance(item, Paragraph):
                paragraph_index += 1
                result.add(
                    item.text,
                    {
                        "type": "docx",
                        "paragraph": paragraph_index,
                        "style": item.style.name if item.style else "",
                    },
                )
            elif isinstance(item, Table):
                table_index += 1
                if (
                    len(item.rows) > MAX_ROWS
                    or len(item.columns) > MAX_COLUMNS
                    or len(item.rows) * len(item.columns) > MAX_CELLS
                ):
                    raise ValueError("Таблица DOCX слишком велика. Разделите её на части.")
                rows = [[cell.text.strip() for cell in row.cells] for row in item.rows]
                _table(result, rows, {"type": "docx", "table": table_index})
                if any(cell.tables for row in item.rows for cell in row.cells):
                    result.warn(
                        "Вложенные таблицы DOCX требуют ручной проверки: извлечён только текст внешних ячеек."
                    )
        for section_index, section in enumerate(document.sections, 1):
            for name, part in (("header", section.header), ("footer", section.footer)):
                for index, paragraph in enumerate(part.paragraphs, 1):
                    result.add(
                        paragraph.text,
                        {
                            "type": "docx",
                            "section": section_index,
                            "part": name,
                            "paragraph": index,
                        },
                    )
    _office_images(path, result, "word/media/")


def _cell_text(value) -> str:
    return "" if value is None else str(value).strip()


def _xlsx_value(value, number_format: str) -> str:
    """Handle common percent displays; retain other custom numeric formats explicitly."""
    text = _cell_text(value)
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        simple_percent = {"0%": 0, "0.0%": 1, "0.00%": 2, "0.000%": 3}
        if number_format in simple_percent:
            return f"{value * 100:.{simple_percent[number_format]}f}%"
        if number_format not in {"General", "0", "0.00", "#,##0", "#,##0.00"}:
            # Do not pretend to implement Excel's custom format language (units,
            # currency, conditional sections). Keep its meaning visible to retrieval.
            return f"{text} [формат Excel: {number_format}]"
    return text


def _xlsx(path: Path, result: _Result) -> None:
    from openpyxl import load_workbook

    # File handles allow immutable storage paths without the original extension.
    with path.open("rb") as source, path.open("rb") as cached_source:
        workbook = load_workbook(source, read_only=True, data_only=False, keep_links=False)
        cached = load_workbook(cached_source, read_only=True, data_only=True, keep_links=False)
        try:
            total_cells = 0
            for sheet in workbook:
                if (sheet.max_row or 0) > MAX_ROWS or (sheet.max_column or 0) > MAX_COLUMNS:
                    raise ValueError(
                        "Лист Excel слишком велик. Разделите его на части или уберите пустое форматирование."
                    )
                # Some spreadsheet producers write stale dimensions (e.g. A1:A1)
                # although more rows exist. Stream actual rows, checking our own
                # limits, instead of silently dropping cells beyond that hint.
                sheet.reset_dimensions()
                cached[sheet.title].reset_dimensions()
                rows: list[list[str]] = []
                row_numbers: list[int] = []
                formulas = {}
                cell_formats = {}
                for number, (row, cached_row) in enumerate(
                    zip(sheet.iter_rows(), cached[sheet.title].iter_rows(), strict=True), 1
                ):
                    if number > MAX_ROWS or len(row) > MAX_COLUMNS:
                        raise ValueError("Лист Excel слишком велик. Разделите его на части.")
                    total_cells += len(row)
                    if total_cells > MAX_CELLS:
                        raise ValueError("Книга Excel превышает ограничение 200000 ячеек.")
                    values = []
                    for cell, cached_cell in zip(row, cached_row, strict=True):
                        number_format = cell.number_format or "General"
                        if cell.value is not None and number_format != "General":
                            cell_formats[cell.coordinate] = number_format
                        if cell.data_type == "f":
                            formula = _cell_text(cell.value)
                            formulas[cell.coordinate] = formula
                            if cached_cell.value is None:
                                result.warn(
                                    f"Лист «{sheet.title}»: у формул нет сохранённого результата; вычисления не выполнялись."
                                )
                                values.append(f"[формула {cell.coordinate}; результат не сохранён]")
                            else:
                                values.append(_xlsx_value(cached_cell.value, number_format))
                        else:
                            values.append(_xlsx_value(cell.value, number_format))
                    if any(values):
                        rows.append(values)
                        row_numbers.append(number)
                _table(
                    result,
                    rows,
                    {"type": "xlsx", "sheet": sheet.title},
                    row_numbers,
                    formulas=formulas,
                    cell_formats=cell_formats,
                )
                if sheet.sheet_state != "visible" and rows:
                    result.warn(f"Извлечён также скрытый лист «{sheet.title}».")
        finally:
            workbook.close()
            cached.close()
    _office_images(path, result, "xl/media/")


def _csv(path: Path, result: _Result) -> None:
    raw = path.read_bytes()
    try:
        text = raw.decode("utf-8-sig")
    except UnicodeDecodeError as exc:
        if raw.startswith((b"\xff\xfe", b"\xfe\xff")):
            text = raw.decode("utf-16")
        else:
            raise ValueError("Сохраните CSV в кодировке UTF-8 (или UTF-16 с BOM).") from exc
    if "\x00" in text:
        raise ValueError("CSV содержит бинарные данные или неподдерживаемую кодировку.")
    sample = text[:8192]
    try:
        dialect = csv.Sniffer().sniff(sample, delimiters=",;\t|")
        reader = csv.reader(io.StringIO(text, newline=""), dialect, strict=True)
    except csv.Error:
        # Single-column CSV is valid; ambiguous multi-column input is not guessed.
        if any(delimiter in sample for delimiter in (",", ";", "\t", "|")):
            raise ValueError(
                "Не удалось определить разделитель CSV. Проверьте структуру и кавычки."
            ) from None
        reader = csv.reader(io.StringIO(text, newline=""), strict=True)
    rows, numbers = [], []
    width = None
    cells = 0
    try:
        for number, row in enumerate(reader, 1):
            if not row or not any(cell.strip() for cell in row):
                continue
            if width is None:
                width = len(row)
            if len(row) != width:
                raise ValueError(f"CSV: число столбцов в записи {number} отличается от заголовка.")
            cells += len(row)
            if number > MAX_ROWS or len(row) > MAX_COLUMNS or cells > MAX_CELLS:
                raise ValueError("CSV слишком велик. Разделите его на части.")
            rows.append([cell.strip() for cell in row])
            numbers.append(number)
    except csv.Error as exc:
        raise ValueError("CSV повреждён: проверьте кавычки, разделители и длину ячеек.") from exc
    _table(result, rows, {"type": "csv", "row_numbering": "record"}, numbers)
