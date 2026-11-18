"""Real parser fixtures plus deterministic OCR failure/partial-result coverage."""

import math
import shutil
import subprocess
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

import pytest
from PIL import Image, ImageDraw, ImageFont
from pypdf import PdfWriter
from pypdf.generic import DecodedStreamObject, DictionaryObject, NameObject

from src.knowledge import extractors


def text_pdf(path: Path, text: str = "Brent oil benchmark. Sulfur content and density.") -> Path:
    writer = PdfWriter()
    page = writer.add_blank_page(width=595, height=842)
    font = DictionaryObject(
        {
            NameObject("/Type"): NameObject("/Font"),
            NameObject("/Subtype"): NameObject("/Type1"),
            NameObject("/BaseFont"): NameObject("/Helvetica"),
        }
    )
    page[NameObject("/Resources")] = DictionaryObject(
        {NameObject("/Font"): DictionaryObject({NameObject("/F1"): writer._add_object(font)})}
    )
    stream = DecodedStreamObject()
    stream.set_data(f"BT /F1 18 Tf 40 760 Td ({text}) Tj ET".encode("ascii"))
    page[NameObject("/Contents")] = writer._add_object(stream)
    writer.write(path)
    return path


def image_fixture(path: Path, *, pdf: bool = False) -> Path:
    image = Image.new("RGB", (1000, 250), "white")
    draw = ImageDraw.Draw(image)
    font = ImageFont.truetype("DejaVuSans.ttf", 40)
    draw.text((30, 70), "Brent oil price: 80 USD", font=font, fill="black")
    image.save(path, format="PDF" if pdf else "PNG")
    image.close()
    return path


def rewrite_archive(path: Path, transform) -> None:
    with ZipFile(path) as source:
        entries = [(info.filename, source.read(info)) for info in source.infolist()]
    with ZipFile(path, "w", ZIP_DEFLATED) as destination:
        for name, content in entries:
            destination.writestr(name, transform(name, content))


def test_native_pdf_preserves_actual_page_and_stable_identity(tmp_path, monkeypatch):
    path = text_pdf(tmp_path / "object")
    monkeypatch.setattr(extractors, "_ocr", lambda image: pytest.fail("Text PDF needs no OCR"))
    first = extractors.extract(path, "report.PDF")
    second = extractors.extract(path, "renamed.pdf")
    assert first == second
    assert first["warnings"] == []
    assert first["blocks"][0]["page"] == 1
    assert first["blocks"][0]["locator"] == {"type": "pdf", "page": 1}
    assert first["blocks"][0]["method"] == "native"
    assert "Sulfur" in first["blocks"][0]["text"]


def test_native_cyrillic_report_keeps_original_text_and_page(monkeypatch):
    path = Path(__file__).resolve().parents[1] / "data/reports/CBR_2025-10-24.pdf"
    monkeypatch.setattr(extractors, "_ocr", lambda image: ("", []))
    result = extractors.extract(path, path.name)
    assert any(
        block["method"] == "native"
        and block["page"] == 1
        and "Среднесрочный прогноз Банка России" in " ".join(block["text"].split())
        for block in result["blocks"]
    )


@pytest.mark.parametrize("placeholder", ["FFFD", "E000"])
def test_broken_font_mapping_uses_ocr_without_a_raster_image(tmp_path, monkeypatch, placeholder):
    writer = PdfWriter(clone_from=text_pdf(tmp_path / "clean.pdf", "Brent"))
    font = writer.pages[0]["/Resources"]["/Font"]["/F1"]
    mappings = "\n".join(f"<{byte:02X}> <{placeholder}>" for byte in b"Brent")
    cmap = DecodedStreamObject()
    cmap.set_data(
        (
            "1 begincodespacerange\n<00> <FF>\nendcodespacerange\n"
            f"5 beginbfchar\n{mappings}\nendbfchar"
        ).encode("ascii")
    )
    font[NameObject("/ToUnicode")] = writer._add_object(cmap)
    path = tmp_path / "broken.pdf"
    writer.write(path)
    monkeypatch.setattr(extractors, "_ocr", lambda image: ("Brent", []))
    result = extractors.extract(path, path.name)
    assert [(b["method"], b["text"], b["page"]) for b in result["blocks"]] == [("ocr", "Brent", 1)]
    assert any("кодировка" in warning for warning in result["warnings"])


def test_scanned_pdf_renders_and_uses_local_ocr(tmp_path, monkeypatch):
    path = image_fixture(tmp_path / "scan", pdf=True)
    seen = []

    def fake_ocr(image):
        seen.append(image.size)
        return "Brent oil price: 80 USD", []

    monkeypatch.setattr(extractors, "_ocr", fake_ocr)
    result = extractors.extract(path, "scan.pdf")
    assert len(seen) == 1
    assert result["blocks"][0]["method"] == "ocr"
    assert result["blocks"][0]["page"] == 1
    assert result["blocks"][0]["locator"]["region"] == "whole_image"
    assert result["warnings"]  # OCR accuracy is not presented as verified table extraction.


def test_mixed_page_deduplicates_native_text(tmp_path, monkeypatch):
    path = text_pdf(tmp_path / "mixed", "Oil report title")
    monkeypatch.setattr(extractors, "_has_raster", lambda page: True)
    monkeypatch.setattr(
        extractors, "_ocr", lambda image: ("Oil report title\nSulfur: 1.5 percent", [])
    )
    result = extractors.extract(path, "mixed.pdf")
    assert [block["method"] for block in result["blocks"]] == ["native", "ocr"]
    assert result["blocks"][1]["text"] == "Sulfur: 1.5 percent"


def test_partial_pdf_keeps_native_text_when_ocr_unavailable(tmp_path, monkeypatch):
    path = text_pdf(tmp_path / "mixed")
    monkeypatch.setattr(extractors, "_has_raster", lambda page: True)
    monkeypatch.setattr(extractors.shutil, "which", lambda name: None)
    result = extractors.extract(path, "mixed.pdf")
    assert result["blocks"][0]["method"] == "native"
    assert any("OCR недоступен" in warning for warning in result["warnings"])


def test_scan_without_ocr_is_error_not_ready(tmp_path, monkeypatch):
    path = image_fixture(tmp_path / "scan", pdf=True)
    monkeypatch.setattr(extractors.shutil, "which", lambda name: None)
    with pytest.raises(ValueError, match="Не удалось извлечь текст.*OCR недоступен"):
        extractors.extract(path, "scan.pdf")


def test_encrypted_pdf_rejected(tmp_path):
    path = tmp_path / "encrypted"
    writer = PdfWriter()
    writer.add_blank_page(100, 100)
    writer.encrypt("secret")
    writer.write(path)
    with pytest.raises(ValueError, match="паролем"):
        extractors.extract(path, "encrypted.pdf")


def test_pdf_page_limit(tmp_path, monkeypatch):
    path = tmp_path / "pages"
    writer = PdfWriter()
    writer.add_blank_page(100, 100)
    writer.add_blank_page(100, 100)
    writer.write(path)
    monkeypatch.setattr(extractors, "MAX_PAGES", 1)
    with pytest.raises(ValueError, match="более 1 страниц"):
        extractors.extract(path, "many.pdf")


@pytest.mark.parametrize("dimensions", [(1e12, 1), (1, 1e12), (595, 842), (1e6, 1e6)])
def test_pdf_render_limit_accounts_for_pixel_rounding(dimensions):
    width, height = dimensions
    scale = extractors._render_scale(width, height)
    pixels = math.ceil(width * scale) * math.ceil(height * scale)
    assert pixels <= extractors.MAX_IMAGE_PIXELS
    assert max(math.ceil(width * scale), math.ceil(height * scale)) <= extractors.MAX_RENDER_EDGE


def test_image_ocr_has_pixel_region_without_invented_page(tmp_path, monkeypatch):
    path = image_fixture(tmp_path / "image")
    monkeypatch.setattr(extractors, "_ocr", lambda image: ("Нефть Urals: сера, %", []))
    block = extractors.extract(path, "image.png")["blocks"][0]
    assert block["page"] is None
    assert block["locator"]["image_width"] == 1000
    assert "bbox" not in block["locator"]


def test_oversized_image_rejected_before_ocr(tmp_path, monkeypatch):
    path = image_fixture(tmp_path / "image")
    monkeypatch.setattr(extractors, "MAX_IMAGE_PIXELS", 100)
    monkeypatch.setattr(extractors, "_ocr", lambda image: pytest.fail("Must validate first"))
    with pytest.raises(ValueError, match="пикселей"):
        extractors.extract(path, "image.png")


def test_ocr_timeout_is_bounded(monkeypatch):
    monkeypatch.setattr(extractors.shutil, "which", lambda name: "/usr/bin/tesseract")
    calls = []

    def run(command, **kwargs):
        calls.append((command, kwargs))
        if "--list-langs" in command:
            return subprocess.CompletedProcess(command, 0, stdout="rus\neng\n")
        raise subprocess.TimeoutExpired(command, kwargs["timeout"])

    monkeypatch.setattr(extractors.subprocess, "run", run)
    with Image.new("RGB", (100, 100)) as image:
        text, warnings = extractors._ocr(image)
    assert text == ""
    assert "тайм-ауту" in warnings[0]
    assert calls[-1][1]["timeout"] == extractors.OCR_TIMEOUT_SECONDS
    assert "shell" not in calls[-1][1]


def test_docx_paragraphs_and_tables_have_real_locators(tmp_path):
    from docx import Document

    path = tmp_path / "object"
    document = Document()
    document.add_heading("Сорта нефти", 1)
    document.add_paragraph("Плотность и содержание серы")
    table = document.add_table(rows=3, cols=3)
    for cells, values in zip(
        table.rows,
        [["Сорт", "Плотность, API", "Сера, %"], ["Urals", "31", "1.5"], ["ESPO", "35", "0.6"]],
        strict=True,
    ):
        for cell, value in zip(cells.cells, values, strict=True):
            cell.text = value
    document.save(path)
    result = extractors.extract(path, "report.docx")
    assert all(block["page"] is None for block in result["blocks"])
    assert result["blocks"][0]["locator"]["paragraph"] == 1
    block = result["blocks"][2]
    assert block["locator"]["table"] == 1
    assert block["table_rows"][2] == ["ESPO", "35", "0.6"]
    assert "Сера, %" in block["table_header"]


def test_docx_embedded_image_uses_ocr(tmp_path, monkeypatch):
    from docx import Document

    path = tmp_path / "document"
    picture = image_fixture(tmp_path / "picture.png")
    document = Document()
    document.add_picture(str(picture))
    document.save(path)
    monkeypatch.setattr(extractors, "_ocr", lambda image: ("Таблица со скана", []))
    block = extractors.extract(path, "scan.docx")["blocks"][0]
    assert block["locator"]["part"].startswith("word/media/")
    assert block["method"] == "ocr"


def test_xlsx_preserves_headers_formulas_and_cell_ranges(tmp_path):
    from openpyxl import Workbook

    path = tmp_path / "object"
    workbook = Workbook()
    sheet = workbook.active
    sheet.title = "Нефть"
    sheet.append(["Сорт", "Цена USD/барр.", "Сценарий"])
    sheet.append(["Brent", 80, "=B2+5"])
    workbook.save(path)
    result = extractors.extract(path, "prices.xlsx")
    block = result["blocks"][0]
    assert block["page"] is None
    assert block["locator"]["sheet"] == "Нефть"
    assert block["locator"]["cell_range"] == "A1:C2"
    assert block["formulas"] == {"C2": "=B2+5"}
    assert "результат не сохранён" in block["text"]
    assert any("вычисления не выполнялись" in warning for warning in result["warnings"])
    assert "USD/барр." in block["table_header"]


def test_xlsx_uses_cached_formula_value_without_evaluation(tmp_path):
    from openpyxl import Workbook

    path = tmp_path / "object"
    workbook = Workbook()
    workbook.active.append(["Price", "Scenario"])
    workbook.active.append([80, "=A2+5"])
    workbook.save(path)
    rewrite_archive(
        path,
        lambda name, data: (
            data.replace(b"<f>A2+5</f><v></v>", b"<f>A2+5</f><v>85</v>")
            if name == "xl/worksheets/sheet1.xml"
            else data
        ),
    )
    result = extractors.extract(path, "cached.xlsx")
    assert result["blocks"][0]["table_rows"][1] == ["80", "85"]
    assert result["warnings"] == []


def test_xlsx_percentage_keeps_displayed_units(tmp_path):
    from openpyxl import Workbook

    path = tmp_path / "object"
    workbook = Workbook()
    sheet = workbook.active
    sheet.append(["Crude", "Sulfur"])
    sheet.append(["Urals", 0.015])
    sheet["B2"].number_format = "0.00%"
    workbook.save(path)
    block = extractors.extract(path, "sulfur.xlsx")["blocks"][0]
    assert block["table_rows"][1] == ["Urals", "1.50%"]
    assert block["cell_formats"] == {"B2": "0.00%"}


def test_write_only_xlsx_without_declared_dimensions(tmp_path):
    from openpyxl import Workbook

    path = tmp_path / "object"
    workbook = Workbook(write_only=True)
    sheet = workbook.create_sheet()
    sheet.append(["Crude", "Price"])
    sheet.append(["Brent", 80])
    workbook.save(path)
    assert "Brent" in extractors.extract(path, "streamed.xlsx")["blocks"][0]["text"]


def test_xlsx_stale_dimension_does_not_silently_drop_rows(tmp_path):
    from openpyxl import Workbook

    path = tmp_path / "object"
    workbook = Workbook()
    workbook.active.append(["Crude", "Price"])
    workbook.active.append(["Brent", 80])
    workbook.save(path)
    rewrite_archive(
        path,
        lambda name, data: (
            data.replace(b'<dimension ref="A1:B2"', b'<dimension ref="A1:A1"')
            if name == "xl/worksheets/sheet1.xml"
            else data
        ),
    )
    block = extractors.extract(path, "stale-dimensions.xlsx")["blocks"][0]
    assert block["table_rows"][1] == ["Brent", "80"]
    assert block["locator"]["cell_range"] == "A1:B2"


def test_office_hyperlink_is_not_fetched(tmp_path, monkeypatch):
    import socket

    from openpyxl import Workbook

    path = tmp_path / "object"
    workbook = Workbook()
    workbook.active["A1"] = "Report source"
    workbook.active["A1"].hyperlink = "https://untrusted.invalid/private"
    workbook.save(path)
    monkeypatch.setattr(
        socket, "create_connection", lambda *args, **kwargs: pytest.fail("No network")
    )
    assert "Report source" in extractors.extract(path, "links.xlsx")["blocks"][0]["text"]


def test_table_groups_repeat_headers_and_preserve_source_rows(tmp_path, monkeypatch):
    path = tmp_path / "object"
    path.write_text("Name;Price USD\nBrent;80\nWTI;75\nUrals;70\n", encoding="utf-8")
    monkeypatch.setattr(extractors, "TABLE_GROUP_ROWS", 2)
    result = extractors.extract(path, "prices.csv")
    assert len(result["blocks"]) == 2
    assert result["blocks"][1]["text"].startswith("Name | Price USD\n")
    assert result["blocks"][1]["locator"]["row_start"] == 3
    assert result["blocks"][1]["header_row"] == 1


def test_csv_quoted_delimiter_and_utf16(tmp_path):
    path = tmp_path / "object"
    path.write_text('"Название";"Цена, USD"\n"Urals; Россия";70\n', encoding="utf-16")
    result = extractors.extract(path, "prices.csv")
    assert result["blocks"][0]["table_rows"][1] == ["Urals; Россия", "70"]


@pytest.mark.parametrize(
    "content", [b"name,price\nBrent,80,extra\n", b"\x00\x01", "нефть".encode("cp1251")]
)
def test_invalid_csv_rejected(tmp_path, content):
    path = tmp_path / "object"
    path.write_bytes(content)
    with pytest.raises(ValueError):
        extractors.extract(path, "invalid.csv")


def test_csv_limit(tmp_path, monkeypatch):
    path = tmp_path / "object"
    path.write_text("name,price\nBrent,80\nWTI,75\n")
    monkeypatch.setattr(extractors, "MAX_ROWS", 2)
    with pytest.raises(ValueError, match="слишком велик"):
        extractors.extract(path, "prices.csv")


@pytest.mark.parametrize("name", ["xl/vbaProject.bin", "xl/embeddings/oleObject1.bin"])
def test_macros_and_embedded_objects_rejected(tmp_path, name):
    from openpyxl import Workbook

    path = tmp_path / "object"
    Workbook().save(path)
    with ZipFile(path, "a") as archive:
        archive.writestr(name, b"untrusted")
    with pytest.raises(ValueError, match="макросами"):
        extractors.extract(path, "untrusted.xlsx")


def test_archive_expansion_limit(tmp_path):
    from openpyxl import Workbook

    path = tmp_path / "object"
    Workbook().save(path)
    with ZipFile(path, "a", ZIP_DEFLATED) as archive:
        archive.writestr("payload.bin", b"a" * 1_000_000)
    with pytest.raises(ValueError, match="степень сжатия"):
        extractors.extract(path, "bomb.xlsx")


def test_xml_entities_rejected(tmp_path):
    from openpyxl import Workbook

    path = tmp_path / "object"
    Workbook().save(path)
    with ZipFile(path, "a") as archive:
        archive.writestr(
            "evil.xml",
            b'<!DOCTYPE x [<!ENTITY payload SYSTEM "file:///etc/passwd">]><x>&payload;</x>',
        )
    with pytest.raises(ValueError, match="DTD"):
        extractors.extract(path, "malicious.xlsx")


@pytest.mark.parametrize(
    "filename", ["corrupt.pdf", "corrupt.png", "corrupt.xlsx", "corrupt.docx", "unsupported.exe"]
)
def test_corrupt_or_unsupported_input_fails_safely(tmp_path, filename):
    path = tmp_path / "object"
    path.write_bytes(b"not a document")
    with pytest.raises(ValueError):
        extractors.extract(path, filename)


def test_file_size_limit(tmp_path, monkeypatch):
    path = tmp_path / "object"
    path.write_text("name,price\nBrent,80")
    monkeypatch.setattr(extractors, "MAX_FILE_BYTES", 5)
    with pytest.raises(ValueError, match="100 МБ"):
        extractors.extract(path, "prices.csv")


@pytest.mark.skipif(shutil.which("tesseract") is None, reason="Requires local Tesseract binary")
@pytest.mark.parametrize("pdf", [False, True])
def test_real_tesseract_generated_image_and_scan(tmp_path, pdf):
    languages = subprocess.run(
        ["tesseract", "--list-langs"], capture_output=True, text=True, check=True, timeout=5
    ).stdout.splitlines()
    if "eng" not in languages:
        pytest.skip("English Tesseract data not installed")
    path = image_fixture(tmp_path / "real-scan", pdf=pdf)
    result = extractors.extract(path, "real.pdf" if pdf else "real.png")
    text = " ".join(block["text"] for block in result["blocks"])
    assert "Brent" in text
    assert "80" in text
    assert all(block["method"] == "ocr" for block in result["blocks"])


@pytest.mark.parametrize("extension", ["png", "pdf"])
def test_real_tesseract_preserves_russian_text_and_number(tmp_path, extension):
    """A Cyrillic OCR smoke regression, not an estimate of OCR dataset quality."""
    binary = shutil.which("tesseract")
    if binary is None:
        pytest.skip("Russian Tesseract OCR unavailable: local binary not installed")
    languages = subprocess.run(
        [binary, "--list-langs"], capture_output=True, text=True, check=True, timeout=5
    ).stdout.splitlines()
    if "rus" not in languages:
        pytest.skip("Russian Tesseract data not installed")
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", 44)
    except OSError:
        pytest.skip("DejaVuSans Cyrillic font not available")
    path = tmp_path / f"russian-scan.{extension}"
    with Image.new("RGB", (1200, 250), "white") as image:
        ImageDraw.Draw(image).text((30, 75), "Содержание серы 1.5%", font=font, fill="black")
        image.save(path)
    result = extractors.extract(path, path.name)
    text = " ".join(block["text"] for block in result["blocks"]).casefold()
    assert "содержание серы" in text
    assert "1.5%" in text.replace(" ", "")
    assert all(block["method"] == "ocr" for block in result["blocks"])
    if extension == "pdf":
        assert all(block["page"] == 1 for block in result["blocks"])
