"""Framework-free tabular input normalization."""
from __future__ import annotations

import csv
from io import BytesIO, StringIO
from pathlib import Path

import pandas as pd


class DataParserError(ValueError):
    """Raised when tabular input cannot be normalized."""


def parse_csv_buffer(buffer: bytes | bytearray | BytesIO, filename: str = "") -> pd.DataFrame:
    """Parse a CSV upload buffer into a DataFrame."""
    if isinstance(buffer, BytesIO):
        payload = buffer.getvalue()
    else:
        payload = bytes(buffer)
    try:
        text = payload.decode("utf-8-sig")
    except UnicodeDecodeError as exc:
        raise DataParserError("CSV files must be UTF-8 encoded") from exc
    if filename and Path(filename).suffix.lower() != ".csv":
        raise DataParserError("Only .csv uploads are supported")
    return parse_tabular_text(text)


def parse_excel_buffer(buffer: bytes | bytearray | BytesIO, filename: str = "") -> pd.DataFrame:
    """Read the first worksheet from an XLSX or XLS upload."""
    payload = buffer.getvalue() if isinstance(buffer, BytesIO) else bytes(buffer)
    suffix = Path(filename).suffix.lower()
    if suffix not in {".xlsx", ".xls"}:
        raise DataParserError("Only .xlsx and .xls uploads are supported")
    try:
        frame = pd.read_excel(BytesIO(payload), sheet_name=0)
    except (ImportError, ValueError, OSError) as exc:
        raise DataParserError(f"Could not read Excel file: {exc}") from exc
    return _clean_frame(frame, "Excel worksheet")


def parse_pdf(buffer: bytes | bytearray | BytesIO) -> pd.DataFrame:
    """Extract tabular text from a PDF, falling back to OCR for scanned pages."""
    payload = buffer.getvalue() if isinstance(buffer, BytesIO) else bytes(buffer)
    try:
        from pypdf import PdfReader

        text = "\n".join(page.extract_text() or "" for page in PdfReader(BytesIO(payload)).pages)
    except Exception as exc:
        raise DataParserError(f"Could not read PDF file: {exc}") from exc
    if text.strip():
        return parse_tabular_text(text)
    try:
        import fitz
        from PIL import Image
        import pytesseract

        pages = []
        document = fitz.open(stream=payload, filetype="pdf")
        for page in document:
            pixmap = page.get_pixmap(matrix=fitz.Matrix(2, 2), alpha=False)
            image = Image.frombytes("RGB", [pixmap.width, pixmap.height], pixmap.samples)
            pages.append(pytesseract.image_to_string(image, config="--psm 6"))
        text = "\n".join(pages)
    except Exception as exc:
        raise DataParserError(f"PDF contains no text and OCR is unavailable: {exc}") from exc
    return parse_tabular_text(text)


def parse_image(buffer: bytes | bytearray | BytesIO) -> pd.DataFrame:
    """OCR a table image and normalize the recognized text."""
    payload = buffer.getvalue() if isinstance(buffer, BytesIO) else bytes(buffer)
    try:
        from PIL import Image, ImageOps
        import pytesseract

        image = Image.open(BytesIO(payload)).convert("L")
        image = ImageOps.autocontrast(image)
        text = pytesseract.image_to_string(image, config="--psm 6")
    except Exception as exc:
        raise DataParserError(f"Could not OCR image: {exc}") from exc
    return parse_tabular_text(text)


def parse_uploaded_file(buffer: bytes | bytearray | BytesIO, filename: str) -> pd.DataFrame:
    """Dispatch supported uploads to CSV, Excel, PDF, or image parsing."""
    suffix = Path(filename).suffix.lower()
    if suffix == ".csv":
        return parse_csv_buffer(buffer, filename)
    if suffix in {".xlsx", ".xls"}:
        return parse_excel_buffer(buffer, filename)
    if suffix == ".pdf":
        return parse_pdf(buffer)
    if suffix in {".png", ".jpg", ".jpeg", ".webp", ".tif", ".tiff", ".bmp"}:
        return parse_image(buffer)
    raise DataParserError("Supported uploads are CSV, XLSX, XLS, PDF, PNG, JPG, WEBP, TIFF, or BMP")


def parse_tabular_text(text: str) -> pd.DataFrame:
    """Parse comma-, tab-, or whitespace-delimited tabular text."""
    cleaned = str(text or "").strip()
    if not cleaned:
        raise DataParserError("No tabular data was supplied")

    lines = [line for line in cleaned.splitlines() if line.strip()]
    labeled = _parse_labeled_rows(lines)
    if labeled is not None:
        return labeled
    delimiter = _detect_delimiter(lines)
    try:
        if delimiter == "whitespace":
            frame = pd.read_csv(StringIO("\n".join(lines)), sep=r"\s+", engine="python")
        else:
            frame = pd.read_csv(StringIO("\n".join(lines)), sep=delimiter)
    except (csv.Error, pd.errors.ParserError, ValueError) as exc:
        raise DataParserError("Could not determine a valid table delimiter") from exc
    frame.columns = [str(column).strip() for column in frame.columns]
    if frame.empty or any(not column for column in frame.columns):
        raise DataParserError("The tabular input must include headers and data rows")
    return frame


def _detect_delimiter(lines: list[str]) -> str:
    sample = "\n".join(lines[:10])
    if "\t" in sample:
        return "\t"
    if "," in sample:
        return ","
    if ";" in sample:
        return ";"
    if any(len(line.split()) > 1 for line in lines):
        return "whitespace"
    raise DataParserError("Could not determine a valid table delimiter")


def _parse_labeled_rows(lines: list[str]) -> pd.DataFrame | None:
    if not all(":" in line for line in lines):
        return None
    rows: list[dict[str, object]] = []
    for line in lines:
        label, values = line.split(":", 1)
        parsed_values = [value.strip() for value in values.split(",") if value.strip()]
        if not label.strip() or not parsed_values:
            raise DataParserError("Labeled rows must contain a name and values")
        for value in parsed_values:
            rows.append({"Factor": label.strip(), "Metric": value})
    return pd.DataFrame(rows)


def _clean_frame(frame: pd.DataFrame, source: str) -> pd.DataFrame:
    frame = frame.dropna(axis=0, how="all").dropna(axis=1, how="all")
    if frame.empty:
        raise DataParserError(f"{source} is empty")
    frame.columns = [str(column).strip() for column in frame.columns]
    if any(not column for column in frame.columns):
        raise DataParserError(f"{source} must include non-empty headers")
    return frame