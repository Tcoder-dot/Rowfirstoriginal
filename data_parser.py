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


def parse_pdf(_: bytes | bytearray | BytesIO) -> pd.DataFrame:
    raise NotImplementedError("PDF and Image parsing are scheduled for v2.0")


def parse_image(_: bytes | bytearray | BytesIO) -> pd.DataFrame:
    raise NotImplementedError("PDF and Image parsing are scheduled for v2.0")


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