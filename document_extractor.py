from __future__ import annotations

import io
import os
import re
import unicodedata
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

try:
    from docx import Document as DocxDocument
except Exception:  # pragma: no cover
    DocxDocument = None

try:
    import cv2
except Exception:  # pragma: no cover
    cv2 = None

try:
    import pytesseract
except Exception:  # pragma: no cover
    pytesseract = None

try:
    from PIL import Image
except Exception:  # pragma: no cover
    Image = None


def _normalize_cell_text(value: Any) -> str:
    if value is None:
        return ""
    text = str(value)
    text = text.replace("\r", " ").replace("\n", " ").replace("\t", " ")
    text = unicodedata.normalize("NFKC", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _clean_numeric_token(token: str) -> str:
    token = (token or "").strip()
    token = token.replace("\u00a0", " ").replace("%", "")
    token = token.replace("$", "").replace("€", "").replace("£", "")
    token = re.sub(r"(?i)[^0-9.\-+eE]", "", token)
    if not token:
        return ""
    if token.lower() in {"nan", "inf", "-inf"}:
        return ""
    return token


def _sanity_fix_ocr_numeric(value: str) -> str:
    token = (value or "").strip()
    if not token:
        return ""
    if re.fullmatch(r"[A-Za-z]+", token):
        return token
    cleaned = token.replace("O", "0").replace("o", "0").replace("l", "1").replace("I", "1")
    cleaned = cleaned.replace("\u2013", "-").replace("\u2014", "-")
    return cleaned


def sanitize_extracted_table(table: dict[str, Any] | None) -> dict[str, Any] | None:
    if not isinstance(table, dict):
        return None
    headers = table.get("headers") or []
    rows = table.get("rows") or []
    if not isinstance(headers, list) or not isinstance(rows, list):
        return None
    cleaned_headers = []
    for header in headers:
        text = _normalize_cell_text(header)
        text = text.replace("_", " ").strip()
        if not text:
            continue
        cleaned_headers.append(text)
    if not cleaned_headers:
        return None
    cleaned_rows = []
    for row in rows:
        if not isinstance(row, list):
            continue
        cleaned_row = []
        for index, cell in enumerate(row[: len(cleaned_headers)]):
            value = _normalize_cell_text(cell)
            if value:
                value = _sanitize_value(value, index)
            cleaned_row.append(value)
        if any(str(item).strip() for item in cleaned_row):
            cleaned_rows.append(cleaned_row)
    if not cleaned_rows:
        return None
    normalized = []
    for row in cleaned_rows:
        if len(row) < len(cleaned_headers):
            row = row + [""] * (len(cleaned_headers) - len(row))
        elif len(row) > len(cleaned_headers):
            row = row[: len(cleaned_headers)]
        normalized.append(row)
    return {"headers": cleaned_headers, "rows": normalized}


def _sanitize_value(value: str, index: int) -> str:
    value = _normalize_cell_text(value)
    value = value.replace("\u00a0", " ")
    value = value.replace("−", "-").replace("–", "-").replace("—", "-")
    value = value.replace("₹", "").replace("$", "").replace("€", "").replace("£", "")
    value = value.replace("O", "0").replace("o", "0").replace("l", "1").replace("I", "1")
    if value and re.fullmatch(r"[\-+]?\d{1,3}(?:[.,]\d+)?", value):
        if "," in value and "." in value:
            value = value.replace(".", "").replace(",", ".")
        elif "," in value and re.search(r"\d,\d", value):
            value = value.replace(",", ".")
        elif "," in value:
            if index == 0:
                value = value.replace(",", "")
            else:
                value = value.replace(",", "")
    if value and re.fullmatch(r"[\-+]?\d+[.,]\d+|[\-+]?\d+", value):
        value = value.replace(".", "", value.count(".") - 1) if value.count(".") > 1 else value
        value = value.replace("O", "0").replace("o", "0").replace("l", "1").replace("I", "1")
    return value.strip()


def _as_dataframe_from_table(table: dict[str, Any] | None) -> pd.DataFrame | None:
    cleaned = sanitize_extracted_table(table)
    if cleaned is None:
        return None
    frame = pd.DataFrame(cleaned["rows"], columns=cleaned["headers"])
    if frame.empty:
        return None
    frame = frame.dropna(axis=0, how="all").dropna(axis=1, how="all")
    if frame.empty:
        return None
    frame.columns = [str(column).strip() for column in frame.columns]
    frame = frame.reset_index(drop=True)
    return frame


def _extract_docx_tables(path: Path) -> list[dict[str, Any]]:
    if DocxDocument is None:
        return []
    document = DocxDocument(str(path))
    tables: list[dict[str, Any]] = []
    for table in document.tables:
        rows = []
        for row in table.rows:
            rows.append([_normalize_cell_text(cell.text) for cell in row.cells])
        if not rows:
            continue
        headers = rows[0]
        data_rows = rows[1:]
        if not headers or any(not str(cell).strip() for cell in headers):
            continue
        tables.append({"headers": headers, "rows": data_rows})
    return tables


def extract_document_table(path: str | os.PathLike[str]) -> pd.DataFrame | None:
    source = Path(path)
    if not source.exists():
        return None
    suffix = source.suffix.lower()
    if suffix == ".docx":
        tables = _extract_docx_tables(source)
        if not tables:
            return None
        frames = []
        for table in tables:
            frame = _as_dataframe_from_table(table)
            if frame is not None:
                frames.append(frame)
        if not frames:
            return None
        if len(frames) == 1:
            return frames[0]
        schema = frames[0].columns.tolist()
        matched = [frame for frame in frames if list(frame.columns) == schema]
        if matched:
            return pd.concat(matched, ignore_index=True)
        return frames[0]
    return None


def _pdf_table_rows_to_frame(table_rows: list[list[Any]]) -> pd.DataFrame | None:
    if not table_rows:
        return None
    rows = [list(map(_normalize_cell_text, row)) for row in table_rows]
    rows = [[cell for cell in row if str(cell).strip()] for row in rows if any(str(cell).strip() for cell in row)]
    if not rows:
        return None
    header = rows[0]
    data_rows = rows[1:]
    if not header or any(not str(cell).strip() for cell in header):
        return None
    table = {"headers": header, "rows": data_rows}
    return _as_dataframe_from_table(table)


def extract_pdf_tables(path: str | os.PathLike[str]) -> list[pd.DataFrame]:
    source = Path(path)
    if not source.exists() or source.suffix.lower() not in {".pdf"}:
        return []
    try:
        import pdfplumber
    except Exception:
        return []
    frames: list[pd.DataFrame] = []
    with pdfplumber.open(str(source)) as document:
        for page in document.pages:
            extracted = page.extract_tables() or []
            for table in extracted:
                frame = _pdf_table_rows_to_frame(table)
                if frame is not None:
                    frames.append(frame)
            tables = page.find_tables()
            for table in tables:
                try:
                    if hasattr(table, "extract"):
                        frame = _pdf_table_rows_to_frame(table.extract())
                        if frame is not None:
                            frames.append(frame)
                except Exception:
                    continue
    return frames


def _ocr_single_cell(image: Any) -> str:
    if pytesseract is None:
        return ""
    try:
        text = pytesseract.image_to_string(image, config="--psm 10")
    except Exception:
        try:
            text = pytesseract.image_to_string(image, config="--psm 6")
        except Exception:
            text = ""
    return _normalize_cell_text(text)


def _ocr_single_cell_confidence(image: Any) -> tuple[str, float]:
    if pytesseract is None:
        return ("", 0.0)
    try:
        data = pytesseract.image_to_data(image, config="--psm 10", output_type=pytesseract.Output.DICT)
    except Exception:
        try:
            data = pytesseract.image_to_data(image, config="--psm 6", output_type=pytesseract.Output.DICT)
        except Exception:
            text = _ocr_single_cell(image)
            return (text, 0.0)
    texts = data.get("text", []) or []
    confs = data.get("conf", []) or []
    if not texts:
        text = _ocr_single_cell(image)
        return (text, 0.0)
    accepted = []
    for text, conf in zip(texts, confs):
        if not text or not str(text).strip():
            continue
        try:
            score = float(conf)
        except Exception:
            score = 0.0
        accepted.append((str(text).strip(), score))
    if not accepted:
        text = _ocr_single_cell(image)
        return (text, 0.0)
    label, score = accepted[0]
    return (" ".join(part for part, _ in accepted), max(score, 0.0))


def _is_malformed_table(frame: pd.DataFrame | None) -> bool:
    if frame is None or frame.empty:
        return True
    if frame.shape[1] == 0 or frame.shape[0] == 0:
        return True
    nonempty = frame.astype(str).replace({"nan": "", "None": ""}).applymap(lambda cell: str(cell).strip())
    filled = (nonempty != "").to_numpy().sum()
    density = filled / max(1, frame.size)
    if density < 0.1:
        return True
    if frame.shape[1] == 1 and frame.shape[0] <= 2:
        return True
    return False


def _deskew_image(image: Any) -> Any:
    if cv2 is None:
        return image
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (3, 3), 0)
    edges = cv2.Canny(blur, 50, 150, apertureSize=3)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=100, minLineLength=50, maxLineGap=10)
    if lines is None:
        return image
    angles = []
    for line in lines:
        x1, y1, x2, y2 = line[0]
        if x2 == x1:
            continue
        angle = np.degrees(np.arctan2(y2 - y1, x2 - x1))
        angles.append(angle)
    if not angles:
        return image
    median_angle = float(np.median(angles))
    if abs(median_angle) < 1:
        return image
    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)
    matrix = cv2.getRotationMatrix2D(center, median_angle, 1.0)
    rotated = cv2.warpAffine(image, matrix, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    return rotated


def _preprocess_image_for_table(image: Any) -> Any:
    if image is None:
        return None
    if cv2 is None:
        return image
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    gray = cv2.equalizeHist(gray)
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    if cv2.countNonZero(thresh) > thresh.size * 0.65:
        thresh = cv2.bitwise_not(thresh)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    return thresh


def _group_contours_into_rows(contours: list[Any], image_shape: tuple[int, int]) -> list[list[tuple[int, int, int, int]]]:
    rects = []
    height, width = image_shape[:2]
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        area = w * h
        if area < 60:
            continue
        if w < max(10, width * 0.02) and h < max(8, height * 0.02):
            continue
        rects.append((x, y, w, h))
    if not rects:
        return []
    grouped: list[list[tuple[int, int, int, int]]] = []
    for rect in sorted(rects, key=lambda value: (value[1], value[0])):
        x, y, w, h = rect
        center = y + h / 2.0
        placed = False
        for row in grouped:
            row_centers = [item[1] + item[3] / 2.0 for item in row]
            if row_centers and abs(center - np.median(row_centers)) <= max(8, np.median([item[3] for item in row]) * 0.9):
                row.append(rect)
                placed = True
                break
        if not placed:
            grouped.append([rect])
    return grouped


def _ocr_cell_from_region(region: Any) -> str:
    text = _ocr_single_cell(region)
    cleaned = _normalize_cell_text(text)
    if not cleaned:
        return ""
    return cleaned


def _detect_table_grid_rows_cols(binary: Any) -> tuple[list[int], list[int]] | None:
    if cv2 is None or binary is None:
        return None
    if len(binary.shape) == 3:
        binary = cv2.cvtColor(binary, cv2.COLOR_BGR2GRAY)
    h, w = binary.shape[:2]
    if h < 20 or w < 20:
        return None
    dark = binary < 200
    row_density = np.sum(dark, axis=1).astype(float)
    col_density = np.sum(dark, axis=0).astype(float)
    row_threshold = max(4.0, 0.08 * w)
    col_threshold = max(4.0, 0.08 * h)
    row_runs = []
    start = None
    for idx, value in enumerate(row_density):
        if value > row_threshold and start is None:
            start = idx
        elif value <= row_threshold and start is not None:
            row_runs.append((start, idx))
            start = None
    if start is not None:
        row_runs.append((start, h))
    col_runs = []
    start = None
    for idx, value in enumerate(col_density):
        if value > col_threshold and start is None:
            start = idx
        elif value <= col_threshold and start is not None:
            col_runs.append((start, idx))
            start = None
    if start is not None:
        col_runs.append((start, w))
    if len(row_runs) < 2 or len(col_runs) < 2:
        return None
    row_bounds = []
    for start, end in row_runs:
        if end - start > max(4, h * 0.03):
            row_bounds.append((start, end))
    col_bounds = []
    for start, end in col_runs:
        if end - start > max(4, w * 0.03):
            col_bounds.append((start, end))
    if len(row_bounds) < 2 or len(col_bounds) < 2:
        return None
    rows = [int(start) for start, _ in row_bounds]
    cols = [int(start) for start, _ in col_bounds]
    return rows, cols


def _extract_grid_table_from_image(image: Any) -> pd.DataFrame | None:
    if cv2 is None or image is None:
        return None
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    blur = cv2.GaussianBlur(gray, (3, 3), 0)
    _, binary = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    grid = _detect_table_grid_rows_cols(binary)
    if grid is None:
        return None
    row_starts, col_starts = grid
    if not row_starts or not col_starts:
        return None
    rows: list[list[str]] = []
    rows_count = max(2, len(row_starts))
    cols_count = max(2, len(col_starts))
    for row_idx in range(rows_count - 1):
        cells: list[str] = []
        y1 = int(row_starts[row_idx])
        y2 = int(row_starts[row_idx + 1]) if row_idx + 1 < len(row_starts) else gray.shape[0]
        if y2 <= y1:
            continue
        for col_idx in range(cols_count - 1):
            x1 = int(col_starts[col_idx])
            x2 = int(col_starts[col_idx + 1]) if col_idx + 1 < len(col_starts) else gray.shape[1]
            if x2 <= x1:
                continue
            cell = binary[max(0, y1):min(binary.shape[0], y2), max(0, x1):min(binary.shape[1], x2)]
            if cell.size == 0:
                cells.append("")
                continue
            text, score = _ocr_single_cell_confidence(cell)
            text = _normalize_cell_text(text)
            if text and score >= 25.0:
                cells.append(text)
            else:
                cells.append("")
        if any(cell.strip() for cell in cells):
            rows.append(cells)
    if not rows:
        return None
    frame = pd.DataFrame(rows)
    frame = frame.dropna(axis=0, how="all").dropna(axis=1, how="all")
    if frame.empty:
        return None
    if _is_malformed_table(frame):
        return None
    return frame


def extract_image_table(path: str | os.PathLike[str]) -> pd.DataFrame | None:
    source = Path(path)
    if not source.exists():
        return None
    if cv2 is None and pytesseract is None:
        return None
    try:
        if cv2 is not None:
            image = cv2.imread(str(source), cv2.IMREAD_COLOR)
            if image is None:
                return None
            image = _deskew_image(image)
            grid_frame = _extract_grid_table_from_image(image)
            if grid_frame is not None and not grid_frame.empty:
                return grid_frame
            preprocessed = _preprocess_image_for_table(image)
            if preprocessed is None:
                return None
            horizontal = cv2.morphologyEx(preprocessed, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (50, 1)))
            vertical = cv2.morphologyEx(preprocessed, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (1, 50)))
            mask = cv2.bitwise_or(horizontal, vertical)
            contours, _ = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
            row_groups = _group_contours_into_rows(contours, preprocessed.shape[:2])
            if not row_groups:
                return None
            rows: list[list[str]] = []
            for row in row_groups:
                row_items = sorted(row, key=lambda value: value[0])
                if not row_items:
                    continue
                cells: list[str] = []
                for x, y, w, h in row_items:
                    cell = preprocessed[max(0, y):min(preprocessed.shape[0], y + h), max(0, x):min(preprocessed.shape[1], x + w)]
                    if cell.size == 0:
                        continue
                    text, score = _ocr_single_cell_confidence(cell)
                    text = _normalize_cell_text(text)
                    if text and score >= 20.0:
                        cells.append(text)
                if cells:
                    rows.append(cells)
            if not rows:
                return None
            max_columns = max(len(row) for row in rows)
            normalized_rows = []
            for row in rows:
                if len(row) < max_columns:
                    row = row + [""] * (max_columns - len(row))
                normalized_rows.append(row[:max_columns])
            frame = pd.DataFrame(normalized_rows)
            frame = frame.dropna(axis=0, how="all").dropna(axis=1, how="all")
            if frame.empty:
                return None
            if _is_malformed_table(frame):
                return None
            if len(frame.columns) == 1 and frame.iloc[0].astype(str).str.len().sum() > 0:
                return frame
            first_row = [str(value).strip() for value in frame.iloc[0].tolist()]
            if all(value == "" for value in first_row):
                frame = frame.iloc[1:]
            if frame.empty:
                return None
            frame.columns = [f"Column_{idx + 1}" for idx in range(len(frame.columns))]
            first_row = [str(value).strip() for value in frame.iloc[0].tolist()]
            if any(value for value in first_row):
                frame = pd.DataFrame(frame.iloc[1:].to_numpy(), columns=[f"Column_{idx + 1}" for idx in range(len(frame.columns))])
            if _is_malformed_table(frame):
                return None
            return frame
        if Image is not None:
            image = Image.open(source)
            text = pytesseract.image_to_string(image, config="--psm 6")
            return _as_dataframe_from_table({"headers": ["Value"], "rows": [[_normalize_cell_text(text)]]})
    except Exception:
        return None
    return None


def extract_structured_table(path: str | os.PathLike[str]) -> pd.DataFrame | None:
    source = Path(path)
    if not source.exists():
        return None
    suffix = source.suffix.lower()
    if suffix == ".docx":
        return extract_document_table(source)
    if suffix == ".pdf":
        frames = extract_pdf_tables(source)
        if frames:
            if len(frames) == 1:
                return frames[0]
            return pd.concat(frames, ignore_index=True)
        # Fallback: try a generic OCR pass on the first page only when a PDF is essentially an image-based scan.
        try:
            import pdfplumber
            with pdfplumber.open(str(source)) as document:
                if not document.pages:
                    return None
                page = document.pages[0]
                image = page.to_image(resolution=220)
                image_bytes = image.original
                if hasattr(image_bytes, "save"):
                    buffer = io.BytesIO()
                    image_bytes.save(buffer, format="PNG")
                    buffer.seek(0)
                    temp = Path(str(source) + ".png")
                    temp.write_bytes(buffer.getvalue())
                    frame = extract_image_table(temp)
                    temp.unlink(missing_ok=True)
                    if frame is not None and not frame.empty:
                        return frame
        except Exception:
            return None
        return None
    if suffix.lower() in {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}:
        frame = extract_image_table(source)
        if frame is not None and not frame.empty:
            return frame
        if Image is not None:
            try:
                image = Image.open(source)
                text = pytesseract.image_to_string(image, config="--psm 6") if pytesseract is not None else ""
                if text.strip():
                    return _as_dataframe_from_table({"headers": ["Value"], "rows": [[_normalize_cell_text(text)]]})
            except Exception:
                return None
    return None


def build_extraction_preview(frame: pd.DataFrame | None) -> str:
    if frame is None or frame.empty:
        return "Table Extracted: 0 rows x 0 columns. Check if columns align properly."
    if frame.shape[1] == 0:
        return f"Table Extracted: {frame.shape[0]} rows x 0 columns. Check if columns align properly."
    preview = frame.head(5).copy()
    preview = preview.fillna("")
    width = max(3, min(120, len(preview.columns) * 14))
    lines = [f"Table Extracted: {frame.shape[0]} rows x {frame.shape[1]} columns. Check if columns align properly."]
    lines.append("| " + " | ".join(str(column) for column in preview.columns) + " |")
    lines.append("|" + "|".join("-" * max(5, len(str(column))) for column in preview.columns) + "|")
    for _, row in preview.iterrows():
        lines.append("| " + " | ".join(str(value) for value in row.tolist()) + " |")
    return "\n".join(lines)[:2000]
