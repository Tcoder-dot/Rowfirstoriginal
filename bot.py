"""Rowfirst Telegram bot.

SciPy owns all statistics. Local CSV/XLSX/ZIP files never use Gemini.
Gemini is used only for table extraction from photo/PDF/DOCX when configured.
"""
from __future__ import annotations

import ast
import csv
import difflib
import hashlib
import io
import json
import os
import re
import tempfile
import threading
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from flask import Flask
from scipy import stats

try:
    from rapidfuzz import process as rapidfuzz_process
except Exception:  # pragma: no cover
    rapidfuzz_process = None

from chapter4 import write_docx
from charts import make_charts
from data_explorer import (
    run_explorer_action,
    run_explorer_query,
)
from document_extractor import build_extraction_preview, extract_document_table, extract_structured_table
from handle import analyze_ingested, build_breakdown, handle_analyze
from ingest import ingest_file, ingest_text
from qa import quality_check

try:
    import telebot
    from telebot import types
except ImportError:
    telebot = None
    types = None


health_app = Flask(__name__)
user_sessions: dict[int, dict[str, Any]] = {}
last_engine: dict[int, dict[str, Any]] = {}
pending_extracted: dict[int, dict[str, Any]] = {}
_INGESTION_DELIVERY_CACHE: set[tuple[int, int]] = set()


@health_app.get("/")
def health_check() -> tuple[str, int]:
    return "ROWFIRST engine active", 200


def _run_health_server() -> None:
    port = int(os.environ.get("PORT", 8080))
    health_app.run(host="0.0.0.0", port=port, use_reloader=False)


_GLOBAL_RESET_RE = re.compile(
    r"(?i)^(?:cancel|stop|abort|clear|reset|exit|quit|back|never\s*mind|start\s*over)$"
)
_GLOBAL_RESET_INTENT_RE = re.compile(
    r"(?ix)"
    r"(?:\b(?:cancel|stop|abort|reset|quit|exit)\b|"
    r"\b(?:never\s*mind|start\s*over|forget\s*(?:it|this|that))\b|"
    r"\b(?:change|changed)\s+my\s+mind\b|"
    r"\b(?:do\s+not|don't|dont)\s+(?:continue|proceed|do\s+that)\b)"
)
_RESET_CALLBACK_DATA = "choice;calcel session"


_TABLE_FACTOR_NAMES = {
    "arm",
    "category",
    "class",
    "condition",
    "group",
    "method",
    "school",
    "sex",
    "site",
    "studytime",
    "treatment",
    "type",
    "variant",
}
_TABLE_ID_NAMES = {
    "code",
    "id",
    "index",
    "patient",
    "patientid",
    "record",
    "recordid",
    "sample",
    "sampleid",
    "student",
    "studentid",
    "subject",
    "subjectid",
    "userid",
}


def _table_column_name(value: Any) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(value).strip().lower())


def _column_title(value: Any) -> str:
    text = str(value or "").strip()
    if not text:
        return "Unnamed"
    lowered = text.lower()
    if re.search(r"(?i)(?:^|_)log(?:_|$)", text) and re.search(r"(?i)cfu", text):
        return "log CFU/g"
    if re.search(r"(?i)(?:^|_)log(?:_|$)", text):
        text = re.sub(r"(?i)(?:^|_)(log)(?:_|$)", " log ", text)
    if re.search(r"(?i)\b(?:pct|percent)\b|\(%\)$|_pct$|_percent$", text):
        text = re.sub(r"(?i)\b(?:pct|percent)\b", "%", text)
        text = re.sub(r"(?i)_pct$|_percent$", " (%)", text)
        text = re.sub(r"\s*\(%\)\s*", " (%)", text)
    if lowered.endswith("_g") or re.search(r"(?i)(?:^|_)(?:mg|g|kg|ml|l)(?:_|$)", text):
        text = re.sub(r"(?i)_?(g|mg|kg|ml|l)(?:_|$)", lambda match: f" {match.group(1).upper()}", text)
    text = text.replace("_", " ")
    text = re.sub(r"\s+", " ", text).strip()
    text = re.sub(r"\bcm\b", "cm", text, flags=re.I)
    text = re.sub(r"\b(?:log|ln|sqrt)\b", lambda match: match.group(0).lower(), text, flags=re.I)
    text = re.sub(r"\bCfU\b", "CFU", text, flags=re.I)
    text = re.sub(r"\b(?:pct|percent)\b", "%", text, flags=re.I)
    if re.search(r"(?i)\bCFU\b", text) and "CFU/g" not in text and "/g" not in text:
        text = re.sub(r"(?i)\bCFU\b", "CFU/g", text)
    if " (%)" not in text and re.search(r"(?i)\b(?:percent|pct)\b|\(%\)", text):
        text = text.replace("%", " (%)")
    text = re.sub(r"\s+\(\%\)", " (%)", text)
    if text.lower().endswith("cfu/g"):
        return text if "log" in text.lower() else f"{text}"
    return text


def classify_columns(frame: pd.DataFrame | None) -> dict[str, list[str]]:
    if frame is None or frame.empty:
        return {"categorical_factors": [], "numeric_metrics": [], "metadata_columns": []}
    metadata_columns: list[str] = []
    categorical_factors: list[str] = []
    numeric_metrics: list[str] = []
    for column in frame.columns:
        if _is_metadata_only_column(frame, column) or _is_identifier_column(frame, column):
            metadata_columns.append(str(column))
            continue
        if _is_temporal_or_datetime_column(frame, column):
            metadata_columns.append(str(column))
            continue

        series = frame[column]
        numeric = pd.to_numeric(series, errors="coerce")
        valid_numbers = numeric.dropna()

        if valid_numbers.empty:
            non_null = series.dropna()
            if not non_null.empty and non_null.astype(str).nunique(dropna=True) > 1:
                categorical_factors.append(str(column))
            continue

        if pd.api.types.is_float_dtype(series) or valid_numbers.apply(lambda x: not float(x).is_integer()).any():
            numeric_metrics.append(str(column))
            continue
        if valid_numbers.size < max(2, int(len(frame) * 0.8)):
            continue
        if _is_binary_or_flag_numeric(series):
            continue
        if valid_numbers.nunique(dropna=True) <= 5:
            categorical_factors.append(str(column))
        else:
            numeric_metrics.append(str(column))

    for column in list(frame.columns):
        name = str(column)
        if name in categorical_factors or name in numeric_metrics:
            continue
        if name in metadata_columns:
            continue
        series = frame[column]
        numeric = pd.to_numeric(series, errors="coerce")
        valid_numbers = numeric.dropna()

        if valid_numbers.empty:
            non_null = series.dropna()
            if not non_null.empty and non_null.astype(str).nunique(dropna=True) > 1:
                categorical_factors.append(name)
            continue

        if pd.api.types.is_float_dtype(series) or valid_numbers.apply(lambda x: not float(x).is_integer()).any():
            numeric_metrics.append(name)
            continue
        if valid_numbers.size < max(2, int(len(frame) * 0.8)):
            continue
        if _is_binary_or_flag_numeric(series):
            continue
        if valid_numbers.nunique(dropna=True) <= 5:
            categorical_factors.append(name)
        else:
            numeric_metrics.append(name)
    return {
        "categorical_factors": list(dict.fromkeys(categorical_factors)),
        "numeric_metrics": list(dict.fromkeys(numeric_metrics)),
        "metadata_columns": list(dict.fromkeys(metadata_columns)),
    }


def _coerce_p_value_text(value: Any) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "not reported"
    if number < 0.001:
        return "p < .001"
    if number >= 1.0:
        return f"p = {number:.3f}"
    return f"p = {number:.4f}"


def _read_delimited_frame(raw_text: str) -> pd.DataFrame | None:
    """Read pasted or uploaded delimited text without changing engine logic."""
    raw = (raw_text or "").strip()
    if not raw or "\n" not in raw:
        return None
    try:
        sample = raw[:8192]
        dialect = csv.Sniffer().sniff(sample, delimiters=",;\t")
        separator = dialect.delimiter
    except csv.Error:
        separator = None
    try:
        frame = pd.read_csv(io.StringIO(raw), sep=separator, engine="python")
    except (pd.errors.ParserError, ValueError):
        return None
    frame = frame.dropna(axis=0, how="all").dropna(axis=1, how="all")
    return frame if frame.shape[1] >= 2 and not frame.empty else None


def _clean_currency_like_cell(value: Any) -> Any:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return value
    if not isinstance(value, str):
        return value
    cleaned = value.strip()
    if not cleaned:
        return ""
    cleaned = cleaned.replace("₦", "").replace("$", "").replace("€", "").replace("£", "")
    cleaned = cleaned.replace(",", "").replace("%", "")
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned


def sanitize_incoming_dataframe(df: pd.DataFrame | None) -> pd.DataFrame | None:
    """Normalize uploaded and extracted data before routing to the deterministic stats engine."""
    if df is None or df.empty:
        return df
    sanitized = df.copy()
    sanitized = sanitized.dropna(axis=0, how="all").dropna(axis=1, how="all")
    sanitized.columns = [str(column).strip() for column in sanitized.columns]
    for column in sanitized.columns:
        series = sanitized[column].copy()
        cleaned = series.map(_clean_currency_like_cell)
        if cleaned.empty:
            continue
        numeric_like = cleaned.map(lambda value: isinstance(value, str) and bool(re.fullmatch(r"[-+]?\d*\.?\d+(?:e[-+]?\d+)?", value.strip())))
        if numeric_like.any():
            try:
                coerced = pd.to_numeric(cleaned, errors="coerce")
                if coerced.notna().sum() >= max(1, int(len(cleaned) * 0.8)):
                    sanitized[column] = coerced
                    continue
            except Exception:
                pass
        sanitized[column] = cleaned.map(lambda value: value.strip() if isinstance(value, str) else value)
    return sanitized


def _numeric_table_columns(frame: pd.DataFrame) -> list[str]:
    numeric: list[str] = []
    for column in frame.columns:
        if _is_identifier_column(frame, column) or _is_temporal_or_datetime_column(frame, column):
            continue
        values = pd.to_numeric(frame[column], errors="coerce")
        if values.notna().sum() >= max(1, int(len(frame) * 0.8)):
            numeric.append(column)
    return numeric


def _is_strict_row_identifier_series(series: pd.Series) -> bool:
    non_null = series.dropna()
    if non_null.empty:
        return False
    numeric = pd.to_numeric(non_null, errors="coerce")
    if numeric.empty or numeric.notna().sum() != len(non_null):
        return False
    sequence = numeric.astype(int).tolist() if (numeric % 1 == 0).all() else []
    if not sequence:
        return False
    expected_one_based = list(range(1, len(non_null) + 1))
    expected_zero_based = list(range(len(non_null)))
    return sequence == expected_one_based or sequence == expected_zero_based


def _is_temporal_or_datetime_column(frame: pd.DataFrame, column: str) -> bool:
    if column not in frame.columns:
        return False
    series = frame[column]
    if isinstance(series.dtype, pd.DatetimeTZDtype) or pd.api.types.is_datetime64_any_dtype(series):
        return True
    name = str(column).lower()
    if re.search(r"(?i)(date|time|timestamp|created|updated|renewal|activity|period|quarter|year|month)", name):
        non_null = series.dropna().astype(str)
        if non_null.empty:
            return True
        text_like = non_null.str.contains(r"[-/T Z]|Q[1-4]\s*\d{4}", case=False, regex=True)
        if text_like.any():
            return True
        numeric = pd.to_numeric(series, errors="coerce").dropna()
        if not numeric.empty and numeric.between(40000, 60000).all() and len(numeric) >= max(1, int(len(series) * 0.8)):
            return True
        return True
    return False


def _is_identifier_column(frame: pd.DataFrame, column: str) -> bool:
    if column not in frame.columns:
        return False
    normalized = _table_column_name(column)
    if re.fullmatch(r"(?i)(index|id|row_id|_id|uuid|record_id|unnamed|sn|s/n)", normalized):
        return True
    if re.search(r"(?i)(?:^|(?:row|record|customer|account|patient|student|subject|user|order|transaction|employee|member|case|lead|invoice|product|client))id$|(?:^|_)index$|(?:^|_)uuid$|(?:^|_)sn$|(?:^|_)unnamed$", normalized):
        return True
    if normalized in _TABLE_ID_NAMES:
        return True
    series = frame[column]
    if _is_strict_row_identifier_series(series):
        return True
    if pd.api.types.is_numeric_dtype(series):
        return False
    non_null = series.dropna()
    if non_null.empty:
        return False
    unique_count = int(non_null.nunique(dropna=True))
    row_count = len(frame)
    if unique_count == row_count:
        return True
    ratio = unique_count / max(1, row_count)
    if ratio > 0.5 and unique_count >= max(10, int(row_count * 0.2)):
        return True
    return False


def _is_binary_or_flag_numeric(series: pd.Series) -> bool:
    values = pd.to_numeric(series, errors="coerce").dropna()
    if values.empty:
        return False
    unique = values.nunique(dropna=True)
    if unique <= 2:
        return True
    return bool(set(values.unique()) <= {0.0, 1.0})


def _is_valid_continuous_outcome(series: pd.Series) -> bool:
    values = pd.to_numeric(series, errors="coerce").dropna()
    if values.empty or len(values) < 10:
        return False
    unique = values.nunique(dropna=True)
    if unique <= 10:
        return False
    variance = float(values.var(ddof=1)) if len(values) > 1 else 0.0
    return bool(np.isfinite(variance) and variance > 0.0)


def _is_metadata_only_column(frame: pd.DataFrame, column: str) -> bool:
    if column not in frame.columns:
        return False
    normalized = _table_column_name(column)
    if re.fullmatch(r"(?i)(index|row_id|id|s/n|sn|record_id|customer_id|user_id|order_id|case_id|lead_id)", normalized):
        return True
    series = frame[column].dropna()
    if series.empty:
        return False
    try:
        numeric = pd.to_numeric(series, errors="coerce")
    except Exception:
        numeric = pd.Series([], dtype=float)
    if numeric.empty:
        return False
    if numeric.notna().sum() != len(series):
        return False
    if not (numeric % 1 == 0).all():
        return False
    expected = list(range(1, len(series) + 1))
    actual = [int(value) for value in numeric.astype(int).tolist()]
    return actual == expected or actual == list(range(0, len(series)))


def _is_temporal_metric_series(series: pd.Series) -> bool:
    if series.empty:
        return False
    text = series.dropna().astype(str)
    if text.empty:
        return False
    date_pattern = r"\d{4}[-/]\d{1,2}[-/]\d{1,2}|\d{1,2}[-/]\d{1,2}[-/]\d{2,4}|Q[1-4]\s*\d{2,4}|\d{4}Q[1-4]|\d{2}:\d{2}|T\d{2}:\d{2}"
    if text.str.contains(date_pattern, case=False, regex=True).any():
        return True
    numeric = pd.to_numeric(series, errors="coerce").dropna()
    if numeric.empty:
        return False
    return bool(numeric.between(40000, 60000).all() and len(numeric) >= max(1, int(len(series) * 0.8)))


def _column_is_factor_candidate(frame: pd.DataFrame, column: str) -> bool:
    if column not in frame.columns:
        return False
    if _is_metadata_only_column(frame, column):
        return False
    if _is_temporal_or_datetime_column(frame, column):
        return False
    if _is_identifier_column(frame, column):
        return False
    if pd.api.types.is_numeric_dtype(frame[column]):
        series = frame[column].dropna()
        if series.empty:
            return False
        values = pd.to_numeric(series, errors="coerce")
        if values.empty:
            return False
        if values.nunique(dropna=True) <= 2 and values.isin([0, 1]).all():
            return False
        return False
    series = frame[column].dropna().astype(str)
    if series.empty:
        return False
    unique_count = int(series.nunique(dropna=True))
    return 1 < unique_count < len(frame)


def _column_is_numeric_metric_candidate(frame: pd.DataFrame, column: str) -> bool:
    if column not in frame.columns:
        return False
    if _is_metadata_only_column(frame, column):
        return False
    if _is_temporal_or_datetime_column(frame, column):
        return False
    if _is_identifier_column(frame, column):
        return False
    if _is_binary_or_flag_numeric(frame[column]):
        return False
    cleaned = frame[column].map(_clean_currency_like_cell)
    coerced = pd.to_numeric(cleaned, errors="coerce")
    valid = coerced.notna()
    if valid.sum() < max(2, int(len(frame) * 0.8)):
        return False
    if coerced.dropna().nunique(dropna=True) <= 1:
        return False
    return True


def _multivariate_table_route(frame: pd.DataFrame | None) -> dict[str, Any] | None:
    """Classify a table before it enters the existing ingestion/engine path."""
    if frame is None or frame.shape[1] < 2:
        return None
    metadata_columns = [column for column in frame.columns if _is_metadata_only_column(frame, column)]
    temporal_columns = [column for column in frame.columns if _is_temporal_or_datetime_column(frame, column)]
    factor_columns = [
        column for column in frame.columns
        if column not in metadata_columns and column not in temporal_columns and _column_is_factor_candidate(frame, column)
    ]
    outcome_columns = [
        column for column in frame.columns
        if column not in metadata_columns and column not in temporal_columns and _column_is_numeric_metric_candidate(frame, column)
    ]

    if not factor_columns or not outcome_columns:
        return {
            "kind": "needs_mapping",
            "frame": frame,
            "factors": list(dict.fromkeys(factor_columns)),
            "outcomes": list(dict.fromkeys(outcome_columns)),
            "columns": list(frame.columns),
            "message": (
                "We could not automatically detect your grouping factor or outcome metric. "
                "Please manually map your columns:"
            ),
        }

    normalized_headers = {_table_column_name(column) for column in frame.columns}
    if (
        len(frame.columns) == 2
        and normalized_headers & {"group", "method", "treatment", "condition"}
        and len(outcome_columns) == 1
    ):
        return None

    return {
        "kind": "multivariate",
        "frame": frame,
        "factors": list(dict.fromkeys(factor_columns)),
        "outcomes": list(dict.fromkeys(outcome_columns)),
    }


def _group_size_error(ingested: dict[str, Any]) -> str | None:
    """Allow singleton triage instead of hard-stopping the flow."""
    return None


def _zero_variance_error(ingested: dict[str, Any]) -> str | None:
    groups = list(ingested.get("groups") or [])
    for outcome in ingested.get("outcomes") or []:
        groups.extend(outcome.get("groups") or [])
    if len(groups) < 2:
        return None
    arrays = [np.asarray(group.get("values", []), dtype=float) for group in groups]
    if all(
        len(values) > 1
        and np.isfinite(values).all()
        and np.all(values == values[0])
        for values in arrays
    ):
        return "F-test undefined: identical replicate measurements detected with zero within-group variance"
    return None


def _singleton_groups_for_frame(frame: pd.DataFrame, factor_name: str) -> list[dict[str, Any]]:
    if frame.empty or factor_name not in frame.columns:
        return []
    if _is_identifier_column(frame, factor_name):
        return []
    counts = frame[factor_name].dropna().value_counts(dropna=False)
    return [{"name": str(name), "n": int(count)} for name, count in counts.items() if count == 1]


def _canonicalize_text(value: Any) -> str:
    text = str(value or "").strip().lower()
    text = re.sub(r"(?i)\b(?:inc|ltd|limited|llc|corp|company|co|sa|s\.a\.|gmbh|plc|pte|pvt)\b$", "", text)
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def _detect_currency_pollution(series: pd.Series) -> bool:
    values = series.dropna().astype(str)
    if values.empty:
        return False
    for value in values.tolist():
        cleaned = value.strip().lower()
        if re.search(r"[$€£₦]|(?<![a-z])\d+(?:\.\d+)?[kmb](?![a-z])", cleaned):
            return True
    return False


def _detect_mixed_date_format(series: pd.Series) -> bool:
    values = series.dropna().astype(str)
    if values.empty:
        return False
    kinds: set[str] = set()
    for value in values.tolist():
        text = value.strip()
        if not text:
            continue
        lowered = text.lower()
        if re.search(r"q[1-4]\s*\d{2,4}", lowered):
            kinds.add("quarter")
        elif re.search(r"\d{4}[-/]\d{1,2}[-/]\d{1,2}", text) or "t" in lowered and re.search(r"\d{2}:\d{2}", text):
            kinds.add("iso")
        elif re.search(r"\d{1,2}[-/]\d{1,2}[-/]\d{2,4}", text):
            kinds.add("slash")
        elif text.isdigit() and 40000 <= int(text) <= 60000:
            kinds.add("excel_serial")
    return len(kinds) >= 2


def _build_data_health_profile(frame: pd.DataFrame | None) -> dict[str, Any]:
    if frame is None or frame.empty:
        return {
            "rows": 0,
            "columns": 0,
            "cells": 0,
            "missing_total": 0,
            "missing_pct": 0.0,
            "duplicate_rows": 0,
            "health_score": 100.0,
            "missing_by_column": [],
            "constant_columns": [],
            "currency_columns": [],
            "date_columns": [],
            "near_duplicate_columns": [],
            "findings": [],
        }
    rows = int(len(frame))
    columns = int(len(frame.columns))
    total_cells = int(frame.size)
    missing_by_column = []
    for column in frame.columns:
        series = frame[column]
        missing_count = int(series.isna().sum())
        if missing_count > 0:
            completeness = 100.0 * (1.0 - (missing_count / max(1, rows)))
            missing_by_column.append({
                "column": str(column),
                "missing": missing_count,
                "missing_pct": round(100.0 * (missing_count / max(1, rows)), 1),
                "completeness": round(completeness, 1),
            })
    missing_total = int(sum(item["missing"] for item in missing_by_column))
    missing_pct = 0.0 if total_cells == 0 else 100.0 * (missing_total / total_cells)
    overall_score = max(0.0, 100.0 - missing_pct)
    duplicate_rows = int(frame.duplicated().sum())
    constant_columns = [str(column) for column in frame.columns if frame[column].dropna().nunique(dropna=True) <= 1]
    currency_columns = []
    date_columns = []
    near_duplicate_columns = []
    for column in frame.columns:
        series = frame[column]
        if _detect_currency_pollution(series):
            currency_columns.append(str(column))
        if _detect_mixed_date_format(series):
            date_columns.append(str(column))
        if pd.api.types.is_object_dtype(series) or pd.api.types.is_string_dtype(series):
            values = series.dropna().astype(str)
            if values.empty:
                continue
            canonical = [_canonicalize_text(value) for value in values.tolist()]
            seen: dict[str, list[str]] = {}
            for value in canonical:
                if not value:
                    continue
                seen.setdefault(value, []).append(value)
            near_examples: list[str] = []
            unique_roots = sorted({value for value in canonical if value})
            for i, left in enumerate(unique_roots):
                for right in unique_roots[i + 1:]:
                    if left == right:
                        continue
                    ratio = difflib.SequenceMatcher(None, left, right).ratio()
                    if ratio >= 0.82:
                        near_examples.extend([left, right])
                        break
                if near_examples:
                    break
            if near_examples:
                near_duplicate_columns.append({"column": str(column), "examples": sorted(set(near_examples))[:3]})
    findings = []
    if missing_total:
        findings.append(f"Missing values: {missing_total} cells ({missing_pct:.1f}% of dataset)")
    if duplicate_rows:
        findings.append(f"Duplicate rows: {duplicate_rows}")
    if constant_columns:
        findings.append(f"Constant columns: {', '.join(constant_columns[:5])}")
    if date_columns:
        findings.append(f"Mixed date formats: {', '.join(date_columns[:5])}")
    if currency_columns:
        findings.append(f"Currency symbols detected: {', '.join(currency_columns[:5])}")
    if near_duplicate_columns:
        examples = "; ".join(f"{item['column']}={', '.join(item['examples'])}" for item in near_duplicate_columns[:2])
        findings.append(f"Likely duplicate labels: {examples}")
    if not findings:
        findings.append("No material hygiene issues detected.")
    return {
        "rows": rows,
        "columns": columns,
        "cells": total_cells,
        "missing_total": missing_total,
        "missing_pct": round(missing_pct, 1),
        "duplicate_rows": duplicate_rows,
        "health_score": round(overall_score, 1),
        "missing_by_column": missing_by_column,
        "constant_columns": constant_columns,
        "currency_columns": currency_columns,
        "date_columns": date_columns,
        "near_duplicate_columns": near_duplicate_columns,
        "findings": findings,
    }


def _format_data_health_card(profile: dict[str, Any], categorical_factors: list[str] | None = None, numeric_metrics: list[str] | None = None) -> str:
    factor_names = list(dict.fromkeys(categorical_factors or profile.get("categorical_factors", [])))
    metric_names = list(dict.fromkeys(numeric_metrics or profile.get("numeric_metrics", [])))
    factor_lines = [f"- {_column_title(name)}" for name in factor_names] if factor_names else ["- None detected"]
    metric_lines = [f"- {_column_title(name)}" for name in metric_names] if metric_names else ["- None detected"]
    lines = [
        "Data Health Card",
        "Data Ingestion Card",
        "",
        "Shape",
        f"- {profile.get('rows', 0)} rows x {profile.get('columns', 0)} columns",
        "",
        "Dataset Overview",
        f"- Total Cells: {profile.get('cells', 0)}",
        f"- Overall Data Health Score: {profile.get('health_score', 100.0):.1f}%",
        f"- Missing Cells: {profile.get('missing_total', 0)} ({profile.get('missing_pct', 0.0):.1f}%)",
        f"- Duplicate Rows: {profile.get('duplicate_rows', 0)}",
        "",
        "Detected Categorical Factors:",
        *factor_lines,
        "",
        "Detected Numeric Metrics:",
        *metric_lines,
        "",
        "Missing Values / Data Health:",
        f"- Missing cells: {profile.get('missing_total', 0)} total",
        f"- Duplicate rows: {profile.get('duplicate_rows', 0)}",
        "",
        "Critical Findings",
    ]
    findings = profile.get("findings") or ["No material hygiene issues detected."]
    for item in findings[:8]:
        lines.append(f"- {item}")
    lines.extend([
        "",
        "Hygiene Actions Taken",
        "- Clean CSV export sent automatically",
        "- Styled Excel export sent automatically",
        "- Step 1 variable selection continues immediately",
    ])
    return "\n".join(lines)


def _summarize_dataframe(frame: pd.DataFrame) -> str:
    classified = classify_columns(frame)
    profile = _build_data_health_profile(frame)
    profile["categorical_factors"] = classified["categorical_factors"]
    profile["numeric_metrics"] = classified["numeric_metrics"]
    return _format_data_health_card(profile, classified["categorical_factors"], classified["numeric_metrics"])


def _safe_python_globals() -> dict[str, Any]:
    safe_builtins = {
        "__import__": __import__,
        "abs": abs,
        "all": all,
        "any": any,
        "bool": bool,
        "dict": dict,
        "enumerate": enumerate,
        "float": float,
        "int": int,
        "len": len,
        "list": list,
        "max": max,
        "min": min,
        "print": print,
        "range": range,
        "round": round,
        "set": set,
        "sorted": sorted,
        "str": str,
        "sum": sum,
        "tuple": tuple,
        "zip": zip,
    }
    namespace: dict[str, Any] = {
        "__builtins__": safe_builtins,
        "__name__": "__main__",
        "np": np,
        "pd": pd,
    }
    return namespace


def _python_script_to_dataframe(script_text: str) -> pd.DataFrame | None:
    text = (script_text or "").strip()
    if not text:
        return None
    lower = text.lower()
    if not any(token in lower for token in ("pd.dataframe", "dataframe", "numpy", "to_csv", "read_csv", "df =")):
        return None
    try:
        tree = ast.parse(text, mode="exec")
    except SyntaxError:
        return None
    namespace = _safe_python_globals()
    try:
        exec(compile(tree, "<rowfirst-script>", "exec"), namespace, namespace)
    except Exception:
        return None
    for value in namespace.values():
        if isinstance(value, pd.DataFrame) and not value.empty:
            return value.copy()
    df_candidates = [
        value for value in namespace.values() if hasattr(value, "to_csv") and hasattr(value, "columns")
    ]
    if df_candidates:
        candidate = df_candidates[0]
        if hasattr(candidate, "copy"):
            return candidate.copy()
    return None


def _analyze_ingested_safely(ingested: dict[str, Any], outcome_name: str | None = None) -> dict[str, Any]:
    guard_error = _group_size_error(ingested)
    if guard_error:
        return {"ok": False, "error": guard_error}
    variance_error = _zero_variance_error(ingested)
    if variance_error:
        return {"ok": False, "error": variance_error}
    try:
        return analyze_ingested(ingested, outcome_name=outcome_name)
    except (ValueError, TypeError, FloatingPointError) as exc:
        return {"ok": False, "error": str(exc) or exc.__class__.__name__}


def _rebuild_ingested_from_frame(frame: pd.DataFrame, factor: str, outcome: str) -> dict[str, Any]:
    grouped = frame[[factor, outcome]].copy()
    grouped[outcome] = pd.to_numeric(grouped[outcome], errors="coerce")
    grouped = grouped.dropna(subset=[factor, outcome])
    groups = []
    for group_name, group in grouped.groupby(factor, dropna=False)[outcome]:
        groups.append({"name": str(group_name), "values": [float(value) for value in group.tolist()]})
    return {"format": "labelled", "groups": groups}


def _nonparametric_for_result(result: dict[str, Any], ingested: dict[str, Any] | None = None) -> dict[str, Any]:
    groups = []
    if ingested and ingested.get("groups"):
        groups = ingested["groups"]
    elif result.get("groups"):
        groups = result["groups"]
    if len(groups) >= 3:
        values = [np.asarray(group.get("values", []), dtype=float) for group in groups]
        statistic, p_value = stats.kruskal(*values)
        return {"test": "kruskal-wallis", "statistic": float(statistic), "p": float(p_value), "isSignificant": bool(p_value < 0.05)}
    if len(groups) == 2:
        a = np.asarray(groups[0].get("values", []), dtype=float)
        b = np.asarray(groups[1].get("values", []), dtype=float)
        statistic, p_value = stats.mannwhitneyu(a, b, alternative="two-sided")
        return {"test": "mann-whitney", "statistic": float(statistic), "p": float(p_value), "isSignificant": bool(p_value < 0.05)}
    return {"test": "nonparametric", "p": 1.0, "isSignificant": False}


def _assumption_failure(engine: dict[str, Any]) -> bool:
    if not engine or not engine.get("ok"):
        return False
    results = engine.get("results") or [engine.get("result")]
    for result in results:
        assumptions = result.get("assumptions") or {}
        levene = assumptions.get("levene") or {}
        levene_p = levene.get("p") if isinstance(levene, dict) else None
        if levene_p is not None and levene_p < 0.05:
            return True
        shapiro_values = []
        for item in assumptions.get("shapiroWilk", []) or []:
            p_value = item.get("p")
            if p_value is not None:
                shapiro_values.append(float(p_value))
        if any(p < 0.05 for p in shapiro_values):
            return True
        if any(result.get("test") == "one-way anova" and isinstance(result.get("p"), (int, float)) and result["p"] < 0.05 for result in [result]):
            continue
    return False


def _assumption_violation(engine: dict[str, Any]) -> bool:
    if not engine or not engine.get("ok"):
        return False
    results = engine.get("results") or [engine.get("result")]
    for result in results:
        assumptions = result.get("assumptions") or {}
        levene = assumptions.get("levene") or {}
        levene_p = levene.get("p") if isinstance(levene, dict) else None
        if levene_p is not None and levene_p < 0.05:
            return True
        shapiro_values = []
        for item in assumptions.get("shapiroWilk", []) or []:
            p_value = item.get("p")
            if p_value is not None:
                shapiro_values.append(float(p_value))
        if any(p < 0.05 for p in shapiro_values):
            return True
    return False


def _cache_assumption_state(session: dict[str, Any], frame: pd.DataFrame | None, factor: str | None, metric: str | None) -> None:
    session["state"] = "AWAITING_ASSUMPTION_CHOICE"
    session["assumption_context"] = {
        "df": frame.copy() if isinstance(frame, pd.DataFrame) else None,
        "factor": factor,
        "metric": metric,
    }


def _save_dataset_artifacts(frame: pd.DataFrame, chat_id: int, prefix: str = "rowfirst") -> tuple[Path, Path]:
    tmp_dir = Path(tempfile.mkdtemp(prefix=f"rowfirst-{chat_id}-"))
    csv_path = tmp_dir / f"{prefix}.csv"
    xlsx_path = tmp_dir / f"{prefix}.xlsx"
    frame.to_csv(csv_path, index=False)
    frame.to_excel(xlsx_path, index=False)
    return csv_path, xlsx_path


def _remember_active_dataset(chat_id: int, frame: pd.DataFrame | None, *, state: str = "ACTIVE_DATASET") -> None:
    session = user_sessions.get(chat_id, {}) if isinstance(user_sessions.get(chat_id), dict) else {}
    if frame is None:
        session = {"state": "IDLE"}
    else:
        session["df"] = frame.copy()
        session["active_df"] = frame.copy()
        session["state"] = state
    user_sessions[chat_id] = session


def _is_global_reset_request(text: str) -> bool:
    normalized = re.sub(r"\s+", " ", str(text or "").strip())
    return bool(_GLOBAL_RESET_RE.fullmatch(normalized) or _GLOBAL_RESET_INTENT_RE.search(normalized))


def _handle_global_reset(
    bot: Any,
    message: Any,
    *,
    pending: dict[int, dict[str, str]] | None = None,
    pending_multivariate: dict[int, dict[str, Any]] | None = None,
    text_buffers: dict[int, dict[str, Any]] | None = None,
) -> bool:
    chat_id = int(getattr(message, "chat", None).id)
    if pending is not None:
        pending.pop(chat_id, None)
    if pending_multivariate is not None:
        pending_multivariate.pop(chat_id, None)
    pending_extracted.pop(chat_id, None)
    if text_buffers is not None:
        text_buffers.pop(chat_id, None)
    user_sessions[chat_id] = {"state": "IDLE"}
    try:
        bot.edit_message_reply_markup(chat_id=chat_id, message_id=message.message_id, reply_markup=None)
    except Exception:
        pass
    bot.reply_to(message, "🔄 Session reset. Active dataset and pending selections cleared. Send a new CSV, Excel, or PDF to begin.")
    return True


def _begin_ingestion(chat_id: int) -> bool:
    session = user_sessions.setdefault(chat_id, {})
    if session.get("ingesting_locked"):
        return False
    session["ingesting_locked"] = True
    user_sessions[chat_id] = session
    return True


def _finish_ingestion(chat_id: int) -> None:
    session = user_sessions.get(chat_id)
    if isinstance(session, dict):
        session.pop("ingesting_locked", None)
    user_sessions[chat_id] = session


def _normalize_copy_text(text: str) -> str:
    if not text:
        return text
    normalized = text.replace("\r\n", "\n").replace("\r", "\n").strip()
    lines: list[str] = []
    for line in normalized.split("\n"):
        stripped = line.strip()
        if not stripped:
            lines.append("")
            continue
        if stripped.startswith(("RESULTS", "BREAKDOWN", "QA", "Data Ingestion Card", "Detected Categorical Factors", "Detected Numeric Metrics", "Missing Values / Data Health")):
            lines.append(line)
            continue
        if stripped.startswith(("-", "•")):
            lines.append(line)
            continue
        lines.append(re.sub(r"\s*[-–—]\s+", ", ", line))
    normalized = "\n".join(lines)
    normalized = re.sub(r"\n\s*\n+", "\n\n", normalized)
    normalized = re.sub(r"[ \t]{2,}", " ", normalized)
    normalized = re.sub(r"\s+,\s*", ", ", normalized)
    return normalized.strip()


def _mark_message_processed(chat_id: int, message_id: int | None) -> bool:
    if message_id is None:
        return True
    cache_key = (int(chat_id), int(message_id))
    if cache_key in _INGESTION_DELIVERY_CACHE:
        return False
    _INGESTION_DELIVERY_CACHE.add(cache_key)
    if len(_INGESTION_DELIVERY_CACHE) > 2000:
        _INGESTION_DELIVERY_CACHE.clear()
    return True


def _explore_dataset_markup() -> Any:
    markup = types.InlineKeyboardMarkup(row_width=1)
    markup.add(types.InlineKeyboardButton("🔍 Explore Dataset Insights", callback_data="explore:dataset"))
    return _add_reset_button(markup)


def _add_reset_button(markup: Any) -> Any:
    markup.add(types.InlineKeyboardButton("🧹 Clear & Start Over", callback_data=_RESET_CALLBACK_DATA))
    return markup


def _explorer_mode_markup() -> Any:
    if types is None:
        return None
    markup = types.InlineKeyboardMarkup(row_width=1)
    markup.add(
        types.InlineKeyboardButton("🏆 Top & Bottom Performers", callback_data="explore:top_bottom"),
        types.InlineKeyboardButton("📊 Key Metrics & Distribution", callback_data="explore:key_metrics"),
        types.InlineKeyboardButton("🔗 Strongest Correlations", callback_data="explore:correlations"),
        types.InlineKeyboardButton("❓ Ask a Question", callback_data="explore:question"),
        types.InlineKeyboardButton("🔙 Back to Analysis", callback_data="explore:back"),
    )
    return _add_reset_button(markup)


def _explorer_back_markup() -> Any:
    if types is None:
        return None
    markup = types.InlineKeyboardMarkup(row_width=1)
    markup.add(types.InlineKeyboardButton("🔙 Back to Analysis", callback_data="explore:back"))
    return _add_reset_button(markup)


def _send_ingestion_card(bot: Any, message: Any, frame: pd.DataFrame, *, offer_download: bool = True) -> None:
    message_id = getattr(message, "message_id", None)
    if not _mark_message_processed(int(message.chat.id), message_id):
        return
    summary = _normalize_copy_text(_summarize_dataframe(frame))
    bot.reply_to(message, summary[:4000], reply_markup=_explore_dataset_markup())
    if offer_download:
        csv_path, xlsx_path = _save_dataset_artifacts(frame, int(message.chat.id))
        with open(csv_path, "rb") as csv_file:
            bot.send_document(message.chat.id, csv_file, caption="Clean CSV export")
        with open(xlsx_path, "rb") as xlsx_file:
            bot.send_document(message.chat.id, xlsx_file, caption="Styled Excel export")


def _output_selector_markup() -> Any:
    markup = types.InlineKeyboardMarkup(row_width=1)
    markup.add(
        types.InlineKeyboardButton("📄 Word Chapter 4 (.docx)", callback_data="output:docx"),
        types.InlineKeyboardButton("📑 Academic Report (.pdf)", callback_data="output:pdf"),
        types.InlineKeyboardButton("📊 Clean Processed Excel (.xlsx) / CSV", callback_data="output:excel"),
        types.InlineKeyboardButton("📦 Full Package (All Formats)", callback_data="output:full"),
    )
    return markup


def _table_confirmation_markup() -> Any:
    markup = types.InlineKeyboardMarkup(row_width=2)
    markup.add(
        types.InlineKeyboardButton("✅ Looks Accurate — Proceed", callback_data="extract:accept"),
        types.InlineKeyboardButton("🔄 Re-upload as CSV/Excel", callback_data="extract:retry"),
    )
    return markup


def _active_dataset_query_markup(frame: pd.DataFrame) -> Any:
    if frame is None or frame.empty:
        return None
    choices = [column for column in frame.columns if column and not _is_identifier_column(frame, column)]
    if not choices:
        return None
    markup = types.InlineKeyboardMarkup(row_width=1)
    for column in choices[:8]:
        markup.add(types.InlineKeyboardButton(str(column), callback_data=f"active_query:{column}"))
    return markup


def _singleton_triage_markup() -> Any:
    markup = types.InlineKeyboardMarkup(row_width=1)
    markup.add(
        types.InlineKeyboardButton("🚫 Exclude Singleton Group(s) & Continue", callback_data="singleton:exclude"),
        types.InlineKeyboardButton("🔄 Select Different Factor", callback_data="singleton:factor"),
        types.InlineKeyboardButton("📁 Cancel / Upload New Data", callback_data="singleton:cancel"),
    )
    return markup


def _assumption_markup() -> Any:
    markup = types.InlineKeyboardMarkup(row_width=1)
    markup.add(
        types.InlineKeyboardButton("Run Kruskal-Wallis / Mann-Whitney", callback_data="choice:kruskal"),
        types.InlineKeyboardButton("Proceed with Standard ANOVA", callback_data="choice:anova_override"),
    )
    return markup


def _manual_mapping_markup(frame: pd.DataFrame | None) -> Any:
    if frame is None or frame.empty:
        return None
    markup = types.InlineKeyboardMarkup(row_width=1)
    markup.add(
        types.InlineKeyboardButton("Select Factor (Independent)", callback_data="manualmap:factor"),
        types.InlineKeyboardButton("Select Metric (Dependent)", callback_data="manualmap:metric"),
    )
    return _add_reset_button(markup)


def _deliver_requested_outputs(bot: Any, call: Any, engine: dict[str, Any], chat_id: int, requested: str) -> None:
    session = user_sessions.get(chat_id, {})
    frame = None
    if isinstance(session, dict):
        frame = session.get("df")
    if frame is None:
        frame = _frame_from_engine(engine)
        try:
            bot.send_chat_action(chat_id, "upload_document")
        except Exception:
            pass
        if requested in {"pdf", "full"}:
            pdf_path = tmp_dir / "Rowfirst_Results.pdf"
            try:
                from chapter4 import write_pdf
                write_pdf(engine, pdf_path)
                with open(pdf_path, "rb") as pdf_file:
                    bot.send_document(chat_id, pdf_file, caption="Compiled by Rowfirst Engine — 100% Deterministic SciPy Execution (Zero LLM Calculation Drift)")
            except Exception:
                pass
        if requested in {"excel", "full"} and frame is not None:
            csv_path = tmp_dir / "rowfirst_clean.csv"
            xlsx_path = tmp_dir / "rowfirst_clean.xlsx"
            frame.to_csv(csv_path, index=False)
            frame.to_excel(xlsx_path, index=False)
            with open(csv_path, "rb") as csv_file:
                bot.send_document(chat_id, csv_file, caption="Clean CSV export")
            with open(xlsx_path, "rb") as xlsx_file:
                bot.send_document(chat_id, xlsx_file, caption="Clean Excel export")
    bot.answer_callback_query(call.id, "Your requested output is ready.")


def _detect_singleton_factor(frame: pd.DataFrame) -> str | None:
    for column in frame.columns:
        if _is_identifier_column(frame, column):
            continue
        series = frame[column]
        non_null = series.dropna()
        if non_null.empty or non_null.nunique() <= 1:
            continue
        if pd.api.types.is_numeric_dtype(series):
            unique_count = int(non_null.nunique(dropna=True))
            if unique_count > min(12, max(3, len(frame) // 10)):
                continue
        else:
            unique_count = int(non_null.nunique(dropna=True))
            row_count = len(frame)
            ratio = unique_count / max(1, row_count)
            if ratio > 0.5 and unique_count >= max(10, int(row_count * 0.2)):
                continue
        counts = non_null.value_counts(dropna=False)
        if (counts == 1).any():
            return str(column)
    return None


def _frame_from_engine(engine: dict[str, Any]) -> pd.DataFrame | None:
    ingested = engine.get("ingested") or {}
    if ingested.get("groups"):
        rows = []
        for group in ingested.get("groups"):
            for value in group.get("values") or []:
                rows.append({"group": str(group.get("name")), "value": float(value)})
        return pd.DataFrame(rows)
    if ingested.get("format") == "two-way" and ingested.get("rows"):
        return pd.DataFrame(ingested["rows"])
    return None


def _multivariate_file_route(path: Path) -> dict[str, Any] | None:
    if path.suffix.lower() not in {".csv", ".txt", ".tsv"}:
        return None
    try:
        raw = path.read_text(encoding="utf-8-sig", errors="replace")
    except OSError:
        return None
    return _multivariate_table_route(_read_delimited_frame(raw))


def _group_multivariate_frame(
    frame: pd.DataFrame,
    factor: str,
    outcome: str,
) -> list[dict[str, Any]]:
    selected = frame[[factor, outcome]].copy()
    selected[outcome] = pd.to_numeric(selected[outcome], errors="coerce")
    selected = selected.dropna(subset=[factor, outcome])
    grouped = selected.groupby(factor, dropna=False)[outcome].apply(list)
    return [
        {"name": str(name), "values": [float(value) for value in values]}
        for name, values in grouped.items()
        if values
    ]


def explain(engine: dict[str, Any]) -> str:
    """Only display verified engine text; do not ask an LLM to rewrite statistics."""
    if not engine.get("ok"):
        return engine.get("error", "I could not analyse that.")
    return engine["message"]


def _table_preview(table: dict[str, Any]) -> str:
    lines = ["Extracted table:"]
    lines.append(" | ".join(str(header) for header in table["headers"]))
    lines.append("-" * min(120, max(3, len(lines[-1]))))
    for row in table["rows"]:
        lines.append(" | ".join(str(value) for value in row))
    lines.append("Reply YES to analyse this table.")
    return "\n".join(lines)


def _engine_from_file(path: Path) -> dict[str, Any]:
    ingested = ingest_file(path)
    engine = _analyze_ingested_safely(ingested)
    engine["ingested"] = ingested
    engine["qa"] = quality_check(ingested)
    engine["breakdown"] = build_breakdown(engine)
    return engine


LOCAL_SUFFIXES = {".csv", ".txt", ".tsv", ".xlsx", ".xls", ".zip"}
GEMINI_SUFFIXES = {".pdf", ".docx", ".doc"}
GEMINI_FALLBACK = (
    "Add GEMINI_API_KEY in Secrets to read PDF/Word. You can still paste the table or send CSV."
)
# Prefer the newest generally available model, but keep a safe fallback chain for
# environments that do not yet expose the newest name.
DEFAULT_GEMINI_MODELS = [
    os.getenv("GEMINI_MODEL"),
    "gemini-2.5-pro",
    "gemini-2.5-flash",
]
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-pro")
GEMINI_UNAVAILABLE = (
    "I couldn’t read that image or document clearly. "
    "Please send a CSV or Excel file, or paste the table directly, and I’ll analyze it right away."
)
GEMINI_FAILED = (
    "I couldn’t extract a clean table from that image or document. "
    "Please send a CSV/Excel file or paste the table directly so I can continue with the analysis."
)
MIME_SUFFIXES = {
    "text/csv": ".csv",
    "application/csv": ".csv",
    "text/plain": ".txt",
    "text/tab-separated-values": ".tsv",
    "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet": ".xlsx",
    "application/vnd.ms-excel": ".xls",
    "application/zip": ".zip",
    "application/x-zip-compressed": ".zip",
    "application/pdf": ".pdf",
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document": ".docx",
    "application/msword": ".doc",
}


class GeminiExtractionError(RuntimeError):
    """An extraction failure whose safe message can be shown to users."""


class GeminiNoTableError(GeminiExtractionError):
    """Gemini returned no usable table."""


def _validate_extracted_table(table: Any) -> bool:
    """Reject malformed or low-value OCR output before it reaches the stats pipeline."""
    if not isinstance(table, dict):
        return False
    headers = table.get("headers")
    rows = table.get("rows")
    if not isinstance(headers, list) or not headers or len(headers) < 2:
        return False
    if not isinstance(rows, list) or not rows:
        return False
    expected_columns = len(headers)
    for row in rows:
        if not isinstance(row, list):
            return False
        if len(row) != expected_columns:
            return False
        if not any(str(cell).strip() for cell in row if cell is not None):
            return False
    return True


def _upload_suffix(filename: str, mime_type: str) -> str:
    suffix = Path(filename).suffix.lower()
    if suffix in LOCAL_SUFFIXES or suffix in GEMINI_SUFFIXES:
        return suffix
    return MIME_SUFFIXES.get((mime_type or "").split(";", 1)[0].strip().lower(), "")


def _csv_text_to_table(raw: str) -> dict[str, Any] | None:
    """Normalize OCR output into a table dict that the bot can safely analyze."""
    cleaned = (raw or "").strip()
    if not cleaned:
        return None
    cleaned = re.sub(r"^```(?:csv|json|text)?\s*|\s*```$", "", cleaned, flags=re.I | re.S).strip()
    if not cleaned or cleaned.upper() in {"NO_TABLE", "NONE", "N/A"}:
        return None
    if cleaned.startswith("{"):
        try:
            payload = json.loads(cleaned)
            if _validate_extracted_table(payload):
                return {"headers": payload["headers"], "rows": payload["rows"]}
        except json.JSONDecodeError:
            pass
    try:
        frame = pd.read_csv(io.StringIO(cleaned), sep=None, engine="python")
    except Exception:
        frame = None
    if frame is None or frame.empty:
        lines = [line.strip() for line in cleaned.splitlines() if line.strip()]
        if not lines or "|" not in cleaned:
            return None
        try:
            rows = [line.split("|") for line in lines if "|" in line]
            if not rows:
                return None
            header = [cell.strip() for cell in rows[0]]
            data_rows = [[cell.strip() for cell in row] for row in rows[1:]]
            if len(header) < 2 or not data_rows:
                return None
            if all(len(row) == len(header) for row in data_rows):
                frame = pd.DataFrame(data_rows, columns=header)
        except Exception:
            return None
    if frame is None or frame.empty:
        return None
    frame = sanitize_incoming_dataframe(frame)
    if frame is None or frame.empty or frame.shape[1] < 2:
        return None
    headers = [str(column).strip() for column in frame.columns.tolist()]
    rows = frame.fillna("").astype(str).values.tolist()
    if not _validate_extracted_table({"headers": headers, "rows": rows}):
        return None
    return {"headers": headers, "rows": rows}


def _gemini_extract(path: Path) -> dict[str, Any]:
    key = os.getenv("GEMINI_API_KEY")
    if not key:
        raise GeminiExtractionError(GEMINI_FALLBACK)
    mime = {
        ".pdf": "application/pdf",
        ".docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        ".doc": "application/msword",
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".png": "image/png",
        ".webp": "image/webp",
    }.get(path.suffix.lower(), "application/octet-stream")
    prompt = (
        "You are Rowfirst's OCR ingestion component. Read the supplied image or document and "
        "recover only the tabular data. Return exactly one JSON object with this shape: "
        '{"headers":["column"],"rows":[["value"]]}. '
        "Preserve headers, signs, decimal points, missing cells, and values exactly as observed. "
        "You may correct obvious OCR character noise, but do not infer missing values, reorder rows, "
        "summarize, calculate, classify, or interpret anything. Use an empty string for an unreadable "
        "cell. If no table is present, return {\"headers\":[],\"rows\":[]}."
    )
    last_error: Exception | None = None
    for model_name in dict.fromkeys(model for model in DEFAULT_GEMINI_MODELS if model):
        try:
            import google.generativeai as genai

            genai.configure(api_key=key)
            model = genai.GenerativeModel(model_name)
            contents = [prompt, {"mime_type": mime, "data": path.read_bytes()}]
            try:
                response = model.generate_content(
                    contents,
                    generation_config={
                        "response_mime_type": "application/json",
                        "temperature": 0,
                    },
                )
            except TypeError:
                response = model.generate_content(contents)
            raw = (getattr(response, "text", "") or "").strip()
            payload = _csv_text_to_table(raw)
            if payload is None:
                raise GeminiNoTableError(GEMINI_FAILED)
            return payload
        except GeminiExtractionError as exc:
            last_error = exc
            continue
        except Exception as exc:
            last_error = exc
            print(f"Gemini extraction failed with model {model_name}: {type(exc).__name__}", flush=True)
            continue
    if isinstance(last_error, GeminiExtractionError):
        raise last_error
    if isinstance(last_error, Exception) and type(last_error).__name__ == "NotFound":
        raise GeminiExtractionError(GEMINI_UNAVAILABLE) from last_error
    raise GeminiExtractionError(GEMINI_FAILED)


def _gemini_response_text(response: Any) -> str:
    text = getattr(response, "text", "") or ""
    if text:
        return str(text).strip()
    for candidate in getattr(response, "candidates", []) or []:
        content = getattr(candidate, "content", None)
        for part in getattr(content, "parts", []) or []:
            part_text = getattr(part, "text", "") or ""
            if part_text:
                return str(part_text).strip()
    return ""


def _read_only_explanation_request(text: str) -> bool:
    normalized = str(text or "").strip()
    if not normalized or not re.search(
        r"(?i)\b(?:why|what does|what do|explain|interpret|meaning|summarize|summary|describe|assumption|significant)\b|\?",
        normalized,
    ):
        return False
    if re.search(r"(?i)\bwhat does\b.*\bmean\b", normalized):
        return True
    return not re.search(
        r"(?i)\b(?:compare|regression|correlation|anova|t[- ]?test|mann|kruskal|calculate|average|mean|median|highest|lowest|top|predict|run|perform|test)\b",
        normalized,
    )


def _gemini_read_only_explanation(chat_id: int, question: str) -> str | None:
    key = os.getenv("GEMINI_API_KEY")
    if not key:
        return None
    session = user_sessions.get(chat_id, {})
    frame = session.get("df") if isinstance(session, dict) else None
    engine = last_engine.get(chat_id)
    if not isinstance(frame, pd.DataFrame) and not isinstance(engine, dict):
        return None
    context: dict[str, Any] = {"question": str(question).strip()}
    if isinstance(frame, pd.DataFrame):
        context["dataset"] = {
            "rows": int(frame.shape[0]),
            "columns": [str(column) for column in frame.columns],
            "dtypes": {str(column): str(dtype) for column, dtype in frame.dtypes.items()},
            "missing_values": {str(column): int(value) for column, value in frame.isna().sum().items()},
            "sample": frame.head(5).fillna("").astype(str).to_dict(orient="records"),
        }
    if isinstance(engine, dict):
        context["engine_results"] = engine
    context_text = json.dumps(context, default=str, ensure_ascii=True)[:18000]
    prompt = (
        "You are Rowfirst's read-only conversational spokesperson. Answer the user's question using "
        "only the supplied dataset metadata, sample, and verified engine results. Explain existing "
        "results in plain language, including reported assumptions or significance when present. "
        "Never calculate, estimate, transform, or invent statistics; never propose a new test; never "
        "edit the dataset or engine state. If the supplied context does not answer the question, say "
        "that clearly and ask the user to request a supported analysis. Return concise plain text.\n\n"
        f"Context JSON:\n{context_text}"
    )
    try:
        import google.generativeai as genai

        genai.configure(api_key=key)
        model = genai.GenerativeModel(GEMINI_MODEL)
        try:
            response = model.generate_content(
                prompt,
                generation_config={"temperature": 0.2, "max_output_tokens": 500},
            )
        except TypeError:
            response = model.generate_content(prompt)
        answer = _gemini_response_text(response)
        return answer[:4000] if answer else None
    except Exception as exc:
        print(f"Gemini explanation failed: {type(exc).__name__}", flush=True)
        return None


def _table_as_csv(table: dict[str, Any]) -> str:
    output = io.StringIO()
    writer = csv.writer(output, lineterminator="\n")
    writer.writerow(table["headers"])
    writer.writerows(table["rows"])
    return output.getvalue()


def _send_analysis(
    bot: Any,
    message: Any,
    engine: dict[str, Any],
    *,
    ask_for_document: bool = True,
) -> None:
    if not engine.get("ok"):
        bot.reply_to(message, engine.get("question") or engine.get("error", "I could not analyse that.")[:4000])
        return
    try:
        bot.send_chat_action(message.chat.id, "typing")
    except Exception:
        pass
    results = engine.get("results") or [engine.get("result")]
    results = [result for result in results if result]
    result_lines = ["Outcome | Test | Statistic | p | Decision", *[_summary_line(result) for result in results]]
    breakdown = engine.get("breakdown") or build_breakdown(engine)
    qa = engine.get("qa") or {}
    qa_lines = qa.get("warnings", []) + qa.get("errors", [])
    qa_text = "\n".join(f"- {item}" for item in qa_lines) if qa_lines else "none"
    if "charts" not in engine:
        try:
            bot.send_chat_action(message.chat.id, "upload_document")
            engine["charts"] = make_charts(
                engine,
                tempfile.mkdtemp(prefix="rowfirst-charts-"),
            )
        except Exception as exc:
            print(f"Chart generation skipped: {type(exc).__name__}", flush=True)
            engine["charts"] = []
    _reply_block(bot, message, "RESULTS\n" + "\n".join(result_lines))
    _reply_block(bot, message, "BREAKDOWN\n" + breakdown)
    _reply_block(bot, message, "QA\n" + qa_text)
    for chart in engine.get("charts", [])[:6]:
        chart_path = Path(chart["path"])
        if not chart_path.exists():
            continue
        try:
            with chart_path.open("rb") as image_file:
                bot.send_photo(message.chat.id, image_file, caption=chart.get("caption", ""))
        except Exception as exc:
            print(f"Chart delivery skipped: {type(exc).__name__}", flush=True)
    if ask_for_document:
        bot.reply_to(message, "Want a Results Document (Word)? Reply YES")


def _reply_block(bot: Any, message: Any, text: str) -> None:
    text = _normalize_copy_text(text)
    if len(text) > 3900:
        text = text[:3890].rstrip() + "\n…"
    bot.reply_to(message, text)


def _send_error(bot: Any, message: Any, exc: Exception, prefix: str = "Could not read your file") -> None:
    detail = str(exc).strip() or exc.__class__.__name__
    text = detail if isinstance(exc, GeminiExtractionError) else f"{prefix}: {detail}"
    try:
        bot.reply_to(message, text[:4000])
    except Exception as reply_exc:
        print(f"{text}; Telegram error while sending it: {reply_exc}")


def _is_small_talk(text: str) -> bool:
    cleaned = re.sub(r"[^a-z ]", "", (text or "").lower()).strip()
    return cleaned in {"hi", "hello", "hey", "thanks", "thank you", "thx"}


def _analysis_request(text: str) -> tuple[dict[str, Any], dict[str, Any]]:
    topic_match = re.search(r"(?im)^\s*topic\s*:\s*(.+?)\s*$", text or "")
    outcome_match = re.search(
        r"(?im)^\s*(?:outcome|dependent\s+variable|dependent_variable|response|measure|metric)\s*:\s*(.+?)\s*$",
        text or "",
    )
    hypothesis_matches = re.finditer(
        r"(?im)^\s*(h(?:0|o)\d*)\s*:\s*(.+?)\s*$",
        text or "",
    )
    hypotheses = [
        {"label": match.group(1), "text": match.group(2).strip()}
        for match in hypothesis_matches
    ]
    data_lines = []
    for line in (text or "").splitlines():
        if re.match(r"(?i)^\s*(topic|outcome|dependent\s+variable|dependent_variable|response|measure|metric|h(?:0|o)\d*)\s*:", line):
            continue
        data_lines.append(line)
    metadata: dict[str, Any] = {
        "topic": topic_match.group(1).strip() if topic_match else "",
        "hypotheses": hypotheses,
        "discuss": bool(re.search(r"\bdiscuss\b", text or "", flags=re.I)),
    }
    request = {"text": "\n".join(data_lines).strip()}
    if outcome_match:
        request["outcome"] = outcome_match.group(1).strip()
    return request, metadata


def _run_analysis(text: str) -> dict[str, Any]:
    request, metadata = _analysis_request(text)
    try:
        ingested = ingest_text(request["text"])
    except (ValueError, TypeError):
        ingested = None
    if ingested:
        guard_error = _group_size_error(ingested)
        if guard_error:
            return {"ok": False, "error": guard_error}
    engine = handle_analyze(request)
    if engine.get("ok"):
        if metadata["topic"]:
            engine["topic"] = metadata["topic"]
        if metadata["hypotheses"]:
            engine["hypotheses"] = metadata["hypotheses"]
        if metadata["discuss"]:
            engine["discuss"] = True
    return engine


def _column_lookup(frame: pd.DataFrame, query: str) -> str | None:
    if frame is None or query is None:
        return None
    normalized_query = re.sub(r"[^a-z0-9]+", "", str(query).lower())
    for column in frame.columns:
        normalized_column = re.sub(r"[^a-z0-9]+", "", str(column).lower())
        if normalized_column == normalized_query or normalized_query in normalized_column or normalized_column in normalized_query:
            return str(column)
    return None


def _evaluate_conversational_query(chat_id: int, text: str) -> dict[str, Any] | None:
    session = user_sessions.get(chat_id, {})
    frame = session.get("df") if isinstance(session, dict) else None
    if frame is None or frame.empty:
        return None
    lower = (text or "").strip()
    if not lower:
        return None

    regression_match = re.search(r"(?:run|do|perform)?\s*regression\s+(?:between|on)\s+(.+?)\s+(?:and|vs|versus)\s+(.+?)(?:\s*$|\?|\.)", lower, flags=re.I)
    if regression_match:
        left = _column_lookup(frame, regression_match.group(1))
        right = _column_lookup(frame, regression_match.group(2))
        if left and right:
            values = frame[[left, right]].dropna()
            if len(values) >= 3:
                return {"kind": "regression", "x": left, "y": right, "frame": values}

    compare_match = re.search(r"compare\s+(.+?)\s+across\s+(.+?)(?:\s*$|\?|\.)", lower, flags=re.I)
    if compare_match:
        metric = _column_lookup(frame, compare_match.group(1))
        factor = _column_lookup(frame, compare_match.group(2))
        if metric and factor:
            values = frame[[factor, metric]].dropna()
            if len(values) >= 4:
                return {"kind": "compare", "factor": factor, "metric": metric, "frame": values}

    test_match = re.search(r"test\s+(.+?)\s+on\s+(.+?)(?:\s*$|\?|\.)", lower, flags=re.I)
    if test_match:
        factor = _column_lookup(frame, test_match.group(1))
        metric = _column_lookup(frame, test_match.group(2))
        if factor and metric:
            values = frame[[factor, metric]].dropna()
            if len(values) >= 4:
                return {"kind": "compare", "factor": factor, "metric": metric, "frame": values}

    return None


def _defense_qa(engine: dict[str, Any]) -> str:
    results = engine.get("results") or [engine.get("result")]
    results = [result for result in results if result]
    statistics = []
    p_values = []
    for result in results:
        test = result.get("test")
        if test in {"student-t", "welch-t", "paired-t"}:
            statistics.append(f"t={result['t']:.12g}")
            p_values.append(f"p={result['p']:.12g}")
        elif test == "one-way anova":
            statistics.append(f"F={result['F']:.12g}")
            p_values.append(f"p={result['p']:.12g}")
        elif test == "two-way anova":
            for effect in result.get("effects", []):
                statistics.append(f"F={effect['F']:.12g}")
                p_values.append(f"p={effect['p']:.12g}")
        elif "p" in result:
            p_values.append(f"p={result['p']:.12g}")
    stat_text = ", ".join(statistics) or "No t or F statistic was reported by the engine."
    p_text = ", ".join(p_values) or "No p-value was reported by the engine."
    decision = "significant at α = .05" if any(result.get("isSignificant") for result in results) else "not significant at α = .05"
    return "\n".join([
        "Defense Q&A",
        f"1. What statistic did the engine report?\n{stat_text}.",
        f"2. What exact p-value did it report?\n{p_text}.",
        f"3. Was the result significant at 5%?\n{decision}; {p_text}.",
        f"4. What evidence should be quoted?\n{stat_text}; {p_text}.",
        f"5. What does the result support?\nThe engine supports only the {stat_text} and {p_text} reported above.",
    ])


def _acquire_polling_lock(token: str):
    """Allow only one bot process to poll a given Telegram token."""
    import fcntl

    lock_id = hashlib.sha256(token.encode("utf-8")).hexdigest()[:24]
    lock_path = Path(tempfile.gettempdir()) / f"rowfirst-telegram-{lock_id}.lock"
    lock_file = lock_path.open("w")
    try:
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        lock_file.close()
        return None
    return lock_file


def _validate_telegram_bot(bot: Any, token: str) -> None:
    if not re.fullmatch(r"\d+:[A-Za-z0-9_-]+", token):
        raise SystemExit(
            "TELEGRAM_BOT_TOKEN is malformed. Copy the current token from @BotFather "
            "into Render without quotes or whitespace."
        )
    try:
        bot.get_me()
    except Exception as exc:
        if getattr(exc, "error_code", None) == 401 or getattr(exc, "status_code", None) == 401:
            raise SystemExit(
                "Telegram rejected TELEGRAM_BOT_TOKEN (401 Unauthorized). "
                "Generate a new token with @BotFather, update the Render secret, and redeploy."
            ) from exc
        raise SystemExit(
            f"Telegram token preflight failed with {type(exc).__name__}. "
            "Check Render networking and Telegram availability."
        ) from exc


def _create_telegram_bot(token: str) -> Any:
    try:
        return telebot.TeleBot(token)
    except (TypeError, ValueError) as exc:
        raise SystemExit(
            "TELEGRAM_BOT_TOKEN is malformed. Copy the current token from @BotFather "
            "into Render without quotes or whitespace."
        ) from exc


def _summary_line(result: dict[str, Any]) -> str:
    test = result.get("test", "")
    outcome = result.get("parameter") or result.get("outcome") or "Measured outcome"
    if test == "one-way anova":
        statistic = f"F({result['dfb']},{result['dfw']})={result['F']:.4f}"
    elif test in {"student-t", "welch-t", "paired-t"}:
        statistic = f"t({result['df']:.0f})={result['t']:.4f}"
    elif test == "simple linear regression":
        statistic = f"r={result['r']:.4f}; r²={result['rSquared']:.4f}"
    elif test == "two-way anova":
        statistic = "; ".join(
            f"{effect['effect']} F={effect['F']:.4f}" for effect in result.get("effects", [])
        ) or "multiple effects"
    elif test in {"pearson", "spearman"}:
        statistic = f"r={result['r']:.4f}"
    elif test == "fisher-exact":
        statistic = f"Fisher exact; OR={result['oddsRatio']:.4f}"
    elif test == "chi-square":
        statistic = f"χ²({result['df']})={result['chi2']:.4f}"
    else:
        statistic = "not reported"
    if test == "two-way anova":
        p = "; ".join(
            f"{effect['effect']} {_p(effect['p'])}" for effect in result.get("effects", [])
        ) or "not reported"
        decision = ", ".join(
            effect["effect"] for effect in result.get("effects", []) if effect.get("isSignificant")
        ) or "not significant"
    else:
        p = _p(result["p"]) if "p" in result else "not reported"
        decision = "significant" if result.get("isSignificant") else "not significant"
    return f"{outcome} | {test} | {statistic} | {p} | {decision}"


def _p(value: Any) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "not reported"
    if number < 0.001:
        return "p < .001"
    return f"p = {number:.4f}" if number < 1 else f"p = {number:.4f}"


def main() -> None:
    token = (os.getenv("TELEGRAM_BOT_TOKEN") or "").strip()
    if not token:
        raise SystemExit("Set TELEGRAM_BOT_TOKEN in Secrets.")
    if telebot is None:
        raise SystemExit("Install pyTelegramBotAPI from requirements.txt.")
    bot = _create_telegram_bot(token)
    _validate_telegram_bot(bot, token)
    polling_lock = _acquire_polling_lock(token)
    if polling_lock is None:
        print("Telegram bot already polling this token; exiting.", flush=True)
        return
    pending: dict[int, dict[str, str]] = {}
    pending_multivariate: dict[int, dict[str, Any]] = {}
    user_sessions: dict[int, dict[str, Any]] = {}
    last_engine: dict[int, dict[str, Any]] = {}
    globals()["user_sessions"] = user_sessions
    globals()["last_engine"] = last_engine
    text_buffers: dict[int, dict[str, Any]] = {}
    text_buffer_lock = threading.Lock()
    text_debounce_seconds = 1.5
    text_chunk_cooldown_seconds = 2.0

    @bot.message_handler(commands=["start", "help"])
    def start(message: Any) -> None:
        try:
            bot.reply_to(
                message,
                "Hey — I’m Rowfirst. Send a table (paste, csv, excel, photo) and I’ll run the test.\n"
                "I’ll ask if the groups aren’t clear.",
            )
        except Exception as exc:
            _send_error(bot, message, exc, "Could not send the welcome message")

    @bot.message_handler(commands=["cancel"])
    def cancel(message: Any) -> None:
        _handle_global_reset(
            bot,
            message,
            pending=pending,
            pending_multivariate=pending_multivariate,
            text_buffers=text_buffers,
        )

    def _session_actions_markup() -> Any:
        markup = types.InlineKeyboardMarkup(row_width=1)
        markup.add(
            types.InlineKeyboardButton("🔄 Test Another Variable", callback_data="session:rerun"),
            types.InlineKeyboardButton("📁 Clear & New Dataset", callback_data="session:clear"),
        )
        return markup

    def send_results_document(message: Any) -> bool:
        engine = last_engine.get(message.chat.id)
        if not engine or not engine.get("ok"):
            bot.reply_to(message, "Send a table first.")
            return False
        try:
            bot.send_chat_action(message.chat.id, "upload_document")
            with tempfile.TemporaryDirectory(prefix="rowfirst-results-") as tmp:
                docx_path = Path(tmp) / "Rowfirst_Results.docx"
                write_docx(engine, docx_path)
                with open(docx_path, "rb") as document_file:
                    bot.send_document(
                        message.chat.id,
                        document_file,
                        caption="Rowfirst_Results.docx — built from the last engine JSON.",
                    )
            if message.chat.id in user_sessions:
                bot.reply_to(
                    message,
                    "Analysis complete! What would you like to do next?",
                    reply_markup=_session_actions_markup(),
                )
            return True
        except Exception as exc:
            _send_error(bot, message, exc, "Could not build the Results Document")
            return False

    def send_defense(message: Any) -> None:
        engine = last_engine.get(message.chat.id)
        if not engine or not engine.get("ok"):
            bot.reply_to(message, "Send a table first.")
            return
        _reply_block(bot, message, _defense_qa(engine))

    @bot.message_handler(commands=["results"])
    def results_document(message: Any) -> None:
        send_results_document(message)

    @bot.message_handler(commands=["full"])
    def full(message: Any) -> None:
        engine = last_engine.get(message.chat.id)
        if not engine or not engine.get("ok"):
            bot.reply_to(message, "Send a table first.")
            return
        _reply_block(
            bot,
            message,
            "RESULTS\n"
            + "\n".join(
                ["Outcome | Test | Statistic | p | Decision"]
                + [_summary_line(result) for result in (engine.get("results") or [engine.get("result")]) if result]
            ),
        )

    def _choice_markup(prefix: str, choices: list[str], *, include_metadata_toggle: bool = False) -> Any:
        markup = types.InlineKeyboardMarkup(row_width=2)
        markup.add(*[
            types.InlineKeyboardButton(
                text=str(choice)[:50],
                callback_data=f"{prefix}:{choice}",
            )
            for choice in choices
        ])
        if include_metadata_toggle:
            markup.add(types.InlineKeyboardButton("+ Show Excluded / Metadata Columns", callback_data="meta:toggle"))
        return _add_reset_button(markup)

    def _metadata_choice_markup(frame: pd.DataFrame, prefix: str, *, include_excluded: bool = False) -> Any:
        options = [column for column in frame.columns if _is_identifier_column(frame, column) or _is_temporal_or_datetime_column(frame, column)]
        if not options:
            return _choice_markup(prefix, [column for column in frame.columns if column and not _is_identifier_column(frame, column)], include_metadata_toggle=False)
        choices = [column for column in frame.columns if column and not _is_identifier_column(frame, column)]
        if include_excluded:
            choices = [f"{column} (⚠️ High Cardinality / ID)" if _is_identifier_column(frame, column) or _is_temporal_or_datetime_column(frame, column) else column for column in frame.columns]
        return _choice_markup(prefix, choices, include_metadata_toggle=True)

    def _prompt_multivariate(
        message: Any,
        route: dict[str, Any],
        *,
        cache_session: bool = True,
    ) -> None:
        if cache_session:
            user_sessions[message.chat.id] = {"df": route["frame"]}
        pending_multivariate[message.chat.id] = {
            "frame": route["frame"],
            "factors": route["factors"],
            "outcomes": route["outcomes"],
        }
        bot.reply_to(
            message,
            "Step 1/2: Select the Grouping Factor (Independent Variable)",
            reply_markup=_choice_markup("factor", route["factors"]),
        )

    @bot.callback_query_handler(
        func=lambda call: (getattr(call, "data", "") or "").startswith("session:")
    )
    def on_session_selection(call: Any) -> None:
        chat_id = call.message.chat.id
        callback_data = call.data or ""
        bot.answer_callback_query(call.id)
        if callback_data == "session:rerun":
            session = user_sessions.get(chat_id)
            if not session or "df" not in session:
                bot.reply_to(call.message, "No active dataset is loaded. Please paste or upload a dataset to continue.")
                return
            route = _multivariate_table_route(session.get("df"))
            if not route or route.get("kind") != "multivariate":
                bot.reply_to(call.message, "The current dataset is not in a multivariate route. Please paste or upload a fresh dataset.")
                return
            _prompt_multivariate(call.message, route, cache_session=False)
            return
        if callback_data == "session:clear":
            user_sessions.pop(chat_id, None)
            pending_multivariate.pop(chat_id, None)
            bot.reply_to(call.message, "Session cleared. Ready for your next dataset!")
            return
        bot.answer_callback_query(call.id, "Invalid session action.", show_alert=True)

    @bot.callback_query_handler(
        func=lambda call: (getattr(call, "data", "") or "") == _RESET_CALLBACK_DATA
    )
    def on_cancel_session(call: Any) -> None:
        bot.answer_callback_query(call.id)
        _handle_global_reset(
            bot,
            call.message,
            pending=pending,
            pending_multivariate=pending_multivariate,
            text_buffers=text_buffers,
        )

    @bot.callback_query_handler(func=lambda call: (getattr(call, "data", "") or "").startswith("singleton:"))
    def on_singleton_triage(call: Any) -> None:
        chat_id = call.message.chat.id
        callback_data = call.data or ""
        session = user_sessions.get(chat_id)
        if not session or "df" not in session:
            bot.answer_callback_query(call.id, "No active dataset is loaded. Please upload or paste a new dataset to continue.", show_alert=True)
            return
        frame = session["df"].copy()
        if callback_data == "singleton:exclude":
            factor = session.get("singleton_factor")
            if not factor or factor not in frame.columns:
                bot.answer_callback_query(call.id, "No active factor was found for singleton triage.", show_alert=True)
                return
            singleton_names = {item["name"] for item in _singleton_groups_for_frame(frame, factor)}
            keep = [name for name, _ in frame[factor].value_counts(dropna=False).items() if name not in singleton_names]
            filtered = frame[frame[factor].isin(keep)].copy() if keep else frame.iloc[0:0].copy()
            session["df"] = filtered
            user_sessions[chat_id] = session
            route = _multivariate_table_route(filtered)
            if route and route.get("kind") == "multivariate":
                _prompt_multivariate(call.message, route, cache_session=False)
            else:
                bot.reply_to(call.message, "The dataset had no valid multivariate route after excluding singleton groups.")
            bot.answer_callback_query(call.id)
            return
        if callback_data == "singleton:factor":
            available = [col for col in frame.columns if frame[col].dropna().nunique() > 1]
            bot.reply_to(call.message, "Choose a different factor to continue.", reply_markup=_choice_markup("factor", available))
            bot.answer_callback_query(call.id)
            return
        if callback_data == "singleton:cancel":
            user_sessions.pop(chat_id, None)
            pending_multivariate.pop(chat_id, None)
            bot.reply_to(call.message, "Upload a new dataset or paste fresh data to continue.")
            bot.answer_callback_query(call.id)
            return
        bot.answer_callback_query(call.id, "Unknown singleton option.", show_alert=True)

    @bot.callback_query_handler(func=lambda call: (getattr(call, "data", "") or "").startswith(("assumption:", "choice:", "meta:")))
    def on_assumption_decision(call: Any) -> None:
        chat_id = call.message.chat.id
        callback_data = call.data or ""
        sess = user_sessions.get(chat_id, {})
        bot.answer_callback_query(call.id)
        try:
            bot.edit_message_reply_markup(chat_id=chat_id, message_id=call.message.message_id, reply_markup=None)
        except Exception:
            pass
        if callback_data == "meta:toggle":
            state = sess.get("assumption_context", {}) if isinstance(sess, dict) else {}
            frame = state.get("df") or sess.get("df")
            if frame is not None and not frame.empty:
                options = [column for column in frame.columns if column and (not _is_identifier_column(frame, column) or _is_temporal_or_datetime_column(frame, column))]
                bot.reply_to(call.message, "Excluded / metadata columns are marked explicitly below:", reply_markup=_choice_markup("factor", [f"{column} (⚠️ High Cardinality / ID)" if _is_identifier_column(frame, column) else column for column in options], include_metadata_toggle=False))
            return
        if callback_data in {"assumption:nonparametric", "choice:kruskal"}:
            engine = last_engine.get(chat_id)
            if not engine:
                bot.reply_to(call.message, "No test result is active for this session.")
                return
            result = engine.get("result") or (engine.get("results") or [{}])[0]
            ingested = engine.get("ingested") or {}
            alt = _nonparametric_for_result(result, ingested)
            sess["state"] = "NONPARAMETRIC_FALLBACK"
            user_sessions[chat_id] = sess
            p_value = alt.get("p", 1.0)
            statistic = alt.get("statistic", 0.0)
            decision = "significant" if p_value < 0.05 else "not significant"
            reply = (
                "| Test | Statistic | p | Decision |\n"
                "| --- | ---: | ---: | --- |\n"
                f"| {alt.get('test', 'Kruskal-Wallis / Mann-Whitney')} | {float(statistic):.4f} | {_p(p_value)} | {decision} |\n\n"
                f"The non-parametric check indicates {decision} evidence at α = .05."
            )
            bot.reply_to(call.message, reply)
            return
        if callback_data in {"assumption:standard", "choice:anova_override"}:
            sess["state"] = "ANOVA_OVERRIDE"
            user_sessions[chat_id] = sess
            bot.reply_to(call.message, "Proceeding with the standard test. The original SciPy calculation remains the source of truth.")
            return
        bot.answer_callback_query(call.id, "Unknown assumption decision.", show_alert=True)

    @bot.callback_query_handler(func=lambda call: (getattr(call, "data", "") or "").startswith("manualmap:"))
    def on_manual_mapping(call: Any) -> None:
        chat_id = call.message.chat.id
        session = user_sessions.get(chat_id)
        if not session or session.get("df") is None:
            bot.answer_callback_query(call.id, "No active dataset is loaded. Please upload or paste one first.", show_alert=True)
            return
        frame = session["df"]
        callback_data = call.data or ""
        bot.answer_callback_query(call.id)
        if callback_data == "manualmap:factor":
            options = [column for column in frame.columns if not _is_metadata_only_column(frame, column)]
            bot.reply_to(call.message, "Select the Grouping Factor (Independent Variable):", reply_markup=_choice_markup("factor", options))
            return
        if callback_data == "manualmap:metric":
            options = [column for column in frame.columns if not _is_metadata_only_column(frame, column)]
            bot.reply_to(call.message, "Select the Outcome Metric (Dependent Variable):", reply_markup=_choice_markup("outcome", options))
            return
        bot.answer_callback_query(call.id, "Unknown manual-mapping selection.", show_alert=True)

    @bot.callback_query_handler(func=lambda call: (getattr(call, "data", "") or "").startswith("explore:"))
    def on_explore_dataset(call: Any) -> None:
        chat_id = call.message.chat.id
        session = user_sessions.get(chat_id, {})
        frame = session.get("df") if isinstance(session, dict) else None
        bot.answer_callback_query(call.id)
        if frame is None or frame.empty:
            bot.answer_callback_query(call.id, "No active dataset is loaded.", show_alert=True)
            return
        callback_data = call.data or ""
        if callback_data == "explore:back":
            session = user_sessions.get(chat_id, {}) if isinstance(user_sessions.get(chat_id), dict) else {}
            session["state"] = "ACTIVE_DATASET"
            session.pop("explore_prompt", None)
            user_sessions[chat_id] = session
            bot.reply_to(call.message, "Back in the main analysis flow. You can continue with standard statistics or upload another dataset.")
            return
        if callback_data == "explore:dataset":
            session["state"] = "EXPLORER_MODE"
            user_sessions[chat_id] = {**session, "explore_prompt": False}
            bot.reply_to(
                call.message,
                "Explore mode is active. Choose an insight or ask me a direct question about the current dataset.",
                reply_markup=_explorer_mode_markup(),
            )
            return
        if callback_data == "explore:question":
            session["state"] = "EXPLORER_MODE"
            user_sessions[chat_id] = {**session, "explore_prompt": True}
            bot.reply_to(
                call.message,
                "Type a question like 'highest sales by region' or 'average moisture'.",
                reply_markup=_explorer_back_markup(),
            )
            return
        if callback_data == "explore:top_bottom":
            try:
                summary = run_explorer_action(frame, "top_bottom")
            except Exception as exc:
                summary = str(exc)
            session["state"] = "EXPLORER_MODE"
            user_sessions[chat_id] = session
            bot.reply_to(call.message, summary, reply_markup=_explorer_back_markup())
            return
        if callback_data == "explore:key_metrics":
            try:
                summary = run_explorer_action(frame, "key_metrics")
            except Exception as exc:
                summary = str(exc)
            session["state"] = "EXPLORER_MODE"
            user_sessions[chat_id] = session
            bot.reply_to(call.message, summary, reply_markup=_explorer_back_markup())
            return
        if callback_data == "explore:correlations":
            try:
                summary = run_explorer_action(frame, "correlations")
            except Exception as exc:
                summary = str(exc)
            session["state"] = "EXPLORER_MODE"
            user_sessions[chat_id] = session
            bot.reply_to(call.message, summary, reply_markup=_explorer_back_markup())
            return
        session["state"] = "EXPLORER_MODE"
        user_sessions[chat_id] = {**session, "explore_prompt": True}
        bot.reply_to(call.message, "Send a quick query like: 'highest Revenue by Region' or 'average log_CFU_g'.", reply_markup=_explorer_back_markup())

    @bot.callback_query_handler(func=lambda call: (getattr(call, "data", "") or "").startswith("output:"))
    def on_output_selection(call: Any) -> None:
        chat_id = call.message.chat.id
        callback_data = call.data or ""
        requested = callback_data.split(":", 1)[1] if ":" in callback_data else ""
        engine = last_engine.get(chat_id)
        if not engine or not engine.get("ok"):
            bot.answer_callback_query(call.id, "No valid analysis result is available yet.", show_alert=True)
            return
        _deliver_requested_outputs(bot, call, engine, chat_id, requested)

    @bot.callback_query_handler(func=lambda call: (getattr(call, "data", "") or "").startswith("extract:"))
    def on_extraction_confirmation(call: Any) -> None:
        chat_id = call.message.chat.id
        callback_data = call.data or ""
        bot.answer_callback_query(call.id)
        state = pending_extracted.get(chat_id)
        if not state or state.get("df") is None:
            bot.reply_to(call.message, "No extracted table is pending confirmation. Please upload or paste a fresh table.")
            return
        frame = state["df"].copy()
        if callback_data == "extract:accept":
            _remember_active_dataset(chat_id, frame)
            pending_extracted.pop(chat_id, None)
            _send_ingestion_card(bot, call.message, frame, offer_download=True)
            route = _multivariate_table_route(frame)
            if route and route.get("kind") == "multivariate":
                _prompt_multivariate(call.message, route)
            else:
                csv_text = frame.to_csv(index=False)
                engine = _run_analysis(csv_text)
                if engine.get("ok"):
                    last_engine[chat_id] = engine
                _send_analysis(bot, call.message, engine)
            return
        if callback_data == "extract:retry":
            pending_extracted.pop(chat_id, None)
            bot.reply_to(call.message, "Please re-upload a CSV/Excel file or paste the table directly so I can reprocess it.")
            return
        bot.answer_callback_query(call.id, "Unknown extraction decision.", show_alert=True)

    def _maybe_route_multivariate(message: Any, raw_text: str) -> bool:
        route = _multivariate_table_route(_read_delimited_frame(raw_text))
        if not route:
            return False
        if route["kind"] == "invalid":
            bot.reply_to(message, route["message"])
            return True
        if route["kind"] == "needs_mapping":
            user_sessions[message.chat.id] = {"df": route["frame"]}
            bot.reply_to(
                message,
                route["message"],
                reply_markup=_manual_mapping_markup(route["frame"]),
            )
            return True
        if route["kind"] == "multivariate":
            _prompt_multivariate(message, route)
            return True
        return False

    def _has_header_line(raw_text: str) -> bool:
        first_line = next((line.strip() for line in raw_text.splitlines() if line.strip()), "")
        if not first_line:
            return False
        try:
            dialect = csv.Sniffer().sniff(first_line, delimiters=",;\t")
            cells = next(csv.reader([first_line], dialect))
        except (csv.Error, StopIteration):
            return False
        return len(cells) >= 2 and any(re.search(r"[A-Za-z]", cell or "") for cell in cells)

    def _looks_like_pasted_table(raw_text: str) -> bool:
        lines = [line for line in raw_text.splitlines() if line.strip()]
        if len(lines) < 2:
            return False
        first_line = lines[0]
        return any(delimiter in first_line for delimiter in (",", ";", "\t"))

    def _process_pasted_text(message: Any, accumulated_text: str) -> None:
        chat_id = message.chat.id
        if not _begin_ingestion(chat_id):
            return
        try:
            line_count = len([line for line in accumulated_text.splitlines() if line.strip()])
            if line_count > 50:
                bot.reply_to(
                    message,
                    "Tip: For large datasets (100+ rows), uploading as a .csv or .txt file avoids Telegram message splitting.",
                )
            if chat_id in pending_multivariate:
                bot.reply_to(message, "Please finish the current column selection before sending another dataset.")
                return
            script_frame = _python_script_to_dataframe(accumulated_text)
            if script_frame is not None:
                script_frame = sanitize_incoming_dataframe(script_frame)
                _remember_active_dataset(chat_id, script_frame)
                _send_ingestion_card(bot, message, script_frame, offer_download=True)
                route = _multivariate_table_route(script_frame)
                if route and route.get("kind") == "multivariate":
                    _prompt_multivariate(message, route)
                else:
                    engine = _run_analysis(accumulated_text)
                    if engine.get("ok"):
                        last_engine[chat_id] = engine
                    _send_analysis(bot, message, engine)
                return
            if _maybe_route_multivariate(message, accumulated_text):
                return
            frame = sanitize_incoming_dataframe(_read_delimited_frame(accumulated_text))
            if frame is not None:
                _remember_active_dataset(chat_id, frame)
                _send_ingestion_card(bot, message, frame, offer_download=True)
            engine = _run_analysis(accumulated_text)
            if engine.get("ok"):
                last_engine[chat_id] = engine
            _send_analysis(bot, message, engine)
        finally:
            _finish_ingestion(chat_id)

    def _flush_text_buffer(chat_id: int) -> None:
        with text_buffer_lock:
            state = text_buffers.pop(chat_id, None)
        if not state:
            return
        try:
            _process_pasted_text(state["message"], state["text"])
        except Exception as exc:
            _send_error(bot, state["message"], exc, "Could not analyse that")

    def _queue_pasted_text(message: Any, text: str) -> bool:
        if not _looks_like_pasted_table(text):
            return False
        chat_id = message.chat.id
        now = time.monotonic()
        has_header = _has_header_line(text)
        with text_buffer_lock:
            previous = text_buffers.get(chat_id)
            within_cooldown = bool(
                previous and now - previous["received_at"] <= text_chunk_cooldown_seconds
            )
            if previous and within_cooldown:
                accumulated = previous["text"].rstrip() + "\n" + text.lstrip()
                timer = previous.get("timer")
                if timer:
                    timer.cancel()
            else:
                accumulated = text
            timer = threading.Timer(
                text_debounce_seconds,
                _flush_text_buffer,
                args=(chat_id,),
            )
            timer.daemon = True
            text_buffers[chat_id] = {
                "text": accumulated,
                "message": message,
                "received_at": now,
                "has_header": bool(previous and within_cooldown and (previous["has_header"] or has_header)),
                "timer": timer,
            }
            timer.start()
        return True

    @bot.callback_query_handler(func=lambda call: (getattr(call, "data", "") or "").startswith(("factor:", "outcome:")))
    def on_multivariate_selection(call: Any) -> None:
        chat_id = call.message.chat.id
        state = pending_multivariate.get(chat_id)
        if not state:
            session = user_sessions.get(chat_id)
            if session and session.get("df") is not None:
                route = _multivariate_table_route(session["df"])
                if route and route.get("kind") == "multivariate":
                    pending_multivariate[chat_id] = route
                    state = route
                else:
                    bot.answer_callback_query(call.id, "No active selection is available. Please choose a new dataset or upload one again.", show_alert=True)
                    return
            else:
                bot.answer_callback_query(call.id, "No active selection is available. Please choose a new dataset or upload one again.", show_alert=True)
                return
        bot.answer_callback_query(call.id)
        callback_data = call.data or ""
        if ":" not in callback_data:
            bot.answer_callback_query(call.id, "Invalid selection.", show_alert=True)
            return
        selection_type, selected_column = callback_data.split(":", 1)
        if selection_type not in {"factor", "outcome"} or not selected_column:
            bot.answer_callback_query(call.id, "Invalid selection.", show_alert=True)
            return

        if selection_type == "factor":
            factors = state["factors"]
            if selected_column not in factors:
                bot.answer_callback_query(call.id, "Invalid grouping factor.", show_alert=True)
                return
            state["chosen_factor"] = selected_column
            bot.edit_message_text(
                "Step 2/2: Select the Outcome Variable to measure (Dependent Variable)",
                chat_id=chat_id,
                message_id=call.message.message_id,
                reply_markup=_choice_markup("outcome", state["outcomes"]),
            )
            return

        outcomes = state["outcomes"]
        chosen_factor = state.get("chosen_factor")
        if selected_column not in outcomes or not chosen_factor:
            bot.answer_callback_query(call.id, "Select a grouping factor first.", show_alert=True)
            return
        state["chosen_outcome"] = selected_column
        chosen_outcome = state["chosen_outcome"]
        pending_multivariate.pop(chat_id, None)
        frame = state["frame"]
        groups = [
            group.dropna().values
            for _, group in frame.groupby(chosen_factor)[chosen_outcome]
        ]
        if any(len(group) < 2 for group in groups):
            user_sessions.setdefault(chat_id, {"df": frame})["singleton_factor"] = chosen_factor
            bot.reply_to(
                call.message,
                "I found a group with only one observation. Choose how to proceed:",
                reply_markup=_singleton_triage_markup(),
            )
            return
        grouped = frame.groupby(chosen_factor)[chosen_outcome]
        ingested = {
            "format": "labelled",
            "groups": [
                {
                    "name": str(group_name),
                    "values": [float(value) for value in group.dropna().values],
                }
                for group_name, group in grouped
            ],
        }
        engine = _analyze_ingested_safely(ingested, outcome_name=str(chosen_outcome))
        engine["ingested"] = ingested
        engine["qa"] = quality_check(ingested)
        engine["breakdown"] = build_breakdown(engine)
        if engine.get("ok"):
            last_engine[chat_id] = engine
            _send_analysis(bot, call.message, engine, ask_for_document=False)
            if _assumption_violation(engine):
                session = user_sessions.get(chat_id, {})
                if isinstance(session, dict):
                    _cache_assumption_state(session, frame, chosen_factor, chosen_outcome)
                    user_sessions[chat_id] = session
                bot.reply_to(call.message, "Assumptions violated: Proceed with standard test or run non-parametric alternative?", reply_markup=_assumption_markup())
                return
            send_results_document(call.message)
        else:
            _send_analysis(bot, call.message, engine)

    @bot.message_handler(content_types=["text"])
    def on_text(message: Any) -> None:
        try:
            text = message.text or ""
            if _is_global_reset_request(text):
                _handle_global_reset(
                    bot,
                    message,
                    pending=pending,
                    pending_multivariate=pending_multivariate,
                    text_buffers=text_buffers,
                )
                return
            if text.strip().startswith("/"):
                return
            if _is_small_talk(text):
                bot.reply_to(message, "Hey — good to hear from you.\nSend a table, CSV, Excel file, or photo and I’ll help run the test.")
                return
            if re.search(r"\b(defense|viva)\b", text, flags=re.I):
                send_defense(message)
                return
            if message.chat.id in pending_multivariate:
                bot.reply_to(message, "Please finish the current column selection before sending another dataset.")
                return
            session = user_sessions.get(message.chat.id, {})
            if isinstance(session, dict) and session.get("df") is not None and not session.get("ingesting_locked"):
                if _read_only_explanation_request(text):
                    explanation = _gemini_read_only_explanation(message.chat.id, text)
                    if explanation:
                        bot.reply_to(message, explanation)
                        return
                if session.get("explore_prompt") or session.get("state") == "EXPLORER_MODE":
                    frame = session["df"]
                    try:
                        summary = run_explorer_query(frame, text)
                    except Exception as exc:
                        summary = str(exc)
                    bot.reply_to(message, summary, reply_markup=_explorer_back_markup())
                    session["explore_prompt"] = False
                    session["state"] = "EXPLORER_MODE"
                    user_sessions[message.chat.id] = session
                    return
                conversational = _evaluate_conversational_query(message.chat.id, text)
                if conversational is not None:
                    if conversational["kind"] == "compare":
                        factor = conversational["factor"]
                        metric = conversational["metric"]
                        frame = conversational["frame"].copy()
                        ingested = _rebuild_ingested_from_frame(frame, factor, metric)
                        engine = _analyze_ingested_safely(ingested, outcome_name=str(metric))
                    else:
                        frame = conversational["frame"].copy()
                        engine = handle_analyze({"text": frame.to_csv(index=False), "outcome": conversational["y"] if conversational.get("y") else conversational.get("x")})
                    if engine.get("ok"):
                        last_engine[message.chat.id] = engine
                    _send_analysis(bot, message, engine)
                    return
                if re.search(r"\b(compare|across|between|difference|predict|regression|anova|test)\b", text, flags=re.I):
                    markup = _active_dataset_query_markup(session["df"])
                    bot.reply_to(
                        message,
                        "I see your active dataset. Which factor and metric would you like to analyze? (e.g., 'Compare Transaction_Revenue across Region')",
                        reply_markup=markup,
                    )
                    return
                    
            if (
                text.strip().upper() in {"YES", "YES4"}
                and message.chat.id in last_engine
                and message.chat.id not in pending
            ):
                send_results_document(message)
                return
            if text.strip().upper() == "YES" and message.chat.id in pending:
                analysis_text = pending.pop(message.chat.id)["text"]
                if _maybe_route_multivariate(message, analysis_text):
                    return
                engine = _run_analysis(analysis_text)
            elif message.chat.id in pending:
                bot.reply_to(message, "I have the extracted table ready. Reply YES to analyse it.")
                return
            else:
                if _queue_pasted_text(message, text):
                    return
                _process_pasted_text(message, text)
                return
            if engine.get("ok"):
                last_engine[message.chat.id] = engine
            _send_analysis(bot, message, engine)
            if _assumption_failure(engine):
                session = user_sessions.get(message.chat.id, {})
                if isinstance(session, dict):
                    _cache_assumption_state(session, session.get("df"), session.get("factor"), session.get("metric"))
                    user_sessions[message.chat.id] = session
                bot.reply_to(message, "Assumptions violated: Proceed with standard test or run non-parametric alternative?", reply_markup=_assumption_markup())
                return
            bot.reply_to(message, "Choose your output format:", reply_markup=_output_selector_markup())
        except Exception as exc:
            _send_error(bot, message, exc, "Could not analyse that")

    def _dataset_explore_summary(frame: pd.DataFrame, query: str) -> str:
        if frame is None or frame.empty:
            return "No dataset is active for exploration."
        match = _match_dataset_columns(frame, query)
        if not match:
            return "I can answer quick queries like 'average Revenue', 'highest Revenue by Region', or 'mean score'."
        metric_name, factor_name, aggregation = match
        if aggregation == "average":
            if factor_name:
                grouped = frame.groupby(factor_name, dropna=False)[metric_name].mean().sort_values(ascending=False)
                if grouped.empty:
                    return f"No valid data was available for {metric_name} by {factor_name}."
                top = grouped.head(3)
                return f"Average {metric_name} by {factor_name}: " + "; ".join(f"{label} = {value:.3f}" for label, value in top.items()) + "."
            return f"Average {metric_name} = {frame[metric_name].mean():.3f}."
        if factor_name:
            grouped = frame.groupby(factor_name, dropna=False)[metric_name].mean().sort_values(ascending=False)
            if grouped.empty:
                return f"No valid data was available for {metric_name} by {factor_name}."
            best_name, best_value = grouped.iloc[0]
            return f"Highest average {metric_name} by {factor_name}: {best_name} = {best_value:.3f}."
        return f"{metric_name} summary: mean = {frame[metric_name].mean():.3f}, median = {frame[metric_name].median():.3f}."

    def _match_dataset_columns(frame: pd.DataFrame, query: str) -> tuple[str, str | None, str] | None:
        text = str(query or "").lower()
        if not text:
            return None
        if rapidfuzz_process is not None:
            metric_candidates = [column for column in frame.columns if not _is_identifier_column(frame, column) and not _is_temporal_or_datetime_column(frame, column)]
            query_terms = re.findall(r"[a-zA-Z0-9_\- ]+", text)
            for candidate in metric_candidates:
                score = rapidfuzz_process.fuzz.ratio(str(candidate).lower(), " ".join(query_terms))
                if score > 60:
                    metric_name = candidate
                    break
            else:
                metric_name = None
        else:
            metric_name = _column_lookup(frame, text)
        if metric_name is None:
            return None
        factor_name = None
        if re.search(r"by\s+([a-z0-9_\- ]+)", text):
            factor_candidate = re.search(r"by\s+([a-z0-9_\- ]+)", text).group(1)
            factor_name = _column_lookup(frame, factor_candidate)
        aggregation = "average" if re.search(r"(?:average|mean|avg)", text) else "highest" if re.search(r"(?:highest|top|max|largest)", text) else "average"
        return metric_name, factor_name, aggregation

    @bot.message_handler(content_types=["document"])
    def on_document(message: Any) -> None:
        if not _mark_message_processed(int(message.chat.id), getattr(message, "message_id", None)):
            return
        if not _begin_ingestion(message.chat.id):
            return
        try:
            bot.reply_to(message, "Got it, reading your file…")
            document = message.document
            name = getattr(document, "file_name", None) or "upload"
            mime_type = getattr(document, "mime_type", "") or ""
            suffix = _upload_suffix(name, mime_type)
            if not suffix:
                raise ValueError("Please send a CSV, TXT, TSV, Excel, ZIP, PDF, or DOCX file.")
            with tempfile.TemporaryDirectory(prefix="rowfirst-upload-") as tmp:
                safe_name = Path(name).name
                if Path(safe_name).suffix.lower() != suffix:
                    safe_name = f"{safe_name}{suffix}"
                path = Path(tmp) / safe_name
                info = bot.get_file(document.file_id)
                path.write_bytes(bot.download_file(info.file_path))
                if suffix in LOCAL_SUFFIXES:
                    frame = pd.read_csv(path) if suffix == ".csv" else None
                    if frame is None and suffix in {".xlsx", ".xls"}:
                        try:
                            frame = pd.read_excel(path)
                        except Exception:
                            frame = None
                    if frame is not None and not frame.empty:
                        frame = sanitize_incoming_dataframe(frame)
                        _remember_active_dataset(message.chat.id, frame)
                        _send_ingestion_card(bot, message, frame, offer_download=True)
                    route = _multivariate_file_route(path)
                    if route and route["kind"] == "invalid":
                        bot.reply_to(message, route["message"])
                        return
                    if route and route["kind"] == "multivariate":
                        _prompt_multivariate(message, route)
                        return
                    engine = _engine_from_file(path)
                    if engine.get("ok"):
                        last_engine[message.chat.id] = engine
                    _send_analysis(bot, message, engine)
                    if _assumption_failure(engine):
                        session = user_sessions.get(message.chat.id, {})
                        if isinstance(session, dict):
                            _cache_assumption_state(session, session.get("df"), session.get("factor"), session.get("metric"))
                            user_sessions[message.chat.id] = session
                        bot.reply_to(message, "Assumptions violated: Proceed with standard test or run non-parametric alternative?", reply_markup=_assumption_markup())
                        return
                    bot.reply_to(message, "Choose your output format:", reply_markup=_output_selector_markup())
                    return
                if suffix in GEMINI_SUFFIXES:
                    frame = extract_structured_table(path)
                    if frame is None or frame.empty:
                        table = _gemini_extract(path)
                        pending[message.chat.id] = {
                            "text": _table_as_csv(table),
                            "source": suffix,
                        }
                        _reply_block(bot, message, _table_preview(table))
                        return
                    pending_extracted[message.chat.id] = {"df": frame}
                    preview = build_extraction_preview(frame)
                    bot.reply_to(message, preview, reply_markup=_table_confirmation_markup())
                    return
        except Exception as exc:
            _send_error(bot, message, exc)
        finally:
            _finish_ingestion(message.chat.id)

    @bot.message_handler(content_types=["photo"])
    def on_photo(message: Any) -> None:
        if not _mark_message_processed(int(message.chat.id), getattr(message, "message_id", None)):
            return
        if not _begin_ingestion(message.chat.id):
            return
        bot.reply_to(message, "🖼️ Waking up Gemini... Analyzing your image, please wait.")

        def process_photo() -> None:
            try:
                with tempfile.TemporaryDirectory(prefix="rowfirst-photo-") as tmp:
                    path = Path(tmp) / "photo.jpg"
                    info = bot.get_file(message.photo[-1].file_id)
                    path.write_bytes(bot.download_file(info.file_path))
                    frame = extract_structured_table(path)
                    if frame is None or frame.empty:
                        table = _gemini_extract(path)
                        csv_text = _table_as_csv(table)
                        frame = sanitize_incoming_dataframe(_read_delimited_frame(csv_text))
                    if frame is not None:
                        pending_extracted[message.chat.id] = {"df": frame}
                        preview = build_extraction_preview(frame)
                        bot.reply_to(message, preview, reply_markup=_table_confirmation_markup())
                        return
                    engine = handle_analyze({"text": ""})
                    if engine.get("ok"):
                        last_engine[message.chat.id] = engine
                    _send_analysis(bot, message, engine)
            except Exception:
                bot.reply_to(
                    message,
                    "⚠️ Gemini failed to process the image. Please try uploading again or use a CSV.",
                )
            finally:
                _finish_ingestion(message.chat.id)

        threading.Thread(
            target=process_photo,
            name="rowfirst-gemini-photo",
            daemon=True,
        ).start()

    threading.Thread(
        target=_run_health_server,
        name="rowfirst-health-server",
        daemon=True,
    ).start()
    print("polling started once", flush=True)
    bot.infinity_polling(skip_pending=True)


if __name__ == "__main__":
    main()