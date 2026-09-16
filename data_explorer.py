from __future__ import annotations

import re
from typing import Any

import numpy as np
import pandas as pd

try:
    from rapidfuzz import process as rapidfuzz_process
except Exception:  # pragma: no cover
    rapidfuzz_process = None


def _canonical_label(value: Any) -> str:
    return re.sub(r"[^a-z0-9]+", " ", str(value).strip().lower()).strip()


def _column_candidates(frame: pd.DataFrame) -> list[str]:
    return [str(column) for column in frame.columns if column is not None]


def _resolve_column(frame: pd.DataFrame, raw_label: str | Any, *, threshold: int = 75) -> str | None:
    text = str(raw_label or "").strip()
    if not text:
        return None
    columns = _column_candidates(frame)
    lowered = text.lower()
    exact = next((column for column in columns if _canonical_label(column) == _canonical_label(text)), None)
    if exact is not None:
        return exact
    if lowered in {str(column).lower() for column in columns}:
        return next(column for column in columns if str(column).lower() == lowered)
    if rapidfuzz_process is not None:
        best_match = rapidfuzz_process.extractOne(lowered, [str(column).lower() for column in columns])
        if best_match is not None:
            candidate, score, _ = best_match
            if score >= threshold:
                for column in columns:
                    if str(column).lower() == candidate:
                        return column
    for column in columns:
        if lowered in str(column).lower():
            return column
    return None


def _resolve_ambiguous(frame: pd.DataFrame, raw_label: str | Any, field_name: str, *, threshold: int = 75) -> str:
    match = _resolve_column(frame, raw_label, threshold=threshold)
    if match is not None:
        return match
    available = ", ".join(_column_candidates(frame))
    raise ValueError(f"I couldn't match '{raw_label}' to your columns. Available columns: {available}.")


def _infer_numeric_metric(frame: pd.DataFrame) -> str | None:
    numeric_columns = frame.select_dtypes(include=["number"]).columns
    for column in numeric_columns:
        series = pd.to_numeric(frame[column], errors="coerce").dropna()
        if len(series) >= max(2, int(len(frame) * 0.8)):
            return str(column)
    return None


def _infer_factor(frame: pd.DataFrame) -> str | None:
    for column in frame.columns:
        if pd.api.types.is_numeric_dtype(frame[column]):
            continue
        series = frame[column].dropna().astype(str)
        if series.empty:
            continue
        unique = series.nunique(dropna=True)
        if 1 < unique < len(frame):
            return str(column)
    return None


def _build_group_rank_message(frame: pd.DataFrame, metric: str, factor: str, direction: str) -> str:
    numeric = pd.to_numeric(frame[metric], errors="coerce").dropna()
    if numeric.empty:
        return f"No usable numeric values are available for {metric}."
    grouped = frame.groupby(factor, dropna=False)[metric].sum().sort_values(ascending=(direction == "highest"))
    if grouped.empty:
        return f"No valid groups were available for {factor}."
    top_group, top_val = grouped.head(1).index[0], float(grouped.head(1).iloc[0])
    bottom_group, bottom_val = grouped.tail(1).index[0], float(grouped.tail(1).iloc[0])
    total = float(grouped.sum())
    share_pct = (top_val / total * 100.0) if total else 0.0
    if direction == "highest":
        return (
            f"Across **{factor}**, **{top_group}** ranks highest with **{top_val:,.2f}** "
            f"(representing **{share_pct:.1f}%** of the total volume). In contrast, **{bottom_group}** "
            f"had the lowest at **{bottom_val:,.2f}**."
        )
    return (
        f"Across **{factor}**, **{bottom_group}** ranks lowest with **{bottom_val:,.2f}** "
        f"(representing **{share_pct:.1f}%** of the total volume). In contrast, **{top_group}** "
        f"had the highest at **{top_val:,.2f}**."
    )


def _build_average_message(frame: pd.DataFrame, metric: str) -> str:
    series = pd.to_numeric(frame[metric], errors="coerce").dropna()
    if series.empty:
        return f"No numeric values are available for {metric}."
    mean_val = float(series.mean())
    median_val = float(series.median())
    std_val = float(series.std(ddof=1)) if len(series) > 1 else 0.0
    min_val = float(series.min())
    max_val = float(series.max())
    n_rows = int(len(series))
    return (
        f"The average **{metric}** is **{mean_val:,.2f}** "
        f"(median: **{median_val:,.2f}**, SD: **{std_val:,.2f}**), "
        f"ranging between **{min_val:,.2f}** and **{max_val:,.2f}** across {n_rows} rows."
    )


def _strength_label(r_value: float) -> str:
    if abs(r_value) >= 0.8:
        return "very strong"
    if abs(r_value) >= 0.6:
        return "strong"
    if abs(r_value) >= 0.4:
        return "moderate"
    if abs(r_value) >= 0.2:
        return "weak"
    return "very weak"


def _build_correlation_message(frame: pd.DataFrame, col1: str, col2: str) -> str:
    x = pd.to_numeric(frame[col1], errors="coerce").dropna()
    y = pd.to_numeric(frame[col2], errors="coerce").dropna()
    aligned = pd.concat([x, y], axis=1).dropna()
    if aligned.empty or len(aligned) < 2:
        return f"Not enough paired data to compute the correlation between **{col1}** and **{col2}**."
    r_value = float(aligned.iloc[:, 0].corr(aligned.iloc[:, 1]))
    return (
        f"The correlation between **{col1}** and **{col2}** is **r = {r_value:.3f}** "
        f"({ _strength_label(r_value) })."
    )


def _parse_group_rank(frame: pd.DataFrame, query: str) -> dict[str, Any]:
    text = str(query or "").strip()
    if not text:
        raise ValueError("Please provide a question to explore.")
    pattern = re.compile(r"(?i)(highest|top|max|greatest)\s+(?P<metric>.+?)\s+(by|per|in)\s+(?P<factor>.+)")
    match = pattern.search(text)
    if match:
        metric = _resolve_ambiguous(frame, match.group("metric"), "metric")
        factor = _resolve_ambiguous(frame, match.group("factor"), "factor")
        return {"kind": "group_rank", "metric": metric, "factor": factor, "direction": "highest"}
    pattern = re.compile(r"(?i)(lowest|bottom|min|least)\s+(?P<metric>.+?)\s+(by|per|in)\s+(?P<factor>.+)")
    match = pattern.search(text)
    if match:
        metric = _resolve_ambiguous(frame, match.group("metric"), "metric")
        factor = _resolve_ambiguous(frame, match.group("factor"), "factor")
        return {"kind": "group_rank", "metric": metric, "factor": factor, "direction": "lowest"}
    raise ValueError(f"I couldn't match '{text}' to a supported group-ranking query.")


def _parse_average(frame: pd.DataFrame, query: str) -> dict[str, Any]:
    text = str(query or "").strip()
    pattern = re.compile(r"(?i)(average|mean|median|spread|distribution)\s+(of\s+)?(?P<metric>.+)")
    match = pattern.search(text)
    if not match:
        raise ValueError(f"I couldn't match '{text}' to a supported summary query.")
    metric = _resolve_ambiguous(frame, match.group("metric"), "metric")
    return {"kind": "average", "metric": metric}


def _parse_correlation(frame: pd.DataFrame, query: str) -> dict[str, Any]:
    text = str(query or "").strip()
    pattern = re.compile(r"(?i)(correlation|relationship|corr)\s+(between\s+)?(?P<col1>.+?)\s+and\s+(?P<col2>.+)")
    match = pattern.search(text)
    if not match:
        raise ValueError(f"I couldn't match '{text}' to a supported correlation query.")
    col1 = _resolve_ambiguous(frame, match.group("col1"), "col1")
    col2 = _resolve_ambiguous(frame, match.group("col2"), "col2")
    if col1 == col2:
        raise ValueError("Please compare two different columns.")
    return {"kind": "correlation", "col1": col1, "col2": col2}


def parse_explorer_query(frame: pd.DataFrame, query: str) -> dict[str, Any]:
    text = str(query or "").strip()
    if not text:
        raise ValueError("Please provide a question to explore.")
    pattern_checks = [
        _parse_group_rank,
        _parse_average,
        _parse_correlation,
    ]
    for parser in pattern_checks:
        try:
            return parser(frame, text)
        except ValueError:
            continue
    raise ValueError(f"I couldn't match '{text}' to your columns. Available columns: {', '.join(_column_candidates(frame))}.")


def _default_top_bottom(frame: pd.DataFrame) -> str:
    factor = _infer_factor(frame)
    metric = _infer_numeric_metric(frame)
    if factor is None or metric is None:
        return "I need at least one categorical grouping column and one numeric metric to rank the dataset."
    return _build_group_rank_message(frame, metric, factor, "highest")


def _default_key_metrics(frame: pd.DataFrame) -> str:
    metric = _infer_numeric_metric(frame)
    if metric is None:
        return "No numeric metric is available for a distribution summary."
    return _build_average_message(frame, metric)


def _default_correlations(frame: pd.DataFrame) -> str:
    numeric_columns = [
        str(column)
        for column in frame.columns
        if pd.api.types.is_numeric_dtype(frame[column])
        and pd.to_numeric(frame[column], errors="coerce").dropna().shape[0] >= max(2, int(len(frame) * 0.8))
    ]
    best_pair: tuple[str, str, float] | None = None
    for index, col1 in enumerate(numeric_columns):
        for col2 in numeric_columns[index + 1 :]:
            if col1 == col2:
                continue
            aligned = pd.concat([
                pd.to_numeric(frame[col1], errors="coerce"),
                pd.to_numeric(frame[col2], errors="coerce"),
            ], axis=1).dropna()
            if aligned.empty or len(aligned) < 2:
                continue
            r_value = float(aligned.iloc[:, 0].corr(aligned.iloc[:, 1]))
            if abs(r_value) > 0.5:
                candidate = (col1, col2, r_value)
                if best_pair is None or abs(candidate[2]) > abs(best_pair[2]):
                    best_pair = candidate
    if best_pair is None:
        return "No strong numeric correlations were found above |r| > 0.5 in the current dataset."
    col1, col2, r_value = best_pair
    return _build_correlation_message(frame, col1, col2)


def run_explorer_action(frame: pd.DataFrame, action: str) -> str:
    action_key = (action or "").strip().lower()
    if action_key == "top_bottom":
        return _default_top_bottom(frame)
    if action_key == "key_metrics":
        return _default_key_metrics(frame)
    if action_key == "correlations":
        return _default_correlations(frame)
    if action_key == "ask_question":
        return "Type a question like 'highest sales by region' or 'average moisture'."
    raise ValueError(f"Unknown explorer action: {action}")


def run_explorer_query(frame: pd.DataFrame, question: str) -> str:
    parsed = parse_explorer_query(frame, question)
    kind = parsed.get("kind")
    if kind == "group_rank":
        return _build_group_rank_message(frame, parsed["metric"], parsed["factor"], parsed["direction"])
    if kind == "average":
        return _build_average_message(frame, parsed["metric"])
    if kind == "correlation":
        return _build_correlation_message(frame, parsed["col1"], parsed["col2"])
    raise ValueError(f"Unsupported explorer intent: {kind}")
