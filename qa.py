"""Quality checks that flag data problems without changing the engine input."""
from __future__ import annotations

from typing import Any

import numpy as np


def quality_check(ingested: dict[str, Any]) -> dict[str, Any]:
    warnings: list[str] = []
    errors: list[str] = []
    suggestions: list[str] = []
    groups = _all_groups(ingested)
    if ingested.get("format") == "paired":
        pairs = ingested.get("pairs", [])
        if len(pairs) <= 3:
            warnings.append(f"Paired design: n_pairs={len(pairs)}; interpret results cautiously.")
            suggestions.append("Add more matched pairs or treat this as exploratory evidence.")
        if pairs and all(float(pair["after"]) - float(pair["before"]) == float(pairs[0]["after"]) - float(pairs[0]["before"]) for pair in pairs):
            warnings.append("Paired design: identical differences across pairs.")
            suggestions.append("Check whether the paired measurements were copied or rounded; identical differences provide no variable spread.")
    if ingested.get("format") == "two-way":
        rows = ingested.get("rows", [])
        if len(rows) <= 3:
            warnings.append(f"Two-way design: n={len(rows)}; interpret results cautiously.")
            suggestions.append("Add replicated observations across factor combinations before interpreting interaction effects.")
    for group in groups:
        values = np.asarray(group.get("values", []), dtype=float)
        name = group.get("name", "group")
        if len(values) == 1:
            warnings.append(f"{name}: n=1; no reliable spread can be estimated.")
            suggestions.append(f"Add replicate observations for {name} before using inferential tests.")
        elif len(values) <= 3:
            warnings.append(f"{name}: n={len(values)}; interpret results cautiously.")
            suggestions.append(f"Increase the sample size for {name} and report this result as exploratory meanwhile.")
        if len(values) > 1 and np.allclose(values, values[0]):
            warnings.append(f"{name}: identical replicates; SD is zero.")
            suggestions.append(f"Verify the raw measurements for {name}; constant replicates cannot support a variance-based test.")
        if len(values) >= 4:
            q1, q3 = np.percentile(values, [25, 75])
            iqr = q3 - q1
            if iqr > 0:
                outliers = values[(values < q1 - 1.5 * iqr) | (values > q3 + 1.5 * iqr)]
                if len(outliers):
                    warnings.append(f"{name}: {len(outliers)} IQR outlier(s) flagged; values were retained.")
                    suggestions.append(f"Review the flagged {name} values for data-entry or measurement errors; they were retained, not silently removed.")
    for outcome in _outcomes(ingested):
        parameter = str(outcome.get("parameter", "")).lower()
        if "moisture" in parameter:
            for group in outcome.get("groups", []):
                bad = [v for v in group.get("values", []) if not 0 <= float(v) <= 100]
                if bad:
                    errors.append(f"{outcome.get('parameter', 'Moisture')}: values must be between 0 and 100.")
                    suggestions.append(f"Correct {outcome.get('parameter', 'moisture')} values outside 0–100 or confirm that the column is not a percentage.")
    return {
        "ok": not errors,
        "warnings": warnings,
        "errors": errors,
        "suggestions": list(dict.fromkeys(suggestions)),
    }


def _outcomes(ingested: dict[str, Any]) -> list[dict[str, Any]]:
    return list(ingested.get("outcomes") or [{
        "parameter": ingested.get("outcome", ""),
        "groups": ingested.get("groups", []),
    }])


def _all_groups(ingested: dict[str, Any]) -> list[dict[str, Any]]:
    groups: list[dict[str, Any]] = []
    for outcome in _outcomes(ingested):
        groups.extend(outcome.get("groups", []))
    return groups