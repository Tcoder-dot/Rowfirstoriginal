"""Deterministic stats engine. LLM must not compute these numbers."""
from __future__ import annotations

import math
from typing import Any

import numpy as np
from scipy import stats
from statsmodels.formula.api import ols
from statsmodels.stats.anova import anova_lm
from statsmodels.stats.multicomp import pairwise_tukeyhsd


def _mean_sd(x: np.ndarray) -> tuple[float, float]:
    mean = float(np.mean(x))
    return mean, float(np.std(x, ddof=1)) if len(x) > 1 else 0.0


def _cohens_d(a: np.ndarray, b: np.ndarray) -> float:
    n1, n2 = len(a), len(b)
    s1, s2 = np.std(a, ddof=1), np.std(b, ddof=1)
    sp = math.sqrt(((n1 - 1) * s1 ** 2 + (n2 - 1) * s2 ** 2) / (n1 + n2 - 2))
    if sp == 0:
        return 0.0
    return float((np.mean(a) - np.mean(b)) / sp)


def _ci_diff(a: np.ndarray, b: np.ndarray, alpha: float = 0.05) -> list[float]:
    n1, n2 = len(a), len(b)
    diff = float(np.mean(a) - np.mean(b))
    s1, s2 = np.std(a, ddof=1), np.std(b, ddof=1)
    sp = math.sqrt(((n1 - 1) * s1 ** 2 + (n2 - 1) * s2 ** 2) / (n1 + n2 - 2))
    se = sp * math.sqrt(1 / n1 + 1 / n2)
    tcrit = float(stats.t.ppf(1 - alpha / 2, n1 + n2 - 2))
    return [diff - tcrit * se, diff + tcrit * se]


def _ci_diff_welch(a: np.ndarray, b: np.ndarray, alpha: float = 0.05) -> list[float]:
    n1, n2 = len(a), len(b)
    diff = float(np.mean(a) - np.mean(b))
    v1, v2 = np.var(a, ddof=1) / n1, np.var(b, ddof=1) / n2
    se = math.sqrt(v1 + v2)
    df = (v1 + v2) ** 2 / ((v1 ** 2) / (n1 - 1) + (v2 ** 2) / (n2 - 1))
    tcrit = float(stats.t.ppf(1 - alpha / 2, df))
    return [diff - tcrit * se, diff + tcrit * se]


def describe_group(name: str, values: list[float]) -> dict[str, Any]:
    x = np.asarray(values, dtype=float)
    mean, sd = _mean_sd(x)
    sw = None
    if 3 <= len(x) <= 5000 and np.std(x) > 0:
        W, p = stats.shapiro(x)
        sw = {"W": float(W), "p": float(p), "isNormal": bool(p >= 0.05)}
    return {
        "name": name,
        "n": int(len(x)),
        "totalSum": float(np.sum(x)),
        "mean": mean,
        "sd": sd,
        "median": float(np.median(x)),
        "shapiroWilk": sw,
    }


def independent_ttest(g1: dict, g2: dict, equal_var: bool = True) -> dict[str, Any]:
    a = np.asarray(g1["values"], dtype=float)
    b = np.asarray(g2["values"], dtype=float)
    res = stats.ttest_ind(a, b, equal_var=equal_var)
    d = _cohens_d(a, b)
    df = float(res.df) if hasattr(res, "df") and res.df is not None else (len(a) + len(b) - 2)
    return {
        "test": "welch-t" if not equal_var else "student-t",
        "group1": describe_group(g1["name"], g1["values"]),
        "group2": describe_group(g2["name"], g2["values"]),
        "t": float(res.statistic),
        "df": float(df),
        "p": float(res.pvalue),
        "difference": float(np.mean(a) - np.mean(b)),
        "confidenceInterval95": _ci_diff(a, b) if equal_var else _ci_diff_welch(a, b),
        "cohensD": d,
        "isSignificant": bool(res.pvalue < 0.05),
    }


def _anova_effect_sizes(arrs: list[np.ndarray]) -> dict[str, float]:
    """Return eta-squared and omega-squared for a one-way ANOVA."""
    all_values = np.concatenate(arrs)
    grand_mean = float(np.mean(all_values))
    ss_between = float(sum(len(values) * (float(np.mean(values)) - grand_mean) ** 2 for values in arrs))
    ss_within = float(sum(np.sum((values - np.mean(values)) ** 2) for values in arrs))
    ss_total = ss_between + ss_within
    df_within = sum(len(values) for values in arrs) - len(arrs)
    ms_within = ss_within / df_within if df_within > 0 else 0.0
    eta_squared = ss_between / ss_total if ss_total else 0.0
    omega_denominator = ss_total + ms_within
    omega_squared = (
        (ss_between - (len(arrs) - 1) * ms_within) / omega_denominator
        if omega_denominator
        else 0.0
    )
    return {
        "etaSquared": float(eta_squared),
        "omegaSquared": float(omega_squared),
    }


def _anova_assumptions(groups: list[dict], arrs: list[np.ndarray]) -> dict[str, Any]:
    shapiro = []
    for group, values in zip(groups, arrs):
        result = describe_group(str(group["name"]), values.tolist()).get("shapiroWilk")
        shapiro.append({
            "group": str(group["name"]),
            **(result or {"available": False, "reason": "Shapiro-Wilk needs 3–5000 non-constant observations."}),
        })

    levene_result: dict[str, Any]
    if len(arrs) >= 2 and all(len(values) >= 2 for values in arrs):
        statistic, p_value = stats.levene(*arrs, center="median")
        levene_result = {
            "W": float(statistic),
            "p": float(p_value),
            "equalVariance": bool(p_value >= 0.05),
        }
    else:
        levene_result = {
            "available": False,
            "reason": "Levene's test needs at least two observations per group.",
        }
    return {"shapiroWilk": shapiro, "levene": levene_result}


def _tukey_post_hoc(groups: list[dict], arrs: list[np.ndarray]) -> dict[str, Any]:
    values = np.concatenate(arrs)
    labels = np.concatenate([
        np.repeat(str(group["name"]), len(values_for_group))
        for group, values_for_group in zip(groups, arrs)
    ])
    tukey = pairwise_tukeyhsd(values, labels, alpha=0.05)
    comparisons = []
    for row in tukey.summary().data[1:]:
        group1, group2, mean_difference, p_adjusted, lower, upper, reject = row
        comparisons.append({
            "group1": str(group1),
            "group2": str(group2),
            "meanDifference": float(mean_difference),
            "pAdjusted": float(p_adjusted),
            "lower": float(lower),
            "upper": float(upper),
            "reject": bool(reject),
        })
    return {
        "test": "Tukey HSD",
        "alpha": 0.05,
        "comparisons": comparisons,
    }


def one_way_anova(groups: list[dict], outcome: str | None = None) -> dict[str, Any]:
    arrs = [np.asarray(g["values"], dtype=float) for g in groups]
    pooled_within_variance = sum(float(np.sum((values - np.mean(values)) ** 2)) for values in arrs)
    if pooled_within_variance == 0:
        raise ValueError(
            "F-test undefined: identical replicate measurements detected with zero within-group variance"
        )
    res = stats.f_oneway(*arrs)
    if np.isinf(res.statistic) or np.isnan(res.statistic):
        raise ValueError(
            "F-test undefined: identical replicate measurements detected with zero within-group variance"
        )
    k = len(groups)
    n = sum(len(a) for a in arrs)
    result = {
        "test": "one-way anova",
        "groups": [describe_group(g["name"], g["values"]) for g in groups],
        "F": float(res.statistic),
        "dfb": k - 1,
        "dfw": n - k,
        "p": float(res.pvalue),
        "isSignificant": bool(res.pvalue < 0.05),
        "assumptions": _anova_assumptions(groups, arrs),
        "effectSize": _anova_effect_sizes(arrs),
    }
    result["shapiroWilk"] = result["assumptions"]["shapiroWilk"]
    result["levene"] = result["assumptions"]["levene"]
    result["etaSquared"] = result["effectSize"]["etaSquared"]
    result["omegaSquared"] = result["effectSize"]["omegaSquared"]
    if outcome:
        result["outcome"] = str(outcome)
    if result["isSignificant"]:
        result["postHoc"] = _tukey_post_hoc(groups, arrs)
    return result


def chi_or_fisher(matrix: list[list[int]]) -> dict[str, Any]:
    m = np.asarray(matrix, dtype=float)
    if m.shape == (2, 2):
        oddsr, p = stats.fisher_exact(m)
        chi = stats.chi2_contingency(m, correction=False)
        return {
            "test": "fisher-exact",
            "matrix": matrix,
            "oddsRatio": float(oddsr),
            "p": float(p),
            "chi2Uncorrected": float(chi.statistic),
            "chi2pUncorrected": float(chi.pvalue),
            "isSignificant": bool(p < 0.05),
        }
    chi = stats.chi2_contingency(m, correction=False)
    return {
        "test": "chi-square",
        "matrix": matrix,
        "chi2": float(chi.statistic),
        "df": int(chi.dof),
        "p": float(chi.pvalue),
        "expected": chi.expected_freq.tolist(),
        "isSignificant": bool(chi.pvalue < 0.05),
    }


def correlation(x: list[float], y: list[float], method: str = "pearson") -> dict[str, Any]:
    """Compute a correlation from two numeric columns without any LLM arithmetic."""
    a = np.asarray(x, dtype=float)
    b = np.asarray(y, dtype=float)
    if len(a) != len(b) or len(a) < 3:
        raise ValueError("Correlation needs at least three paired observations.")
    if not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError("Correlation columns must contain finite numbers only.")
    if np.ptp(a) == 0 or np.ptp(b) == 0:
        raise ValueError("Correlation needs non-constant numeric columns.")
    if method not in {"pearson", "spearman"}:
        raise ValueError("Correlation method must be pearson or spearman.")
    result = stats.pearsonr(a, b) if method == "pearson" else stats.spearmanr(a, b)
    return {
        "test": method,
        "n": int(len(a)),
        "r": float(result.statistic),
        "p": float(result.pvalue),
        "isSignificant": bool(result.pvalue < 0.05),
    }


def paired_ttest(
    before: list[float], after: list[float], pair_ids: list[str] | None = None
) -> dict[str, Any]:
    """Paired t-test and Cohen's dz from already matched observations."""
    a = np.asarray(before, dtype=float)
    b = np.asarray(after, dtype=float)
    if len(a) != len(b) or len(a) < 2:
        raise ValueError("Paired t-test needs at least two matched pairs.")
    if not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError("Paired columns must contain finite numbers only.")
    differences = b - a
    if np.ptp(differences) == 0:
        raise ValueError("Paired t-test needs variable paired differences.")
    result = stats.ttest_rel(a, b)
    sd_diff = float(np.std(differences, ddof=1))
    dz = float(np.mean(differences) / sd_diff) if sd_diff else 0.0
    return {
        "test": "paired-t",
        "nPairs": int(len(a)),
        "before": {
            "name": "before",
            "n": int(len(a)),
            "totalSum": float(np.sum(a)),
            "mean": float(np.mean(a)),
            "sd": float(np.std(a, ddof=1)),
        },
        "after": {
            "name": "after",
            "n": int(len(b)),
            "totalSum": float(np.sum(b)),
            "mean": float(np.mean(b)),
            "sd": float(np.std(b, ddof=1)),
        },
        "meanDifference": float(np.mean(differences)),
        "t": float(result.statistic),
        "df": float(len(a) - 1),
        "p": float(result.pvalue),
        "cohensDz": dz,
        "isSignificant": bool(result.pvalue < 0.05),
        "pairIds": pair_ids or [str(i + 1) for i in range(len(a))],
        "pairs": list(zip(a.tolist(), b.tolist())),
    }


def two_way_anova(
    rows: list[dict[str, Any]], factor_a: str, factor_b: str, outcome: str
) -> dict[str, Any]:
    """Type-II two-way ANOVA using statsmodels for the design matrix and tests."""
    import pandas as pd

    frame = pd.DataFrame(rows)
    frame = frame[[factor_a, factor_b, outcome]].dropna()
    if len(frame) < 4:
        raise ValueError("Two-way ANOVA needs at least four complete observations.")
    frame[outcome] = pd.to_numeric(frame[outcome], errors="raise")
    frame["_a"] = frame[factor_a].astype(str)
    frame["_b"] = frame[factor_b].astype(str)
    frame["_y"] = frame[outcome].astype(float)
    if frame["_a"].nunique() < 2 or frame["_b"].nunique() < 2:
        raise ValueError("Both two-way ANOVA factors need at least two levels.")
    model = ols("_y ~ C(_a) * C(_b)", data=frame).fit()
    table = anova_lm(model, typ=2)
    residual_df = float(table.loc["Residual", "df"])
    if not np.isfinite(residual_df) or residual_df <= 0:
        raise ValueError("Two-way ANOVA needs a non-singular design with residual degrees of freedom.")
    effects = []
    labels = [
        (f"{factor_a}", "C(_a)"),
        (f"{factor_b}", "C(_b)"),
        (f"{factor_a}:{factor_b}", "C(_a):C(_b)"),
    ]
    for label, row_name in labels:
        row = table.loc[row_name]
        if not np.isfinite(float(row["F"])) or not np.isfinite(float(row["PR(>F)"])):
            raise ValueError("Two-way ANOVA produced a non-finite effect; the design is singular.")
        effects.append(
            {
                "effect": label,
                "df": float(row["df"]),
                "F": float(row["F"]),
                "p": float(row["PR(>F)"]),
                "isSignificant": bool(row["PR(>F)"] < 0.05),
            }
        )
    return {
        "test": "two-way anova",
        "factorA": factor_a,
        "factorB": factor_b,
        "outcome": outcome,
        "n": int(len(frame)),
        "residualDf": residual_df,
        "effects": effects,
        "interactionSignificant": bool(effects[2]["p"] < 0.05),
        "isSignificant": bool(any(effect["p"] < 0.05 for effect in effects)),
    }


def linear_regression(x: list[float], y: list[float], x_name: str = "X", y_name: str = "Y") -> dict[str, Any]:
    """One-predictor linear regression from scipy.stats.linregress."""
    a = np.asarray(x, dtype=float)
    b = np.asarray(y, dtype=float)
    if len(a) != len(b) or len(a) < 3:
        raise ValueError("Simple linear regression needs at least three paired observations.")
    if not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError("Regression columns must contain finite numbers only.")
    if np.ptp(a) == 0 or np.ptp(b) == 0:
        raise ValueError("Regression needs non-constant predictor and outcome columns.")
    result = stats.linregress(a, b)
    return {
        "test": "simple linear regression",
        "predictor": x_name,
        "outcome": y_name,
        "n": int(len(a)),
        "slope": float(result.slope),
        "intercept": float(result.intercept),
        "r": float(result.rvalue),
        "rSquared": float(result.rvalue ** 2),
        "p": float(result.pvalue),
        "stderr": float(result.stderr),
        "isSignificant": bool(result.pvalue < 0.05),
        "pairs": list(zip(a.tolist(), b.tolist())),
    }


def multiple_linear_regression(
    frame: Any,
    predictors: list[str],
    outcome: str,
) -> dict[str, Any]:
    """Deterministic ordinary least-squares regression for explicit columns."""
    import pandas as pd
    import statsmodels.api as sm

    if len(predictors) < 2:
        raise ValueError("Multiple regression needs at least two predictor columns.")
    if outcome in predictors or len(set(predictors)) != len(predictors):
        raise ValueError("Regression predictors and outcome must be distinct columns.")
    columns = [*predictors, outcome]
    if any(column not in frame.columns for column in columns):
        raise ValueError("All regression columns must exist in the uploaded data.")
    working = frame[columns].apply(pd.to_numeric, errors="coerce").dropna()
    if len(working) <= len(predictors) + 1:
        raise ValueError("Multiple regression needs more observations than parameters.")
    x_values = working[predictors].to_numpy(dtype=float)
    y_values = working[outcome].to_numpy(dtype=float)
    if not np.isfinite(x_values).all() or not np.isfinite(y_values).all():
        raise ValueError("Regression columns must contain finite numbers only.")
    design = sm.add_constant(x_values, has_constant="add")
    if np.linalg.matrix_rank(design) < design.shape[1]:
        raise ValueError("Multiple regression predictors are linearly dependent.")
    fitted = sm.OLS(y_values, design).fit()
    if not np.isfinite(fitted.params).all() or not np.isfinite(fitted.pvalues).all():
        raise ValueError("Multiple regression produced non-finite coefficients.")
    coefficients = []
    names = ["intercept", *predictors]
    for name, coefficient, standard_error, p_value in zip(names, fitted.params, fitted.bse, fitted.pvalues):
        coefficients.append({
            "term": name,
            "coefficient": float(coefficient),
            "standardError": float(standard_error),
            "p": float(p_value),
            "isSignificant": bool(p_value < 0.05),
        })
    return {
        "test": "multiple linear regression",
        "predictors": predictors,
        "outcome": outcome,
        "n": int(len(working)),
        "dfModel": int(fitted.df_model),
        "dfResidual": int(fitted.df_resid),
        "rSquared": float(fitted.rsquared),
        "adjustedRSquared": float(fitted.rsquared_adj),
        "F": float(fitted.fvalue),
        "p": float(fitted.f_pvalue),
        "isSignificant": bool(fitted.f_pvalue < 0.05),
        "coefficients": coefficients,
    }


def logistic_regression(
    frame: Any,
    predictors: list[str],
    outcome: str,
) -> dict[str, Any]:
    """Deterministic binary logistic regression for an explicit 0/1 outcome."""
    import pandas as pd
    import statsmodels.api as sm

    if len(predictors) < 1:
        raise ValueError("Logistic regression needs at least one predictor column.")
    if outcome in predictors or len(set(predictors)) != len(predictors):
        raise ValueError("Logistic predictors and outcome must be distinct columns.")
    columns = [*predictors, outcome]
    if any(column not in frame.columns for column in columns):
        raise ValueError("All logistic regression columns must exist in the uploaded data.")
    working = frame[columns].apply(pd.to_numeric, errors="coerce").dropna()
    if len(working) <= len(predictors) + 2:
        raise ValueError("Logistic regression needs more observations than parameters.")
    y_values = working[outcome].to_numpy(dtype=float)
    if set(np.unique(y_values)) != {0.0, 1.0}:
        raise ValueError("Logistic regression outcome must contain exactly the values 0 and 1.")
    x_values = working[predictors].to_numpy(dtype=float)
    design = sm.add_constant(x_values, has_constant="add")
    if np.linalg.matrix_rank(design) < design.shape[1]:
        raise ValueError("Logistic regression predictors are linearly dependent.")
    try:
        fitted = sm.Logit(y_values, design).fit(disp=False, method="lbfgs")
    except Exception as exc:
        raise ValueError("Logistic regression could not fit a stable binary model.") from exc
    if not np.isfinite(fitted.params).all() or not np.isfinite(fitted.pvalues).all():
        raise ValueError("Logistic regression produced non-finite coefficients.")
    coefficients = []
    names = ["intercept", *predictors]
    for name, coefficient, standard_error, p_value in zip(names, fitted.params, fitted.bse, fitted.pvalues):
        coefficients.append({
            "term": name,
            "coefficient": float(coefficient),
            "oddsRatio": float(np.exp(coefficient)),
            "standardError": float(standard_error),
            "p": float(p_value),
            "isSignificant": bool(p_value < 0.05),
        })
    return {
        "test": "logistic regression",
        "predictors": predictors,
        "outcome": outcome,
        "n": int(len(working)),
        "dfModel": int(fitted.df_model),
        "pseudoRSquared": float(fitted.prsquared),
        "logLikelihood": float(fitted.llf),
        "likelihoodRatioP": float(fitted.llr_pvalue),
        "isSignificant": bool(fitted.llr_pvalue < 0.05),
        "coefficients": coefficients,
    }


def linear_forecast(values: list[float], horizon: int = 1) -> dict[str, Any]:
    """Deterministic linear trend forecast for an explicitly ordered series."""
    if len(values) < 3:
        raise ValueError("Forecasting needs at least three ordered observations.")
    if horizon < 1 or horizon > 24:
        raise ValueError("Forecast horizon must be between 1 and 24 periods.")
    observed = np.asarray(values, dtype=float)
    if not np.isfinite(observed).all():
        raise ValueError("Forecast values must contain finite numbers only.")
    time = np.arange(len(observed), dtype=float)
    fitted = stats.linregress(time, observed)
    if not np.isfinite([fitted.slope, fitted.intercept, fitted.pvalue]).all():
        raise ValueError("Forecast trend produced non-finite values.")
    future_time = np.arange(len(observed), len(observed) + horizon, dtype=float)
    return {
        "test": "linear forecast",
        "n": int(len(observed)),
        "horizon": int(horizon),
        "slope": float(fitted.slope),
        "intercept": float(fitted.intercept),
        "rSquared": float(fitted.rvalue ** 2),
        "p": float(fitted.pvalue),
        "fittedValues": (fitted.intercept + fitted.slope * time).tolist(),
        "forecast": (fitted.intercept + fitted.slope * future_time).tolist(),
        "isSignificant": bool(fitted.pvalue < 0.05),
    }


def analyze_groups(
    groups: list[dict],
    design: str | None = None,
    outcome: str | None = None,
) -> dict[str, Any]:
    groups = [g for g in groups if g.get("values")]
    if len(groups) < 2:
        raise ValueError("Need at least two groups with numeric values.")
    if design == "anova" or (design is None and len(groups) >= 3):
        return one_way_anova(groups, outcome)
    if len(groups) == 2:
        a, b = groups[0]["values"], groups[1]["values"]
        equal = True
        if len(a) >= 3 and len(b) >= 3:
            w, p = stats.levene(a, b, center="median")
            equal = p >= 0.05
        return independent_ttest(groups[0], groups[1], equal_var=equal)
    return one_way_anova(groups, outcome)
