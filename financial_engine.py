"""Deterministic executive financial analysis for monthly ledger uploads."""
from __future__ import annotations

import base64
from io import BytesIO
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import linregress


DATE_ALIASES = {"date", "month", "period", "monthdate", "transactiondate"}
REVENUE_ALIASES = {"revenue", "revenues", "revenueusd", "netrevenue", "netrevenueusd", "sales", "salesusd", "income", "turnover"}
CASH_ALIASES = {"cashreserves", "cashreservesusd", "cashreserve", "cashreserveusd", "cash", "cashusd", "cashbalance", "endingcash"}
COGS_ALIASES = {"cogs", "costofgoods sold", "costofgoodssold", "costofsales"}
ACTIVE_ALIASES = {"activeclients", "activeclientcount", "activeunits", "activeunitcount", "customers", "units"}
EXPENSE_ALIASES = {
    "operatingexpenses": "operating_expenses",
    "operatingexpensesusd": "operating_expenses",
    "operatingexpense": "operating_expenses",
    "operatingexpenseusd": "operating_expenses",
    "opex": "operating_expenses",
    "payroll": "payroll",
    "payrollusd": "payroll",
    "salaries": "payroll",
    "wages": "payroll",
    "marketingspend": "marketing_spend",
    "marketingspendusd": "marketing_spend",
    "marketing": "marketing_spend",
    "advertising": "marketing_spend",
}


def _key(value: Any) -> str:
    return "".join(character for character in str(value).lower() if character.isalnum())


def _find_column(frame: pd.DataFrame, aliases: set[str]) -> str | None:
    for column in frame.columns:
        if _key(column) in {_key(alias) for alias in aliases}:
            return str(column)
    return None


def _number_series(frame: pd.DataFrame, column: str) -> pd.Series:
    return pd.to_numeric(frame[column], errors="coerce")


def _money(value: float) -> str:
    return f"${value:,.2f}"


def _regression(values: pd.Series) -> dict[str, float | None]:
    clean = pd.to_numeric(values, errors="coerce").dropna().astype(float)
    if len(clean) < 2 or clean.nunique() < 2:
        return {"slope": None, "r_squared": None}
    fit = linregress(np.arange(len(clean), dtype=float), clean.to_numpy())
    return {"slope": float(fit.slope), "r_squared": float(fit.rvalue ** 2)}


def _diagnostics(frame: pd.DataFrame, date_column: str, revenue_column: str, expense_columns: list[str]) -> list[dict[str, Any]]:
    warnings: list[dict[str, Any]] = []
    required = [date_column, revenue_column, *expense_columns]
    for column in required:
        missing = int(frame[column].isna().sum())
        if missing:
            warnings.append({"type": "missing_values", "column": column, "count": missing})
    dates = pd.to_datetime(frame[date_column], format="mixed", errors="coerce")
    invalid_dates = int(dates.isna().sum())
    if invalid_dates:
        warnings.append({"type": "invalid_dates", "column": date_column, "count": invalid_dates})
    valid_dates = dates.dropna().dt.to_period("M").sort_values().unique()
    if len(valid_dates) > 1:
        expected = pd.period_range(valid_dates[0], valid_dates[-1], freq="M")
        missing_periods = [str(period) for period in expected if period not in valid_dates]
        if missing_periods:
            warnings.append({"type": "mismatched_date_sequence", "missing_periods": missing_periods})
    for column in [revenue_column, *expense_columns]:
        values = _number_series(frame, column)
        if values.notna().sum() == 0:
            warnings.append({"type": "non_numeric_values", "column": column})
        elif values.dropna().std(ddof=0) == 0:
            warnings.append({"type": "zero_variance", "column": column})
    return warnings


def _expense_columns(frame: pd.DataFrame) -> list[str]:
    return [str(column) for column in frame.columns if _key(column) in EXPENSE_ALIASES]


def _expense_field(column: str) -> str:
    return EXPENSE_ALIASES[_key(column)]


def is_financial_dataframe(frame: pd.DataFrame) -> bool:
    """Return whether a frame has the minimum shape of a financial time series."""
    if frame.empty:
        return False
    return bool(
        _find_column(frame, DATE_ALIASES)
        and _find_column(frame, REVENUE_ALIASES)
        and _expense_columns(frame)
    )


def _monthly_frame(frame: pd.DataFrame, date_column: str, revenue_column: str, cash_column: str | None, cogs_column: str | None, active_column: str | None, expense_columns: list[str]) -> pd.DataFrame:
    working = pd.DataFrame(index=frame.index)
    working["period"] = pd.to_datetime(frame[date_column], format="mixed", errors="coerce").dt.to_period("M").astype("string")
    working["revenue"] = _number_series(frame, revenue_column)
    for column in expense_columns:
        field = _expense_field(column)
        working[field] = _number_series(frame, column)
    if cash_column:
        working["cash_reserves"] = _number_series(frame, cash_column)
    if cogs_column:
        working["cogs"] = _number_series(frame, cogs_column)
    if active_column:
        working["active_units"] = _number_series(frame, active_column)
    working = working.dropna(subset=["period"])
    aggregations: dict[str, str] = {column: "sum" for column in working.columns if column not in {"period", "cash_reserves", "active_units"}}
    if cash_column:
        aggregations["cash_reserves"] = "last"
    if active_column:
        aggregations["active_units"] = "last"
    return working.groupby("period", sort=True).agg(aggregations).reset_index()


def _chart_base64(monthly: pd.DataFrame) -> str | None:
    if monthly.empty:
        return None
    figure, revenue_axis = plt.subplots(figsize=(11, 5.5), facecolor="#120d1f")
    expense_axis = revenue_axis.twinx()
    periods = monthly["period"].astype(str).tolist()
    x_values = np.arange(len(monthly), dtype=float)
    revenue = monthly["revenue"].to_numpy(dtype=float)
    expenses = monthly["total_expenses"].to_numpy(dtype=float)
    revenue_axis.set_facecolor("#120d1f")
    revenue_axis.plot(x_values, revenue, color="#d8b4fe", marker="o", label="Net Revenue", linewidth=2.5)
    expense_axis.plot(x_values, expenses, color="#fb7185", marker="o", label="Operating Expenses", linewidth=2.5)
    if len(monthly) >= 2:
        revenue_fit = np.polyfit(x_values, revenue, 1)
        expense_fit = np.polyfit(x_values, expenses, 1)
        revenue_axis.plot(x_values, np.polyval(revenue_fit, x_values), color="#f5d0fe", linestyle="--", alpha=0.85, label="Revenue trend")
        expense_axis.plot(x_values, np.polyval(expense_fit, x_values), color="#fecdd3", linestyle="--", alpha=0.85, label="Expense trend")
    mean_expense = float(np.mean(expenses))
    deviation = float(np.std(expenses, ddof=0))
    expense_axis.fill_between(x_values, mean_expense, mean_expense + 2 * deviation, color="#f43f5e", alpha=0.12, label="2 SD expense band")
    revenue_axis.set_xticks(x_values, periods, rotation=45, ha="right")
    revenue_axis.set_ylabel("Revenue", color="#f5d0fe")
    expense_axis.set_ylabel("Operating expenses", color="#fecdd3")
    revenue_axis.tick_params(colors="#f5d0fe")
    expense_axis.tick_params(colors="#fecdd3")
    revenue_axis.grid(axis="y", color="#6b5a83", alpha=0.25)
    revenue_axis.set_title("Monthly Revenue vs Operating Expenses", color="#f8fafc")
    handles_a, labels_a = revenue_axis.get_legend_handles_labels()
    handles_b, labels_b = expense_axis.get_legend_handles_labels()
    revenue_axis.legend(handles_a + handles_b, labels_a + labels_b, loc="upper left", facecolor="#241631", labelcolor="#f8fafc")
    figure.tight_layout()
    buffer = BytesIO()
    figure.savefig(buffer, format="png", dpi=160, facecolor=figure.get_facecolor())
    plt.close(figure)
    return base64.b64encode(buffer.getvalue()).decode("ascii")


def _encode_figure(figure: Any) -> str:
    buffer = BytesIO()
    figure.savefig(buffer, format="png", dpi=160, facecolor=figure.get_facecolor())
    plt.close(figure)
    return base64.b64encode(buffer.getvalue()).decode("ascii")


def _chart_suite_base64(monthly: pd.DataFrame) -> list[str]:
    """Return the executive chart suite while preserving the legacy primary chart."""
    if monthly.empty:
        return []
    periods = monthly["period"].astype(str).tolist()
    x_values = np.arange(len(monthly), dtype=float)
    charts = [_chart_base64(monthly)]

    figure, axis = plt.subplots(figsize=(11, 5.5), facecolor="#120d1f")
    axis.set_facecolor("#120d1f")
    axis.axhline(0, color="#c4b5fd", linewidth=1, alpha=0.5)
    axis.bar(x_values - 0.18, monthly["ebitda"], width=0.36, color="#a78bfa", label="EBITDA")
    axis.bar(x_values + 0.18, monthly["net_operating_cash_flow"], width=0.36, color="#fb7185", label="Net operating cash flow")
    axis.set_xticks(x_values, periods, rotation=45, ha="right")
    axis.set_ylabel("Cash flow", color="#f8fafc")
    axis.set_title("Monthly EBITDA and Operating Cash Flow", color="#f8fafc")
    axis.tick_params(colors="#f8fafc")
    axis.grid(axis="y", color="#6b5a83", alpha=0.25)
    axis.legend(facecolor="#241631", labelcolor="#f8fafc")
    charts.append(_encode_figure(figure))

    figure, margin_axis = plt.subplots(figsize=(11, 5.5), facecolor="#120d1f")
    margin_axis.set_facecolor("#120d1f")
    margin_axis.plot(x_values, monthly["gross_margin"], color="#67e8f9", marker="o", linewidth=2.5, label="Gross margin")
    margin_axis.plot(x_values, monthly["operating_margin"], color="#fbbf24", marker="o", linewidth=2.5, label="Operating margin")
    if "active_units" in monthly:
        unit_axis = margin_axis.twinx()
        unit_axis.plot(x_values, monthly["active_units"], color="#86efac", marker="s", linewidth=2.2, label="Active clients/units")
        unit_axis.set_ylabel("Active clients/units", color="#86efac")
        unit_axis.tick_params(colors="#86efac")
        handles_b, labels_b = unit_axis.get_legend_handles_labels()
    else:
        handles_b, labels_b = [], []
    margin_axis.set_xticks(x_values, periods, rotation=45, ha="right")
    margin_axis.set_ylabel("Margin (%)", color="#f8fafc")
    margin_axis.set_title("Monthly Margins and Active Units", color="#f8fafc")
    margin_axis.tick_params(colors="#f8fafc")
    margin_axis.grid(axis="y", color="#6b5a83", alpha=0.25)
    handles_a, labels_a = margin_axis.get_legend_handles_labels()
    margin_axis.legend(handles_a + handles_b, labels_a + labels_b, facecolor="#241631", labelcolor="#f8fafc")
    charts.append(_encode_figure(figure))
    return [chart for chart in charts if chart]


def analyze_financial_dataframe(frame: pd.DataFrame) -> dict[str, Any]:
    """Analyze a ledger with one row per transaction or month."""
    if frame.empty:
        raise ValueError("Financial ledger is empty")
    date_column = _find_column(frame, DATE_ALIASES)
    revenue_column = _find_column(frame, REVENUE_ALIASES)
    cash_column = _find_column(frame, CASH_ALIASES)
    cogs_column = _find_column(frame, COGS_ALIASES)
    active_column = _find_column(frame, ACTIVE_ALIASES)
    expense_columns = _expense_columns(frame)
    if not date_column or not revenue_column:
        raise ValueError("Financial ledger requires a Date or Month column and a Revenue column")
    if not expense_columns:
        raise ValueError("Financial ledger requires Operating Expenses, Payroll, or Marketing Spend")

    warnings = _diagnostics(frame, date_column, revenue_column, expense_columns)
    monthly = _monthly_frame(frame, date_column, revenue_column, cash_column, cogs_column, active_column, expense_columns)
    if monthly.empty:
        raise ValueError("No valid financial periods were found")
    expense_fields = [_expense_field(column) for column in expense_columns]
    monthly["total_expenses"] = monthly[expense_fields].sum(axis=1, min_count=1)
    monthly["ebitda"] = monthly["revenue"] - monthly["total_expenses"]
    monthly["net_operating_cash_flow"] = monthly["ebitda"]
    monthly["gross_profit"] = monthly["revenue"] - monthly.get("cogs", monthly["total_expenses"])
    monthly["operating_margin"] = np.where(monthly["revenue"] != 0, monthly["ebitda"] / monthly["revenue"] * 100, np.nan)
    monthly["gross_margin"] = np.where(monthly["revenue"] != 0, monthly["gross_profit"] / monthly["revenue"] * 100, np.nan)
    monthly["net_margin"] = monthly["operating_margin"]
    monthly["mom_growth_rate"] = monthly["revenue"].pct_change() * 100

    total_net_revenue = float(monthly["revenue"].sum())
    avg_monthly_burn = float(monthly["total_expenses"].mean())
    avg_mom_growth = float(monthly["mom_growth_rate"].dropna().mean()) if monthly["mom_growth_rate"].notna().any() else 0.0
    total_expense_mean = float(monthly["total_expenses"].mean())
    legacy_expense_mean = float(monthly["operating_expenses"].mean()) if "operating_expenses" in monthly else total_expense_mean
    current_cash = float(monthly["cash_reserves"].dropna().iloc[-1]) if "cash_reserves" in monthly and monthly["cash_reserves"].notna().any() else None
    runway = current_cash / total_expense_mean if current_cash is not None and total_expense_mean > 0 else None
    revenue_regression = _regression(monthly["revenue"])
    growth_values = monthly["mom_growth_rate"].replace([np.inf, -np.inf], np.nan).dropna()
    growth_regression = _regression(growth_values)

    for column in expense_columns:
        normalized = _expense_field(column)
        if normalized not in monthly:
            continue
        values = monthly[normalized].astype(float)
        mean = float(values.mean())
        standard_deviation = float(values.std(ddof=0))
        threshold = mean + 2 * standard_deviation
        if standard_deviation > 0:
            for index, value in values.items():
                if value > threshold:
                    warnings.append({"type": "expenditure_spike", "category": normalized, "period": str(monthly.loc[index, "period"]), "value": float(value), "threshold": threshold})
    for _, row in monthly[monthly["net_operating_cash_flow"] < 0].iterrows():
        warnings.append({"type": "negative_cash_flow", "period": str(row["period"]), "amount": float(row["net_operating_cash_flow"]), "message": "Net operating cash flow is below zero."})
    if monthly["revenue"].eq(0).any():
        warnings.append({"type": "zero_division_hazard", "column": revenue_column, "message": "Margin and growth rates are undefined for zero-revenue periods."})
    critical_types = {"negative_cash_flow", "zero_division_hazard"}
    for warning in warnings:
        warning["severity"] = "critical" if warning["type"] in critical_types else "warning"

    latest = monthly.iloc[-1]
    latest_growth = float(latest["mom_growth_rate"]) if pd.notna(latest["mom_growth_rate"]) else 0.0
    latest_margin = float(latest["operating_margin"]) if pd.notna(latest["operating_margin"]) else 0.0
    latest_ebitda = float(latest["ebitda"])
    narrative = (
        f"Financial Performance Summary: Net Revenue: {_money(float(latest['revenue']))} "
        f"(Representing a {latest_growth:.2f}% MoM growth rate via linear trend analysis). "
        f"Operating Margin: {latest_margin:.2f}% (EBITDA: {_money(latest_ebitda)}). "
        f"Capital Efficiency: Current cash burn yields a Runway of {runway:.2f} months based on deterministic mean expense calculations."
        if runway is not None else
        f"Financial Performance Summary: Net Revenue: {_money(float(latest['revenue']))} "
        f"(Representing a {latest_growth:.2f}% MoM growth rate via linear trend analysis). "
        f"Operating Margin: {latest_margin:.2f}% (EBITDA: {_money(latest_ebitda)}). "
        "Capital Efficiency: Current cash burn yields an undetermined Runway because mean expenses are zero or cash reserves are unavailable."
    )
    records = monthly.replace({np.nan: None}).to_dict(orient="records")
    return {
        "ok": True,
        "analysis_type": "executive_financial",
        "periods": records,
        "kpis": {
            "net_revenue": float(latest["revenue"]),
            "total_period_revenue": total_net_revenue,
            "mean_monthly_revenue": float(monthly["revenue"].mean()),
            "period_over_period_growth_rate": latest_growth,
            "mom_growth_rate": latest_growth,
            "revenue_trend": revenue_regression,
            "growth_trend": growth_regression,
            "gross_profit": float(latest["gross_profit"]),
            "gross_profit_margin": float(latest["gross_margin"]) if pd.notna(latest["gross_margin"]) else None,
            "net_profit_margin": float(latest["net_margin"]) if pd.notna(latest["net_margin"]) else None,
            "ebitda": latest_ebitda,
            "operating_margin": latest_margin,
            "mean_monthly_expenses": legacy_expense_mean,
            "mean_total_operating_expenses": total_expense_mean,
            "mean_monthly_cash_burn": total_expense_mean,
            "avg_mom_growth": avg_mom_growth,
            "current_cash_reserves": current_cash,
            "runway_months": runway,
        },
        "template_context": {
            "total_net_revenue": total_net_revenue,
            "avg_monthly_burn": avg_monthly_burn,
            "avg_mom_growth": avg_mom_growth,
        },
        "diagnostics": {"warnings": warnings, "warning_count": len(warnings)},
        "executive_summary": narrative,
        "executive_text_blocks": [narrative],
        "chart_base64": _chart_base64(monthly),
        "charts_base64": _chart_suite_base64(monthly),
    }
