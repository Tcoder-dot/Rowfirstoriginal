"""Exam A/B/C against SciPy gold. Run: python3 tests/test_gold.py"""
import math
import os
import sys
import tempfile
import zipfile
from pathlib import Path

import pandas as pd
from scipy import stats
from statsmodels.formula.api import ols
from statsmodels.stats.anova import anova_lm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import bot as bot_module
from bot import (
    _detect_singleton_factor,
    _multivariate_table_route,
    _send_analysis,
    _python_script_to_dataframe,
    _summarize_dataframe,
    _singleton_groups_for_frame,
    classify_columns,
    _coerce_p_value_text,
)
from data_explorer import run_explorer_action, run_explorer_query
from PIL import Image
from docx import Document

from chapter4 import write_docx, to_markdown
from charts import make_charts
from handle import handle_analyze
from document_extractor import extract_document_table, sanitize_extracted_table


def near(a, b, tol=0.02):
    return abs(a - b) <= tol


def chart_regression_failures():
    """Protect the chart type chosen for each supported reporting outcome."""
    group = handle_analyze({
        "text": "Brining: 4.22, 4.18, 4.25, 4.20\nQuick pickling: 3.71, 3.78, 3.74, 3.80",
    })
    paired = handle_analyze({
        "text": "id,before,after\n1,72,78\n2,75,79\n3,70,74\n4,80,84\n5,68,73",
    })
    regression = handle_analyze({
        "text": "hours,score\n2.1,14\n2.4,16\n2.8,15\n3.1,19\n3.5,21\n3.9,24",
    })
    correlation = {
        "ok": True,
        "result": {
            "test": "pearson",
            "xName": "hours",
            "yName": "score",
            "outcome": "score",
            "pairs": [(2.1, 14.0), (2.4, 16.0), (2.8, 15.0), (3.1, 19.0)],
        },
    }
    count_share = handle_analyze({"text": "18  2\n11  9"})
    cases = [
        ("group mean", group, "group means"),
        ("paired", paired, "matched before/after"),
        ("regression", regression, "scatter plot"),
        ("correlation", correlation, "scatter plot"),
        ("count-share", count_share, "count-share pie chart"),
    ]
    failures = []
    for label, engine, caption_fragment in cases:
        with tempfile.TemporaryDirectory(prefix="rowfirst-chart-test-") as output_dir:
            charts = make_charts(engine, output_dir)
            if len(charts) != 1:
                failures.append(f"{label} chart count was {len(charts)}")
                continue
            chart = charts[0]
            if caption_fragment not in chart.get("caption", ""):
                failures.append(f"{label} selected caption {chart.get('caption')!r}")
            if not Path(chart["path"]).is_file():
                failures.append(f"{label} chart file was not created")
    return failures


def reporting_regression_failures():
    """Protect DOCX generation when chart metadata points at a missing file."""
    engine = handle_analyze({
        "text": "Brining: 4.22, 4.18, 4.25, 4.20\nQuick pickling: 3.71, 3.78, 3.74, 3.80",
    })
    failures = []
    with tempfile.TemporaryDirectory(prefix="rowfirst-docx-test-") as output_dir:
        output_dir = Path(output_dir)
        missing_docx = output_dir / "Rowfirst_Results_missing_chart.docx"
        engine["charts"] = [{
            "result_index": 0,
            "path": str(output_dir / "does-not-exist.png"),
            "caption": "missing chart",
        }]
        try:
            write_docx(engine, missing_docx)
        except Exception as exc:
            failures.append(f"missing chart blocked DOCX creation: {type(exc).__name__}: {exc}")
        if not missing_docx.is_file():
            failures.append("missing chart did not produce a DOCX")

        chart_docx = output_dir / "Rowfirst_Results_with_chart.docx"
        engine["charts"] = make_charts(engine, output_dir / "charts")
        try:
            write_docx(engine, chart_docx)
        except Exception as exc:
            failures.append(f"chart embedding blocked DOCX creation: {type(exc).__name__}: {exc}")
        if not chart_docx.is_file():
            failures.append("chart embedding did not produce a DOCX")
        if chart_docx.is_file():
            with zipfile.ZipFile(chart_docx) as archive:
                if not any(name.startswith("word/media/") for name in archive.namelist()):
                    failures.append("generated chart was not embedded in DOCX")
    return failures


def telegram_regression_failures():
    """Protect the RESULTS/QA/prompt blocks and Telegram chart photo delivery."""
    class FakeChat:
        id = 42

    class FakeMessage:
        chat = FakeChat()

    class FakeBot:
        def __init__(self):
            self.replies = []
            self.photos = []

        def reply_to(self, message, text):
            self.replies.append(text)

        def send_photo(self, chat_id, image_file, caption=""):
            self.photos.append((chat_id, image_file.read(), caption))

    engine = handle_analyze({
        "text": "Brining: 4.22, 4.18, 4.25, 4.20\nQuick pickling: 3.71, 3.78, 3.74, 3.80",
    })
    engine["qa"] = {"warnings": ["test warning"], "errors": []}
    failures = []
    with tempfile.TemporaryDirectory(prefix="rowfirst-telegram-test-") as output_dir:
        engine["charts"] = make_charts(engine, output_dir)
        bot = FakeBot()
        _send_analysis(bot, FakeMessage(), engine)
        if sum(text.startswith("RESULTS\n") for text in bot.replies) != 1:
            failures.append("RESULTS header was not sent exactly once")
        if sum(text.startswith("QA\n") for text in bot.replies) != 1:
            failures.append("QA block was not sent exactly once")
        if not any(text == "Want a Results Document (Word)? Reply YES" for text in bot.replies):
            failures.append("YES Results Document prompt was not sent")
        if len(bot.photos) != 1 or not bot.photos[0][1]:
            failures.append("chart PNG was not delivered as a Telegram photo")
    return failures


def telegram_token_startup_failures():
    """Protect actionable startup diagnostics when TeleBot rejects a token."""
    original_telebot = bot_module.telebot

    class FakeTelebot:
        @staticmethod
        def TeleBot(token):
            raise ValueError("Token must contain a colon")

    failures = []
    bot_module.telebot = FakeTelebot
    try:
        try:
            bot_module._create_telegram_bot("not-a-token")
        except SystemExit as exc:
            if "TELEGRAM_BOT_TOKEN is malformed" not in str(exc):
                failures.append(f"malformed token diagnostic was {exc}")
        else:
            failures.append("malformed token did not stop startup")
    finally:
        bot_module.telebot = original_telebot
    return failures


def variable_classification_regressions():
    fails = []

    crm_df = pd.DataFrame({
        "Row ID": list(range(1, 9)),
        "Customer ID": [1001, 1002, 1003, 1004, 1005, 1006, 1007, 1008],
        "Created Date": [
            "2024-01-02", "2024-01-03", "2024-01-04", "2024-01-05",
            "2024-01-06", "2024-01-07", "2024-01-08", "2024-01-09",
        ],
        "Region": ["North", "North", "South", "South", "East", "East", "West", "West"],
        "Revenue": [4200.0, 4305.0, 4520.0, 4700.0, 4865.0, 4920.0, 5040.0, 5155.0],
        "Lead Score": [61, 68, 72, 79, 74, 88, 91, 85],
    })
    crm_route = _multivariate_table_route(crm_df)
    if crm_route is None or crm_route.get("kind") != "multivariate":
        fails.append(f"CRM route should stay multivariate but returned {crm_route!r}")
    else:
        bad_metrics = {"Row ID", "Customer ID", "Created Date"}
        metric_names = set(crm_route.get("outcomes", []))
        if bad_metrics & metric_names:
            fails.append(f"CRM metric selector leaked metadata/date columns: {metric_names}")
        if "Revenue" not in metric_names or "Lead Score" not in metric_names:
            fails.append(f"CRM metric selector lost valid metrics: {metric_names}")

    serial_date_df = pd.DataFrame({
        "Index": list(range(1, 7)),
        "Group": ["A", "A", "B", "B", "C", "C"],
        "Date Created": [44001, 44012, 44023, 44034, 44045, 44056],
        "Dose_mg": [20.2, 21.1, 22.4, 23.5, 24.8, 25.6],
        "Outcome_Count": [102, 118, 130, 150, 162, 175],
    })
    serial_route = _multivariate_table_route(serial_date_df)
    if serial_route is None or serial_route.get("kind") != "multivariate":
        fails.append(f"serial-date route should stay multivariate but returned {serial_route!r}")
    else:
        metric_names = set(serial_route.get("outcomes", []))
        if "Date Created" in metric_names:
            fails.append(f"Excel serial date leaked into metrics: {metric_names}")
        if "Dose_mg" not in metric_names or "Outcome_Count" not in metric_names:
            fails.append(f"valid continuous metrics missing from serial-date route: {metric_names}")

    return fails


def dataset1_guard_regressions():
    fails = []
    dataset1 = pd.DataFrame({
        "Index": [1, 2, 3, 4, 5, 6],
        "Company_Name": ["Acme", "Beta", "Acme", "Gamma", "Beta", "Gamma"],
        "Created_Date": ["2024-01-02", "2024-01-03", "2024-01-04", "2024-01-05", "2024-01-06", "2024-01-07"],
        "Renewal_Date": ["2024-02-01", "2024-02-02", "2024-02-03", "2024-02-04", "2024-02-05", "2024-02-06"],
        "Lead_Score": [61, 68, 72, 79, 74, 88],
    })
    route = _multivariate_table_route(dataset1)
    if route is None:
        fails.append("Dataset 1 should halt to manual mapping rather than returning None")
    elif route.get("kind") != "needs_mapping":
        fails.append(f"Dataset 1 should require manual mapping, got {route!r}")
    elif set(route.get("factors", [])) != {"Company_Name"}:
        fails.append(f"Dataset 1 factor candidate detection failed: {route.get('factors')!r}")
    return fails


def data_health_regressions():
    fails = []

    messy = pd.DataFrame({
        "Row ID": [1, 2, 3, 4, 5, 6],
        "Customer": ["Acme Inc", "Acme Inc", "Beta Ltd", "Beta Ltd", "Gamma", "Gamma"],
        "Region": ["North", "North", "South", "South", "East", ""],
        "Revenue": ["$1200", "$1,250", "£900", "£950", "N/A", "1100"],
        "Created On": ["2024-01-02", "01/03/2024", "2024-Q1", "2024-01-05", "44050", "2024-01-07"],
        "Units Sold": [10, 12, None, 15, 20, 21],
        "Order ID": [101, 101, 203, 204, 205, 206],
    })
    messy = messy.drop_duplicates(subset=["Order ID"], keep="first")
    summary = _summarize_dataframe(messy)
    if "Data Health Card" not in summary:
        fails.append("health card heading missing")
    if "Missing Cells" not in summary or "Duplicate Rows" not in summary:
        fails.append("health card missing key metrics")
    if "Currency symbols detected" not in summary or "Mixed date formats" not in summary:
        fails.append("health card failed to flag formatting anomalies")

    route = _multivariate_table_route(messy)
    if route is None or route.get("kind") != "multivariate":
        fails.append(f"messy dataset should remain multivariate but got {route!r}")
    else:
        leaked = {"Row ID", "Order ID", "Created On"}
        if leaked & set(route.get("outcomes", [])):
            fails.append(f"metadata/date columns leaked into outcome selection: {route.get('outcomes')}")

    return fails


def document_narrative_regressions():
    fails = []
    engine = {
        "ok": True,
        "topic": "Microbial viability screening",
        "ingested": {"format": "labelled", "groups": [
            {"name": "Control", "values": [7.1, 7.4, 7.5, 7.2, 7.3]},
            {"name": "Treatment_A", "values": [8.8, 9.1, 9.3, 9.2, 8.9]},
            {"name": "Treatment_B", "values": [10.2, 10.5, 10.6, 10.3, 10.4]},
        ]},
        "results": [{
            "test": "one-way anova",
            "parameter": "total_viable_count_log_cfu",
            "groups": [
                {"name": "Control", "n": 5, "mean": 7.3, "sd": 0.17},
                {"name": "Treatment_A", "n": 5, "mean": 9.06, "sd": 0.19},
                {"name": "Treatment_B", "n": 5, "mean": 10.4, "sd": 0.15},
            ],
            "dfb": 2,
            "dfw": 12,
            "F": 145.2,
            "p": 0.0001,
            "isSignificant": True,
            "effectSize": {"etaSquared": 0.96, "omegaSquared": 0.94},
            "postHoc": {"comparisons": [
                {"group1": "Control", "group2": "Treatment_A", "pAdjusted": 0.002, "reject": True},
                {"group1": "Control", "group2": "Treatment_B", "pAdjusted": 0.0001, "reject": True},
            ]},
        }],
    }
    markdown = to_markdown(engine)
    if "the submitted outcome" in markdown.lower() or "the variable" in markdown.lower() or "test metric" in markdown.lower():
        fails.append("dynamic narrative still contains fallback placeholder wording")
    if "Total Viable Count (log CFU/g)" not in markdown:
        fails.append("human-readable outcome label was not injected into the narrative")
    with tempfile.TemporaryDirectory(prefix="rowfirst-docx-narrative-") as output_dir:
        output_path = Path(output_dir) / "Rowfirst_Chapter4.docx"
        write_docx(engine, output_path)
        with zipfile.ZipFile(output_path) as archive:
            xml = archive.read("word/document.xml").decode("utf-8", errors="ignore").lower()
            if "the submitted outcome" in xml or "the variable" in xml or "test metric" in xml:
                fails.append("DOCX output still contains generic placeholder text")
            if "total viable count" not in xml and "total viable count (log cfu/g)" not in xml:
                fails.append("DOCX narrative did not include the cleaned variable label")
    return fails


def assumption_and_ingestion_regressions():
    fails = []

    df = pd.DataFrame({
        "Patient_ID": [1, 2, 3, 4, 5, 6],
        "Treatment": ["A", "A", "B", "B", "C", "C"],
        "log_CFU_g": [2.0, 2.1, 6.5, 6.9, 10.1, 9.8],
        "Moisture_pct": [12.0, 13.5, 18.0, 18.1, 25.4, 26.0],
    })
    classified = classify_columns(df)
    if set(classified["categorical_factors"]) != {"Treatment"}:
        fails.append(f"categorical factors misclassified: {classified['categorical_factors']}")
    if set(classified["numeric_metrics"]) != {"log_CFU_g", "Moisture_pct"}:
        fails.append(f"numeric metrics misclassified: {classified['numeric_metrics']}")

    summary = _summarize_dataframe(df)
    if "Detected Categorical Factors" not in summary or "Detected Numeric Metrics" not in summary:
        fails.append("ingestion card did not show dynamic factor and metric sections")
    if "Treatment" not in summary or "log CFU/g" not in summary:
        fails.append(f"dynamic ingestion summary missed detected columns: {summary}")

    formatted = _coerce_p_value_text(8.1038e-131)
    if formatted != "p < .001":
        fails.append(f"p-value formatter returned {formatted!r}, expected 'p < .001'")
    return fails


def explorer_regressions():
    fails = []
    frame = pd.DataFrame({
        "Region": ["North", "North", "South", "South", "East", "East"],
        "Revenue": [1200, 1400, 900, 950, 1100, 1300],
        "Moisture_pct": [10.0, 12.0, 8.0, 9.0, 11.0, 13.0],
    })

    action = run_explorer_action(frame, "top_bottom")
    if "North" not in action or "highest" not in action.lower():
        fails.append(f"top-bottom action output was not ranked as expected: {action!r}")

    summary = run_explorer_query(frame, "average moisture")
    if "The average" not in summary or "10.50" not in summary:
        fails.append(f"average summary output was not deterministic: {summary!r}")

    corr = run_explorer_query(frame, "correlation between Revenue and Moisture_pct")
    if "r =" not in corr:
        fails.append(f"correlation output was not generated: {corr!r}")

    try:
        run_explorer_query(frame, "highest nonsense by region")
    except ValueError as exc:
        if "Available columns" not in str(exc):
            fails.append(f"fuzzy query fallback did not explain missing match: {exc!r}")
    else:
        fails.append("ambiguous explorer query did not fail with a helpful message")
    return fails


def feature_regressions():
    fails = []

    script = '''
import pandas as pd
import numpy as np
rng = np.random.default_rng(7)
df = pd.DataFrame({
    "group": ["A", "A", "B", "B", "C", "C"],
    "score": [10, 11, 14, 15, 9, 12],
    "age": [20, 21, 23, 24, 19, 28],
})
'''
    frame = _python_script_to_dataframe(script)
    if frame is None or frame.shape != (6, 3):
        fails.append(f"python script did not produce a 6x3 dataframe: {frame.shape if frame is not None else frame}")

    summary = _summarize_dataframe(frame)
    if "Shape" not in summary or "Categorical Factors" not in summary or "Numeric Metrics" not in summary:
        fails.append("dataset summary card missing required sections")

    singleton_frame = pd.DataFrame({
        "group": ["A", "A", "B", "C"],
        "score": [10, 11, 12, 13],
    })
    singleton_groups = _singleton_groups_for_frame(singleton_frame, "group")
    expected = [{"name": "B", "n": 1}, {"name": "C", "n": 1}]
    if singleton_groups != expected:
        fails.append(f"singleton detection returned {singleton_groups!r}, expected {expected!r}")

    unique_df = pd.DataFrame({
        "Record_ID": [f"R{i}" for i in range(1, 11)],
        "Department": ["A", "A", "B", "B", "A", "B", "A", "B", "A", "B"],
        "Value": [10, 12, 9, 13, 11, 15, 12, 14, 11, 16],
    })
    route = _multivariate_table_route(unique_df)
    if route is None or route.get("kind") != "multivariate":
        fails.append(f"ID filtering route should remain multivariate, got {route!r}")
    elif "Record_ID" in route.get("factors", []):
        fails.append("ID column leaked into factor list")
    if _detect_singleton_factor(unique_df) is not None:
        fails.append("singleton detection should ignore unique row identifiers")

    stress_path = Path(__file__).resolve().parent.parent / "stress_test_200col_1000rows.csv"
    if stress_path.exists():
        stress_df = pd.read_csv(stress_path)
        route = _multivariate_table_route(stress_df)
        if route is None or route.get("kind") != "multivariate":
            fails.append(f"stress dataset route unexpectedly rejected: {route!r}")
        else:
            factors = route.get("factors", [])
            if "Department" not in factors or "Customer_Segment" not in factors:
                fails.append(f"valid department/segment factors missing from stress route: {factors[:12]}")
            if any(col in factors for col in ["Record_ID", "Customer_ID", "Order_ID"]):
                fails.append(f"identifier columns reached stress route factors: {factors[:12]}")
            singleton_factor = _detect_singleton_factor(stress_df)
            if singleton_factor is not None:
                fails.append(f"stress dataset singleton detection should ignore IDs and valid factors, got {singleton_factor!r}")

    flag_df = pd.DataFrame({
        "Department": ["A", "A", "B", "B", "A", "B", "A", "B", "A", "B"],
        "Is_Loyal": [0, 1, 0, 1, 0, 1, 1, 0, 1, 1],
        "Net_Sales_Value": [120, 135, 90, 140, 125, 150, 110, 160, 130, 170],
        "Basket_Item_Count": [3, 4, 2, 5, 4, 6, 3, 5, 4, 7],
    })
    route = _multivariate_table_route(flag_df)
    if route is None or route.get("kind") != "multivariate":
        fails.append(f"flagged multivariate route should stay multivariate: {route!r}")
    elif "Is_Loyal" in route.get("factors", []) or "Is_Loyal" in route.get("outcomes", []):
        fails.append(f"binary flag column leaked into route factors/outcomes: {route!r}")
    elif "Net_Sales_Value" not in route.get("outcomes", []):
        fails.append(f"continuous outcome not preserved in route: {route!r}")

    preview_engine = {
        "ok": True,
        "ingested": {"format": "labelled", "groups": [{"name": "A", "values": [10, 11, 12, 13, 14, 15, 16, 17, 18, 19]}, {"name": "B", "values": [20, 21, 22, 23, 24, 25, 26, 27, 28, 29]}]},
        "results": [{"test": "one-way anova", "groups": [{"name": "A", "n": 10, "mean": 15.0, "sd": 3.0}, {"name": "B", "n": 10, "mean": 25.0, "sd": 3.0}], "F": 42.0, "dfb": 1, "dfw": 18, "p": 0.0001, "isSignificant": True, "parameter": "Net_Sales_Value"}],
    }
    preview_markdown = to_markdown(preview_engine)
    if "Data Sample Preview (First 5 Observations)" not in preview_markdown:
        fails.append("word/pdf markdown preview heading missing")
    if "10 | 11 | 12 | 13 | 14 | 15 | 16 | 17 | 18 | 19" in preview_markdown:
        fails.append("raw preview leaked full observation rows into markdown output")
    return fails


def extraction_regressions():
    fails = []
    with tempfile.TemporaryDirectory(prefix="rowfirst-doc-extract-") as output_dir:
        path = Path(output_dir) / "sample_table.docx"
        document = Document()
        table = document.add_table(rows=3, cols=3)
        table.cell(0, 0).text = "Treatment"
        table.cell(0, 1).text = "Dose\nmg"
        table.cell(0, 2).text = "Result"
        table.cell(1, 0).text = "A"
        table.cell(1, 1).text = " 20 \n mg "
        table.cell(1, 2).text = " 10.5 "
        table.cell(2, 0).text = "B"
        table.cell(2, 1).text = "30\nmg"
        table.cell(2, 2).text = "12.4"
        document.save(path)

        extracted = extract_document_table(path)
        if extracted is None or extracted.empty:
            fails.append("docx extraction did not create a dataframe")
        else:
            if list(extracted.columns) != ["Treatment", "Dose mg", "Result"]:
                fails.append(f"docx columns were not normalized correctly: {list(extracted.columns)}")
            if extracted.iloc[0].tolist()[:2] != ["A", "20 mg"]:
                fails.append(f"docx cell sanitation was not normalized: {extracted.iloc[0].tolist()}")

        sanitised = sanitize_extracted_table({"headers": ["Dose", "Result"], "rows": [[" 1,0 ", "O.8"], ["1l", "1.0"]]})
        if sanitised is None or sanitised["rows"][0][0] != "1.0":
            fails.append("sanitizer did not clean OCR artifacts and numeric noise")

        blank_image = Path(output_dir) / "blank_image.png"
        Image.new("RGB", (200, 200), "white").save(blank_image)
        if extract_document_table(blank_image) is not None:
            fails.append("blank image should be rejected as malformed extraction input")
    return fails


def main():
    fails = []

    A = handle_analyze({"text": "Brining: 4.22, 4.18, 4.25, 4.20\nQuick pickling: 3.71, 3.78, 3.74, 3.80"})
    r = A["result"]
    if not A["ok"]:
        fails.append("A not ok: " + A.get("error", ""))
    else:
        if not near(r["t"], 18.1396, 0.02):
            fails.append(f"A t {r['t']}")
        if r["p"] > 1e-5:
            fails.append(f"A p {r['p']}")
        if int(round(r["df"])) != 6:
            fails.append(f"A df {r['df']}")

    B = handle_analyze({"text": "Fresh: 91.2, 91.5, 90.9, 91.3\nBrining: 87.6, 88.1, 87.9, 88.3\nQuick pickling: 90.4, 90.1, 89.8, 90.6"})
    r = B["result"]
    if not B["ok"]:
        fails.append("B not ok: " + B.get("error", ""))
    else:
        if r.get("test") != "one-way anova":
            fails.append(f"B test {r.get('test')}")
        if not near(r["F"], 121.2766, 0.05):
            fails.append(f"B F {r['F']}")
        if r["p"] > 1e-5:
            fails.append(f"B p {r['p']} (must be ~3e-7, not 0.003)")
        if r.get("dfb") != 2 or r.get("dfw") != 9:
            fails.append(f"B df {r.get('dfb')},{r.get('dfw')}")

    C_result = handle_analyze({"text": "18  2\n11  9"})
    r = C_result["result"]
    if not C_result["ok"]:
        fails.append("C not ok: " + C_result.get("error", ""))
    else:
        p = r.get("p") or r.get("chi2pUncorrected")
        if p is None or not (0.01 <= p <= 0.04):
            fails.append(f"C p {p}")

    D = handle_analyze({"text": "id,before,after\n1,72,78\n2,75,79\n3,70,74\n4,80,84\n5,68,73"})
    r = D.get("result", {})
    if not D.get("ok") or r.get("test") != "paired-t":
        fails.append(f"D paired result {r}")
    else:
        if not near(r["t"], -11.5, 0.001) or not near(r["p"], 0.000326, 0.00001):
            fails.append(f"D t/p {r.get('t')}/{r.get('p')}")
        if r.get("df") != 4:
            fails.append(f"D df {r.get('df')}")

    E = handle_analyze({
        "text": (
            "method,day,score\n"
            "A,1,10\nA,1,11\nA,1,9\nA,2,12\nA,2,13\nA,2,11\n"
            "B,1,20\nB,1,21\nB,1,19\nB,2,22\nB,2,23\nB,2,21"
        )
    })
    r = E.get("result", {})
    if not E.get("ok") or r.get("test") != "two-way anova" or len(r.get("effects", [])) != 3:
        fails.append(f"E two-way result {r}")
    else:
        expected = anova_lm(
            ols("_y ~ C(_a) * C(_b)", data={
                "_a": ["A"] * 6 + ["B"] * 6,
                "_b": [1, 1, 1, 2, 2, 2] * 2,
                "_y": [10, 11, 9, 12, 13, 11, 20, 21, 19, 22, 23, 21],
            }).fit(),
            typ=2,
        )
        expected_p = [float(expected.loc[key, "PR(>F)"]) for key in ("C(_a)", "C(_b)", "C(_a):C(_b)")]
        actual_p = [effect["p"] for effect in r["effects"]]
        if any(not near(actual, target, 1e-10) for actual, target in zip(actual_p, expected_p)):
            fails.append(f"E p-values {actual_p} != {expected_p}")

    F = handle_analyze({"text": "hours,score\n2.1,14\n2.4,16\n2.8,15\n3.1,19\n3.5,21\n3.9,24"})
    r = F.get("result", {})
    expected_regression = stats.linregress([2.1, 2.4, 2.8, 3.1, 3.5, 3.9], [14, 16, 15, 19, 21, 24])
    if not F.get("ok") or r.get("test") != "simple linear regression":
        fails.append(f"F regression result {r}")
    elif not near(r["slope"], expected_regression.slope, 1e-10) or not near(r["r"], expected_regression.rvalue, 1e-10):
        fails.append(f"F slope/r {r.get('slope')}/{r.get('r')}")

    fails.extend(explorer_regressions())
    fails.extend(assumption_and_ingestion_regressions())
    fails.extend(feature_regressions())
    fails.extend(variable_classification_regressions())
    fails.extend(data_health_regressions())
    fails.extend(chart_regression_failures())
    fails.extend(reporting_regression_failures())
    fails.extend(telegram_regression_failures())
    fails.extend(telegram_token_startup_failures())

    if fails:
        print("FAIL")
        for f in fails:
            print(" -", f)
        sys.exit(1)
    print("PASS A/B/C")
    print("A", A["message"])
    print("---")
    print("B", B["message"])
    print("---")
    print("C", C_result["message"])
    print("---")
    print("PASS D/E/F")
    print("D", D["message"])
    print("---")
    print("E", E["message"])
    print("---")
    print("F", F["message"])
    print("---")
    print("PASS chart selection, DOCX resilience/embedding, and Telegram reporting")


if __name__ == "__main__":
    main()
