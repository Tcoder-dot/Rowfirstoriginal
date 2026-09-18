"""Regression checks for the deterministic service boundary."""
from __future__ import annotations

import os
import base64
from io import BytesIO
import tempfile
import zipfile
from pathlib import Path

import pandas as pd
from fastapi.testclient import TestClient

from analysis_service import analyze_dataframe, generate_docx
from api import app
from data_parser import (
    DataParserError,
    parse_csv_buffer,
    parse_excel_buffer,
    parse_image,
    parse_pdf,
    parse_tabular_text,
    parse_uploaded_file,
)
from handle import build_breakdown


def test_parser_delimiters() -> None:
    assert list(parse_csv_buffer(b"Group,Score\nA,1\nB,2\n").columns) == ["Group", "Score"]
    assert parse_tabular_text("Group\tScore\nA\t1").shape == (1, 2)
    assert parse_tabular_text("Group Score\nA 1").shape == (1, 2)


def test_labeled_text_and_analysis() -> None:
    frame = parse_tabular_text("A: 1, 2, 3\nB: 4, 5, 6")
    engine = analyze_dataframe(frame, "Factor", "Metric")
    assert engine["ok"] is True
    assert engine["factor"] == "Factor"
    assert engine["metric"] == "Metric"
    assert "n caveat" not in engine["breakdown"]
    assert "Sample-size caveat:" in engine["breakdown"]


def test_api_inference_skips_id_columns_and_requires_metric_variance() -> None:
    from api import _infer_columns

    frame = pd.DataFrame({
        "Record_ID": [f"R{index:03d}" for index in range(20)],
        "Teaching_Method": ["A"] * 10 + ["B"] * 10,
        "Constant_Index": list(range(20)),
        "Exam_Score": list(range(60, 80)),
    })
    assert _infer_columns(frame) == ("Teaching_Method", "Constant_Index")

    frame["Constant_Index"] = 1
    assert _infer_columns(frame) == ("Teaching_Method", "Exam_Score")

    frame["Exam_Score"] = 1
    try:
        _infer_columns(frame)
    except DataParserError as exc:
        assert "non-zero variance" in str(exc)
    else:
        raise AssertionError("constant metrics should not be inferred")


def test_docx_generation() -> None:
    frame = pd.DataFrame({"Treatment": ["A", "A", "B", "B"], "Value": [1, 2, 4, 5]})
    engine = analyze_dataframe(frame, "Treatment", "Value")
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "results.docx"
        assert generate_docx(engine, path) == str(path)
        assert path.is_file()


def test_engine_chart_is_returned_and_embedded_in_docx() -> None:
    frame = pd.DataFrame({"Treatment": ["A", "A", "B", "B"], "Value": [1, 2, 4, 5]})
    engine = analyze_dataframe(frame, "Treatment", "Value")
    chart_bytes = base64.b64decode(engine["chart_base64"])
    assert chart_bytes.startswith(b"\x89PNG\r\n\x1a\n")
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "results.docx"
        generate_docx(engine, path)
        with zipfile.ZipFile(path) as archive:
            media = [name for name in archive.namelist() if name.startswith("word/media/")]
            assert media
            assert chart_bytes in [archive.read(name) for name in media]


def test_api_batch_returns_success_and_descriptive_fallbacks() -> None:
    os.environ["ROWFIRST_ID"] = "demo-id"
    os.environ["ROWFIRST_SECRET_KEY"] = "demo-secret"
    text = """Record_ID,Teaching_Method,Exam_Score,ID_Number
R01,A,85.2,1
R02,A,88.5,1
R03,A,86.1,1
R04,B,72.4,2
R05,B,75.1,2
R06,B,71.8,2
R07,C,92.5,3
R08,C,94.1,3
R09,C,91.8,3
"""
    response = TestClient(app).post(
        "/api/v1/analyze",
        data={"raw_text": text, "metric_column": "ALL", "response_format": "json"},
        headers={"X-Rowfirst-Id": "demo-id", "X-Rowfirst-Secret-Key": "demo-secret"},
    )
    assert response.status_code == 200
    analyses = response.json()["analyses"]
    by_metric = {analysis["metric"]: analysis for analysis in analyses}
    assert by_metric["Exam_Score"]["status"] == "success"
    assert by_metric["Exam_Score"]["chart_base64"].startswith("iVBORw0KGgo")
    assert "ID_Number" not in by_metric


def test_api_gate_refuses_crm_row_id_and_index_design() -> None:
    os.environ["ROWFIRST_ID"] = "demo-id"
    os.environ["ROWFIRST_SECRET_KEY"] = "demo-secret"
    text = "Company,Website,Revenue,Index,Row ID\n" + "\n".join(
        f"Company {index},example{index}.com,{100 + index},{index},{index}"
        for index in range(1, 101)
    )
    response = TestClient(app).post(
        "/api/v1/analyze",
        data={"raw_text": text, "metric_column": "ALL", "response_format": "json"},
        headers={"X-Rowfirst-Id": "demo-id", "X-Rowfirst-Secret-Key": "demo-secret"},
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["mode"] in {"ask", "refuse"}
    assert payload["profile"]["n_rows"] == 100
    assert payload["reason"] in {"no_obvious_group_column", "unique_or_singleton_groups"}


def test_api_profile_mode_returns_value_counts_without_testing() -> None:
    os.environ["ROWFIRST_ID"] = "demo-id"
    os.environ["ROWFIRST_SECRET_KEY"] = "demo-secret"
    response = TestClient(app).post(
        "/api/v1/analyze",
        data={
            "raw_text": "Department,Revenue\nEngineering,10\nEngineering,12\nSales,8\nSales,9\n",
            "factor_column": "Department",
            "mode": "profile",
            "response_format": "json",
        },
        headers={"X-Rowfirst-Id": "demo-id", "X-Rowfirst-Secret-Key": "demo-secret"},
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["mode"] == "profile"
    assert payload["value_counts"] == {"Engineering": 2, "Sales": 2}


def test_api_batch_limits_charts_for_large_batches() -> None:
    os.environ["ROWFIRST_ID"] = "demo-id"
    os.environ["ROWFIRST_SECRET_KEY"] = "demo-secret"
    headers = {"X-Rowfirst-Id": "demo-id", "X-Rowfirst-Secret-Key": "demo-secret"}
    rows = ["Treatment," + ",".join(f"Metric_{index}" for index in range(21))]
    for row_index, treatment in enumerate(("A", "B", "C") * 4):
        values = [str(row_index + index * 0.01) for index in range(21)]
        rows.append(f"{treatment}," + ",".join(values))
    response = TestClient(app).post(
        "/api/v1/analyze",
        data={"raw_text": "\n".join(rows), "metric_column": "ALL", "response_format": "json"},
        headers=headers,
    )
    assert response.status_code == 200
    analyses = response.json()["analyses"]
    assert len(analyses) == 21
    assert sum(analysis["chart_base64"] is not None for analysis in analyses) == 5


def test_large_batch_narrative_summarizes_non_significant_metrics() -> None:
    from analysis_service import _discussion, _interpretations

    significant = {
        "test": "one-way anova",
        "outcome": "Discovery",
        "p": 0.001,
        "isSignificant": True,
        "F": 12.0,
        "dfb": 2,
        "dfw": 9,
        "groups": [],
    }
    non_significant = [
        {"test": "one-way anova", "outcome": f"Metric_{index}", "p": 0.4, "isSignificant": False, "groups": []}
        for index in range(11)
    ]
    engine = {
        "ok": True,
        "results": [significant, *non_significant],
        "result": significant,
        "ingested": {"format": "long", "factor": "Treatment", "outcome": "Discovery", "groups": []},
    }
    interpretation = _interpretations(engine)[0]
    discussion = " ".join(_discussion(engine))
    assert "Discovery: group means were not reported" in interpretation
    assert "No significant differences were observed for 11 other variables tested" in interpretation
    assert "Metric_10: group means were not reported" not in interpretation
    assert "Metric_10: group means were not reported" not in discussion


def test_excel_and_pdf_uploads_are_parsed() -> None:
    excel_buffer = BytesIO()
    pd.DataFrame({"Treatment": ["A", "A", "B", "B"], "Value": [1, 2, 4, 5]}).to_excel(excel_buffer, index=False)
    excel = parse_excel_buffer(excel_buffer.getvalue(), "study.xlsx")
    assert list(excel.columns) == ["Treatment", "Value"]
    assert parse_uploaded_file(excel_buffer.getvalue(), "study.xlsx").shape == (4, 2)

    from reportlab.pdfgen.canvas import Canvas

    pdf_buffer = BytesIO()
    pdf = Canvas(pdf_buffer)
    for index, line in enumerate(("Treatment,Value", "A,1", "A,2", "B,4", "B,5")):
        pdf.drawString(72, 760 - index * 16, line)
    pdf.save()
    parsed_pdf = parse_pdf(pdf_buffer.getvalue())
    assert list(parsed_pdf.columns) == ["Treatment", "Value"]


def test_image_upload_uses_ocr(monkeypatch) -> None:
    from PIL import Image
    import pytesseract

    image_buffer = BytesIO()
    Image.new("RGB", (120, 40), "white").save(image_buffer, format="PNG")
    monkeypatch.setattr(pytesseract, "image_to_string", lambda *args, **kwargs: "Treatment,Value\nA,1\nB,2")
    assert list(parse_image(image_buffer.getvalue()).columns) == ["Treatment", "Value"]


def test_bad_table_is_rejected() -> None:
    try:
        parse_tabular_text("single-value")
    except DataParserError:
        pass
    else:
        raise AssertionError("invalid delimiter was accepted")


def test_public_api_uses_shared_engine_and_docx_output() -> None:
    os.environ["ROWFIRST_ID"] = "demo-id"
    os.environ["ROWFIRST_SECRET_KEY"] = "demo-secret"
    client = TestClient(app)
    headers = {"X-Rowfirst-ID": "demo-id", "X-Rowfirst-Secret": "demo-secret"}

    json_response = client.post(
        "/api/v1/analyze",
        data={
            "raw_text": "Treatment\tValue\nA\t1\nA\t2\nB\t4\nB\t5\n",
            "factor_column": "Treatment",
            "metric_column": "Value",
            "response_format": "json",
        },
        headers=headers,
    )
    assert json_response.status_code == 200
    payload = json_response.json()
    assert payload["ok"] is True
    assert payload["factor"] == "Treatment"
    assert payload["metric"] == "Value"

    docx_response = client.post(
        "/api/v1/analyze",
        data={
            "raw_text": "Treatment\tValue\nA\t1\nA\t2\nB\t4\nB\t5\n",
            "factor_column": "Treatment",
            "metric_column": "Value",
            "response_format": "docx",
        },
        headers=headers,
    )
    assert docx_response.status_code == 200
    assert docx_response.headers["content-type"].startswith("application/vnd.openxmlformats-officedocument.wordprocessingml.document")
    assert len(docx_response.content) > 1000


def test_public_api_accepts_form_query_and_bearer_credentials() -> None:
    os.environ["ROWFIRST_ID"] = "demo-id"
    os.environ["ROWFIRST_SECRET_KEY"] = "demo-secret"
    client = TestClient(app)
    request_data = {
        "raw_text": "Treatment\tValue\nA\t1\nA\t2\nB\t4\nB\t5\n",
        "factor_column": "Treatment",
        "metric_column": "Value",
        "response_format": "json",
    }

    form_response = client.post(
        "/api/v1/analyze",
        data={**request_data, "rowfirst_id": "demo-id", "rowfirst_secret_key": "demo-secret"},
    )
    assert form_response.status_code == 200

    query_response = client.post(
        "/api/v1/analyze?rowfirst_id=demo-id&rowfirst_secret_key=demo-secret",
        data=request_data,
    )
    assert query_response.status_code == 200

    bearer_response = client.post(
        "/api/v1/analyze",
        data={**request_data, "rowfirst_id": "demo-id"},
        headers={"Authorization": "Bearer demo-secret"},
    )
    assert bearer_response.status_code == 200

    alias_header_response = client.post(
        "/api/v1/analyze",
        data=request_data,
        headers={"X-Rowfirst-Client-Id": "demo-id", "Rowfirst-Secret-Key": "demo-secret"},
    )
    assert alias_header_response.status_code == 200


def test_public_api_infers_columns_when_not_supplied() -> None:
    os.environ["ROWFIRST_ID"] = "demo-id"
    os.environ["ROWFIRST_SECRET_KEY"] = "demo-secret"
    response = TestClient(app).post(
        "/api/v1/analyze",
        data={
            "raw_text": "Treatment,Value\nA,1\nA,2\nA,3\nB,4\nB,5\nB,6\n",
            "response_format": "json",
        },
        headers={"X-Rowfirst-Id": "demo-id", "X-Rowfirst-Secret-Key": "demo-secret"},
    )
    assert response.status_code == 200
    assert response.json()["factor"] == "Treatment"
    assert response.json()["metric"] == "Value"
