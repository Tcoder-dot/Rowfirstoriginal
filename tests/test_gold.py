"""Regression checks for the deterministic service boundary."""
from __future__ import annotations

import os
import tempfile
from pathlib import Path

import pandas as pd
from fastapi.testclient import TestClient

from analysis_service import analyze_dataframe, generate_docx
from api import app
from data_parser import DataParserError, parse_csv_buffer, parse_image, parse_pdf, parse_tabular_text
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


def test_docx_generation() -> None:
    frame = pd.DataFrame({"Treatment": ["A", "A", "B", "B"], "Value": [1, 2, 4, 5]})
    engine = analyze_dataframe(frame, "Treatment", "Value")
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "results.docx"
        assert generate_docx(engine, path) == str(path)
        assert path.is_file()


def test_unsupported_inputs_are_explicit() -> None:
    for parser in (parse_pdf, parse_image):
        try:
            parser(b"data")
        except NotImplementedError as exc:
            assert str(exc) == "PDF and Image parsing are scheduled for v2.0"
        else:
            raise AssertionError("unsupported parser did not raise")


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
            "raw_text": "Treatment,Value\nA,1\nA,2\nB,4\nB,5\n",
            "response_format": "json",
        },
        headers={"X-Rowfirst-Id": "demo-id", "X-Rowfirst-Secret-Key": "demo-secret"},
    )
    assert response.status_code == 200
    assert response.json()["factor"] == "Treatment"
    assert response.json()["metric"] == "Value"
