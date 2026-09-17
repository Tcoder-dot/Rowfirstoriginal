"""Regression checks for the deterministic service boundary."""
from __future__ import annotations

import tempfile
from pathlib import Path

import pandas as pd

from analysis_service import analyze_dataframe, generate_docx
from data_parser import DataParserError, parse_csv_buffer, parse_image, parse_pdf, parse_tabular_text


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
