"""Standalone REST API for deterministic tabular analysis."""
from __future__ import annotations

from io import BytesIO
import os
import secrets
from typing import Any

from pandas.api.types import is_numeric_dtype, is_object_dtype, is_string_dtype
from fastapi import Depends, FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse

from analysis_service import analyze_dataframe, generate_docx
from data_parser import DataParserError, parse_csv_buffer, parse_tabular_text


app = FastAPI(title="Rowfirst Analysis API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    # The browser rejects wildcard origins when credentialed cookies are enabled.
    # This API authenticates with headers/form values, not browser cookies.
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"],
)


@app.exception_handler(Exception)
async def internal_engine_error(_, __: Exception) -> JSONResponse:
    return JSONResponse(status_code=500, content={"detail": "Internal Engine Error"})


async def _verify_integration(request: Request) -> None:
    form = await request.form()
    headers = request.headers
    rowfirst_id = (
        headers.get("X-Rowfirst-Id")
        or headers.get("X-Rowfirst-ID")
        or headers.get("Rowfirst-Id")
        or headers.get("X-Rowfirst-Client-Id")
        or form.get("rowfirst_id")
        or request.query_params.get("rowfirst_id")
    )
    secret_key = (
        headers.get("X-Rowfirst-Secret-Key")
        or headers.get("X-Rowfirst-Secret")
        or headers.get("Rowfirst-Secret-Key")
        or form.get("rowfirst_secret_key")
        or request.query_params.get("rowfirst_secret_key")
    )
    if not secret_key:
        authorization = headers.get("Authorization", "")
        scheme, _, token = authorization.partition(" ")
        if scheme.lower() == "bearer":
            secret_key = token.strip() or None

    expected_id = os.getenv("ROWFIRST_ID")
    expected_secret = os.getenv("ROWFIRST_SECRET_KEY")
    if not expected_id or not expected_secret:
        raise HTTPException(status_code=500, detail="Rowfirst integration credentials are not configured")
    if not rowfirst_id or not secret_key:
        raise HTTPException(status_code=401, detail="Rowfirst ID and secret key are required")
    if not (
        secrets.compare_digest(rowfirst_id, expected_id)
        and secrets.compare_digest(secret_key, expected_secret)
    ):
        raise HTTPException(status_code=401, detail="Invalid Rowfirst integration credentials")


async def _analyze_request(
    file: UploadFile | None,
    raw_text: str | None,
    factor_column: str | None,
    metric_column: str | None,
) -> dict[str, Any]:
    if file is not None:
        frame = parse_csv_buffer(await file.read(), file.filename or "")
    elif raw_text:
        frame = parse_tabular_text(raw_text)
    else:
        raise DataParserError("Provide a CSV file or raw_text")
    factor_column, metric_column = _infer_columns(frame, factor_column, metric_column)
    return analyze_dataframe(frame, factor_column, metric_column)


def _infer_columns(
    frame: Any,
    factor_column: str | None = None,
    metric_column: str | None = None,
) -> tuple[str, str]:
    if len(frame.columns) < 2:
        raise DataParserError("The table must contain a factor and metric column")

    if not factor_column:
        ignored_factor_names = ("id", "record", "index")
        half_row_count = len(frame) * 0.5
        factor_column = next(
            (
                str(column)
                for column in frame.columns
                if not any(term in str(column).lower() for term in ignored_factor_names)
                and (is_object_dtype(frame[column]) or is_string_dtype(frame[column]))
                and frame[column].nunique(dropna=True) < half_row_count
            ),
            None,
        )
    if not factor_column:
        raise DataParserError(
            "Could not infer a categorical factor column with fewer than half as many unique values as rows"
        )

    if not metric_column:
        metric_column = next(
            (
                str(column)
                for column in frame.columns
                if is_numeric_dtype(frame[column]) and float(frame[column].var()) > 0
            ),
            None,
        )
    if not metric_column:
        raise DataParserError("Could not infer a numeric metric column with non-zero variance")
    return str(factor_column), str(metric_column)


def _public_engine(engine: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in engine.items() if key != "source_frame"}


@app.post("/api/v1/analyze", response_model=None)
async def analyze(
    file: UploadFile | None = File(default=None),
    raw_text: str | None = Form(default=None),
    factor_column: str | None = Form(default=None),
    metric_column: str | None = Form(default=None),
    response_format: str = Form(default="docx"),
    _: None = Depends(_verify_integration),
) -> StreamingResponse | JSONResponse:
    try:
        engine = await _analyze_request(file, raw_text, factor_column, metric_column)
        if response_format.lower() == "json":
            return JSONResponse(content=_public_engine(engine))
        if response_format.lower() != "docx":
            raise DataParserError("response_format must be 'docx' or 'json'")
        buffer = generate_docx(engine)
        if not isinstance(buffer, BytesIO):
            raise RuntimeError("DOCX generator returned an invalid buffer")
        buffer.seek(0)
        headers = {"Content-Disposition": 'attachment; filename="Rowfirst_Results.docx"'}
        return StreamingResponse(
            buffer,
            media_type="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            headers=headers,
        )
    except (DataParserError, ValueError, KeyError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
