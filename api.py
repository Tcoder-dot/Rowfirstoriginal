"""Standalone REST API for deterministic tabular analysis."""
from __future__ import annotations

import tempfile
import os
import secrets
from pathlib import Path
from typing import Any

from fastapi import Depends, FastAPI, File, Form, Header, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from starlette.background import BackgroundTask

from analysis_service import analyze_dataframe, generate_docx
from data_parser import DataParserError, parse_csv_buffer, parse_tabular_text


app = FastAPI(title="Rowfirst Analysis API", version="1.0.0")

cors_origins = [
    origin.strip()
    for origin in os.getenv(
        "ROWFIRST_CORS_ORIGINS",
        "http://localhost:3000,http://localhost:5173",
    ).split(",")
    if origin.strip()
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=cors_origins,
    allow_credentials=False,
    allow_methods=["POST", "OPTIONS"],
    allow_headers=["*"],
)


def _verify_integration(
    rowfirst_id: str | None = Header(default=None, alias="X-Rowfirst-ID"),
    secret_key: str | None = Header(default=None, alias="X-Rowfirst-Secret"),
) -> None:
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
    factor_column: str,
    metric_column: str,
) -> dict[str, Any]:
    if file is not None:
        frame = parse_csv_buffer(await file.read(), file.filename or "")
    elif raw_text:
        frame = parse_tabular_text(raw_text)
    else:
        raise DataParserError("Provide a CSV file or raw_text")
    return analyze_dataframe(frame, factor_column, metric_column)


def _public_engine(engine: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in engine.items() if key != "source_frame"}


@app.post("/api/v1/analyze")
async def analyze(
    file: UploadFile | None = File(default=None),
    raw_text: str | None = Form(default=None),
    factor_column: str = Form(...),
    metric_column: str = Form(...),
    response_format: str = Form(default="docx"),
    _: None = Depends(_verify_integration),
) -> FileResponse | JSONResponse:
    try:
        engine = await _analyze_request(file, raw_text, factor_column, metric_column)
        if response_format.lower() == "json":
            return JSONResponse(content=_public_engine(engine))
        if response_format.lower() != "docx":
            raise DataParserError("response_format must be 'docx' or 'json'")
        with tempfile.NamedTemporaryFile(
            prefix="rowfirst-results-",
            suffix=".docx",
            delete=False,
        ) as temporary_file:
            output = Path(temporary_file.name)
        generate_docx(engine, output)
        return FileResponse(
            output,
            media_type="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            filename="Rowfirst_Results.docx",
            background=BackgroundTask(output.unlink, missing_ok=True),
        )
    except (DataParserError, ValueError, KeyError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc