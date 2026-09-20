"""Standalone REST API for deterministic tabular analysis."""
from __future__ import annotations

from contextlib import asynccontextmanager
from io import BytesIO
import logging
import os
import secrets
from threading import Lock, Thread
from typing import Any

from pandas.api.types import is_numeric_dtype, is_object_dtype, is_string_dtype
from fastapi import Depends, FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse

from analysis_service import analyze_dataframe, generate_docx
from charts import make_chart_base64
from data_parser import DataParserError, parse_tabular_text, parse_uploaded_file
from financial_engine import analyze_financial_dataframe


IDENTIFIER_COLUMNS = {
    "id", "rowid", "recordid", "uuid", "index", "sampleid", "idnumber",
    "identifier", "rownumber", "recordnumber", "recordcode",
}
GROUP_COLUMN_HINTS = {
    "treatment", "group", "groups", "arm", "method", "methods", "condition",
    "department", "category", "type", "variant", "segment", "region", "cohort",
}


@asynccontextmanager
async def _lifespan(_: FastAPI):
    _start_telegram_polling()
    yield


app = FastAPI(title="Rowfirst Analysis API", version="1.0.0", lifespan=_lifespan)
logger = logging.getLogger("rowfirst.api")
_telegram_start_lock = Lock()
_telegram_started = False

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


@app.get("/", include_in_schema=False)
@app.get("/health", include_in_schema=False)
async def health() -> dict[str, str]:
    return {"status": "ok", "service": "rowfirst-fastapi"}


@app.exception_handler(Exception)
async def internal_engine_error(_, __: Exception) -> JSONResponse:
    return JSONResponse(status_code=500, content={"detail": "Internal Engine Error"})


def _start_telegram_polling() -> None:
    """Start polling once when the Cloud Run service is configured for Telegram."""
    global _telegram_started
    token = os.getenv("TELEGRAM_BOT_TOKEN") or os.getenv("BOT_TOKEN")
    enabled = os.getenv("TELEGRAM_ENABLE_POLLING", "true").lower() not in {"0", "false", "no"}
    if not token or not enabled:
        return
    with _telegram_start_lock:
        if _telegram_started:
            return
        _telegram_started = True

    def poll() -> None:
        try:
            from bot import create_bot

            create_bot(token).infinity_polling(skip_pending=True)
        except Exception:
            logger.exception("Telegram polling stopped")

    Thread(target=poll, name="telegram-polling", daemon=True).start()
    logger.info("Telegram polling started")


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


async def _load_frame(
    file: UploadFile | None,
    raw_text: str | None,
) -> Any:
    if file is not None:
        frame = parse_uploaded_file(await file.read(), file.filename or "")
    elif raw_text:
        frame = parse_tabular_text(raw_text)
    else:
        raise DataParserError("Provide a CSV file or raw_text")
    return frame


async def _analyze_request(
    file: UploadFile | None,
    raw_text: str | None,
    factor_column: str | None,
    metric_column: str | None,
) -> tuple[list[dict[str, Any]], str]:
    frame = await _load_frame(file, raw_text)
    return _analyze_frame(frame, factor_column, metric_column)


def _analyze_frame(
    frame: Any,
    factor_column: str | None,
    metric_column: str | None,
) -> tuple[list[dict[str, Any]], str]:
    factor_column = _infer_factor_column(frame, factor_column)
    metric_columns = (
        [str(metric_column)]
        if metric_column and metric_column.upper() != "ALL"
        else _numeric_metric_columns(frame, factor_column)
    )
    if not metric_columns:
        raise DataParserError("Could not find any numeric metric columns")
    engines = [
        analyze_dataframe(frame, factor_column, metric, generate_chart=False)
        for metric in metric_columns
    ]
    _add_ranked_batch_charts(engines)
    return engines, factor_column


def _column_key(column: Any) -> str:
    return "".join(character for character in str(column).lower() if character.isalnum())


def _profile(frame: Any) -> dict[str, Any]:
    return {
        "n_rows": int(len(frame)),
        "columns": [str(column) for column in frame.columns],
        "missing": {
            str(column): int(frame[column].isna().sum())
            for column in frame.columns
        },
    }


def _is_identifier_column(column: Any) -> bool:
    return _column_key(column) in IDENTIFIER_COLUMNS


def _profile_mode(frame: Any, factor_column: str) -> dict[str, Any]:
    if factor_column not in frame.columns:
        raise DataParserError(f"Unknown profile column: {factor_column}")
    if _is_identifier_column(factor_column):
        return {
            "mode": "refuse",
            "reason": "identifier_column",
            "profile": _profile(frame),
            "offers": ["Choose a categorical business group such as Department or Treatment."],
            "question": "Which column represents the business group or category?",
        }
    values = frame[factor_column].dropna().astype(str)
    return {
        "mode": "profile",
        "reason": "value_counts",
        "profile": _profile(frame),
        "value_counts": values.value_counts().head(100).to_dict(),
        "offers": ["Choose this column as a grouping factor for analysis."],
    }


def _design_gate(
    frame: Any,
    factor_column: str | None,
    metric_column: str | None,
    mode: str | None,
) -> dict[str, Any] | None:
    if mode == "profile":
        if not factor_column:
            raise DataParserError("profile mode requires factor_column")
        return _profile_mode(frame, factor_column)

    profile = _profile(frame)
    if factor_column and _is_identifier_column(factor_column):
        return {
            "mode": "refuse",
            "reason": "identifier_column",
            "profile": profile,
            "offers": ["Choose a real categorical group, such as Treatment, Department, or Segment."],
            "question": "Which column represents the repeated business group?",
        }

    candidate_factor = factor_column
    if not candidate_factor:
        obvious = [
            str(column) for column in frame.columns
            if any(hint in _column_key(column) for hint in GROUP_COLUMN_HINTS)
            and (is_object_dtype(frame[column]) or is_string_dtype(frame[column]))
        ]
        candidate_factor = obvious[0] if obvious else None
    if not candidate_factor:
        return {
            "mode": "ask",
            "reason": "no_obvious_group_column",
            "profile": profile,
            "offers": ["Select a grouping column", "Run profile mode for column counts"],
            "question": "Which column should define the groups, departments, segments, or treatments?",
        }
    if candidate_factor not in frame.columns:
        raise DataParserError(f"Unknown factor column: {candidate_factor}")
    if not (is_object_dtype(frame[candidate_factor]) or is_string_dtype(frame[candidate_factor])):
        return {
            "mode": "refuse",
            "reason": "factor_not_categorical",
            "profile": profile,
            "offers": ["Choose a categorical text column as the group."],
            "question": "Which text column represents the groups?",
        }

    counts = frame[candidate_factor].dropna().value_counts()
    if len(counts) < 2:
        return {
            "mode": "refuse",
            "reason": "one_group_only",
            "profile": profile,
            "offers": ["Provide at least two repeated groups."],
            "question": "Which column contains at least two business groups?",
        }
    if len(counts) >= len(frame) * 0.9 or counts.min() < 2:
        return {
            "mode": "refuse",
            "reason": "unique_or_singleton_groups",
            "profile": profile,
            "offers": ["Choose a real grouping column, not a row number, ID, company name, or URL."],
            "question": "Which column contains repeated groups suitable for comparison?",
        }

    numeric = [
        str(column) for column in frame.columns
        if is_numeric_dtype(frame[column]) and not _is_identifier_column(column)
    ]
    if metric_column and metric_column.upper() != "ALL" and _is_identifier_column(metric_column):
        return {
            "mode": "refuse",
            "reason": "identifier_column_outcome",
            "profile": profile,
            "offers": ["Choose a numeric business measure, such as Revenue or Score."],
            "question": "Which numeric column is the business outcome?",
        }
    if not numeric:
        return {
            "mode": "ask",
            "reason": "no_numeric_outcome",
            "profile": profile,
            "offers": ["Choose a numeric business measure."],
            "question": "Which numeric column should be analyzed?",
        }
    return None


def _add_ranked_batch_charts(engines: list[dict[str, Any]]) -> None:
    """Render charts only for the most significant successful batch results."""
    if len(engines) <= 1:
        if engines:
            chart_base64 = make_chart_base64(engines[0])
            if chart_base64:
                engines[0]["chart_base64"] = chart_base64
                engines[0]["result"]["chart_base64"] = chart_base64
        return

    successful = [
        engine for engine in engines
        if engine.get("result", {}).get("status", "success") == "success"
        and isinstance(engine.get("result", {}).get("p"), (int, float))
    ]
    chart_limit = 5 if len(engines) > 20 else 10
    ranked = sorted(successful, key=lambda engine: float(engine["result"]["p"]))[:chart_limit]
    for engine in ranked:
        chart_base64 = make_chart_base64(engine)
        if chart_base64:
            engine["chart_base64"] = chart_base64
            engine["result"]["chart_base64"] = chart_base64


def _infer_columns(
    frame: Any,
    factor_column: str | None = None,
    metric_column: str | None = None,
) -> tuple[str, str]:
    if len(frame.columns) < 2:
        raise DataParserError("The table must contain a factor and metric column")

    factor_column = _infer_factor_column(frame, factor_column)
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


def _infer_factor_column(frame: Any, factor_column: str | None = None) -> str:
    if len(frame.columns) < 2:
        raise DataParserError("The table must contain a factor and metric column")
    if factor_column:
        if factor_column not in frame.columns:
            raise DataParserError(f"Unknown factor column: {factor_column}")
        return str(factor_column)
    ignored_factor_names = ("id", "record", "index")
    half_row_count = len(frame) * 0.5
    inferred = next(
        (
            str(column)
            for column in frame.columns
            if not any(term in str(column).lower() for term in ignored_factor_names)
            and (is_object_dtype(frame[column]) or is_string_dtype(frame[column]))
            and frame[column].nunique(dropna=True) < half_row_count
        ),
        None,
    )
    if not inferred:
        raise DataParserError(
            "Could not infer a categorical factor column with fewer than half as many unique values as rows"
        )
    return inferred


def _numeric_metric_columns(frame: Any, factor_column: str) -> list[str]:
    return [
        str(column)
        for column in frame.columns
        if str(column) != factor_column
        and is_numeric_dtype(frame[column])
        and not _is_identifier_column(column)
    ]


def _analysis_payload(engine: dict[str, Any]) -> dict[str, Any]:
    result = engine["result"]
    return {
        "metric": engine.get("metric"),
        "status": engine.get("status", "success"),
        "result_text": engine.get("message", ""),
        "result": result,
        "chart_base64": engine.get("chart_base64"),
        **({"reason": engine["reason"], "descriptive_stats": engine["descriptive_stats"]} if engine.get("status") == "fallback" else {}),
    }


def _public_engine(engine: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in engine.items() if key != "source_frame"}


@app.post("/api/v1/analyze", response_model=None)
async def analyze(
    file: UploadFile | None = File(default=None),
    raw_text: str | None = Form(default=None),
    factor_column: str | None = Form(default=None),
    metric_column: str | None = Form(default=None),
    mode: str | None = Form(default=None),
    response_format: str = Form(default="docx"),
    _: None = Depends(_verify_integration),
) -> StreamingResponse | JSONResponse:
    try:
        frame = await _load_frame(file, raw_text)
        gate_response = _design_gate(frame, factor_column, metric_column, mode)
        if gate_response is not None:
            return JSONResponse(content=gate_response)
        engines, factor = _analyze_frame(frame, factor_column, metric_column)
        if response_format.lower() == "json":
            analyses = [_analysis_payload(engine) for engine in engines]
            payload: dict[str, Any] = {"analyses": analyses, "factor": factor}
            if len(analyses) == 1:
                payload.update(_public_engine(engines[0]))
            return JSONResponse(content=payload)
        if response_format.lower() != "docx":
            raise DataParserError("response_format must be 'docx' or 'json'")
        engine = dict(engines[0])
        engine["results"] = [item["result"] for item in engines]
        engine["result"] = engine["results"][0]
        engine["factor"] = factor
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


@app.post("/api/v1/financial-analysis", response_model=None)
@app.post("/api/v1/financial/analyze", response_model=None)
async def financial_analysis(
    request: Request,
    _: None = Depends(_verify_integration),
) -> JSONResponse:
    """Return deterministic executive KPIs and diagnostics for a financial ledger."""
    try:
        content_type = request.headers.get("content-type", "").lower()
        if content_type.startswith("multipart/") or content_type.startswith("application/x-www-form-urlencoded"):
            form = await request.form()
            file = form.get("file")
            raw_text = form.get("raw_text")
            if file is not None and not hasattr(file, "read"):
                file = None
            if raw_text is not None and not isinstance(raw_text, str):
                raw_text = str(raw_text)
            frame = await _load_frame(file, raw_text)
        else:
            payload = await request.json()
            if isinstance(payload, dict) and isinstance(payload.get("rows"), list):
                payload = payload["rows"]
            if not isinstance(payload, list) or not payload:
                raise DataParserError("Provide a CSV file, raw_text, or a JSON array of ledger rows")
            import pandas as pd

            frame = pd.DataFrame(payload)
        return JSONResponse(content=analyze_financial_dataframe(frame))
    except (DataParserError, ValueError, KeyError, TypeError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
