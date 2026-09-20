"""Thin Telegram client for the deterministic Rowfirst analysis service."""
from __future__ import annotations

import base64
from io import BytesIO
import os
import tempfile
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from threading import Thread
from typing import Any

from pandas.api.types import is_numeric_dtype

try:
    import telebot
    from telebot import types
except ImportError:  # pragma: no cover
    telebot = None
    types = None

from analysis_service import analyze_dataframe, generate_docx
from data_parser import DataParserError, parse_tabular_text, parse_uploaded_file
from financial_engine import analyze_financial_dataframe, is_financial_dataframe


UNSUPPORTED_MEDIA_MESSAGE = (
    "⚠️ Unsupported file type. Please send CSV, Excel, PDF, Word, PNG, JPG, WEBP, TIFF, "
    "or BMP data, or paste your raw table text."
)
last_engines: dict[int, dict[str, Any]] = {}


class _HealthHandler(BaseHTTPRequestHandler):
    def do_GET(self) -> None:
        self.send_response(200)
        self.send_header("Content-Type", "text/plain; charset=utf-8")
        self.end_headers()
        self.wfile.write(b"ok")

    def log_message(self, format: str, *args: Any) -> None:
        return


def _start_health_server() -> None:
    raw_port = os.getenv("PORT")
    if not raw_port:
        return
    try:
        port = int(raw_port)
    except ValueError as exc:
        raise RuntimeError(f"Invalid PORT value: {raw_port!r}") from exc
    if port <= 0:
        raise RuntimeError(f"Invalid PORT value: {raw_port!r}")
    server = ThreadingHTTPServer(("0.0.0.0", port), _HealthHandler)
    Thread(target=server.serve_forever, daemon=True).start()


def _choose_columns(frame: Any) -> tuple[str, str]:
    if len(frame.columns) < 2:
        raise DataParserError("The table must contain a factor and metric column")
    numeric = [column for column in frame.columns if is_numeric_dtype(frame[column])]
    metric = numeric[0] if numeric else frame.columns[1]
    factor = next((column for column in frame.columns if column != metric), frame.columns[0])
    return str(factor), str(metric)


def _send_text_chunks(bot: Any, chat_id: int, text: str) -> None:
    cleaned = str(text or "").strip()
    if not cleaned:
        return
    for start in range(0, len(cleaned), 4000):
        bot.send_message(chat_id, cleaned[start:start + 4000])


def _send_engine_outputs(bot: Any, message: Any, engine: dict[str, Any]) -> None:
    chat_id = message.chat.id
    if engine.get("analysis_type") == "executive_financial":
        _send_text_chunks(bot, chat_id, engine.get("executive_summary", ""))
        warnings = engine.get("diagnostics", {}).get("warnings", [])
        if warnings:
            lines = ["Diagnostics:"]
            lines.extend(
                f"- {warning.get('message') or warning.get('type') or warning.get('period', 'Review required')}"
                for warning in warnings
            )
            _send_text_chunks(bot, chat_id, "\n".join(lines))
    else:
        breakdown = engine.get("breakdown", "")
        result_text = engine.get("message", "")
        _send_text_chunks(bot, chat_id, breakdown)
        if result_text and result_text != breakdown:
            _send_text_chunks(bot, chat_id, f"Verified results:\n{result_text}")

    charts = engine.get("charts_base64") or [engine.get("chart_base64")]
    chart_titles = (
        "Monthly revenue and operating expenses",
        "Monthly EBITDA and operating cash flow",
        "Monthly margins and active units",
    )
    for index, chart_base64 in enumerate(charts):
        if not chart_base64:
            continue
        caption = chart_titles[index] if index < len(chart_titles) else "Analysis chart"
        bot.send_photo(chat_id, BytesIO(base64.b64decode(chart_base64)), caption=caption)


def _analyze_and_send(bot: Any, message: Any, frame: Any) -> None:
    if is_financial_dataframe(frame):
        engine = analyze_financial_dataframe(frame)
        filename = "Rowfirst_Financial_Report.docx"
    else:
        factor_column, metric_column = _choose_columns(frame)
        engine = analyze_dataframe(frame, factor_column, metric_column)
        filename = "Rowfirst_Results.docx"
    last_engines[message.chat.id] = engine
    with tempfile.TemporaryDirectory(prefix="rowfirst-results-") as directory:
        path = Path(directory) / filename
        generate_docx(engine, path)
        with path.open("rb") as report:
            bot.send_document(message.chat.id, report, caption=filename)
    _send_engine_outputs(bot, message, engine)


def _send_invoice(bot: Any, message: Any) -> None:
    provider_token = os.getenv("TELEGRAM_PAYMENT_PROVIDER_TOKEN")
    if not provider_token:
        bot.reply_to(message, "Payments are not configured. Send a CSV or paste your data to analyze it.")
        return
    bot.send_invoice(
        message.chat.id,
        "Rowfirst analysis",
        "Unlock a deterministic statistical Results document.",
        f"rowfirst:{message.chat.id}",
        provider_token,
        "USD",
        [types.LabeledPrice("Analysis report", 500)],
    )


def create_bot(token: str) -> Any:
    if telebot is None:
        raise RuntimeError("pyTelegramBotAPI is required to run the Telegram client")
    bot = telebot.TeleBot(token)

    @bot.message_handler(commands=["start", "help"])
    def start(message: Any) -> None:
        bot.reply_to(message, "Send a .csv file or paste comma-, tab-, or space-delimited data.")

    @bot.message_handler(commands=["buy"])
    def buy(message: Any) -> None:
        _send_invoice(bot, message)

    @bot.pre_checkout_query_handler(func=lambda _: True)
    def pre_checkout(query: Any) -> None:
        bot.answer_pre_checkout_query(query.id, ok=True)

    @bot.message_handler(func=lambda message: getattr(message, "successful_payment", None) is not None)
    def successful_payment(message: Any) -> None:
        bot.reply_to(message, "Payment received. Send your CSV or pasted table to generate the report.")

    @bot.message_handler(func=lambda message: getattr(message, "content_type", "") in {"photo", "document"})
    def media(message: Any) -> None:
        filename = str(getattr(getattr(message, "document", None), "file_name", "") or "")
        try:
            if getattr(message, "content_type", "") == "photo":
                info = bot.get_file(message.photo[-1].file_id)
                filename = "uploaded-image.jpg"
            else:
                info = bot.get_file(message.document.file_id)
            frame = parse_uploaded_file(bot.download_file(info.file_path), filename)
            _analyze_and_send(bot, message, frame)
        except (DataParserError, ValueError) as exc:
            bot.reply_to(message, f"⚠️ {exc}")
        except Exception:
            bot.reply_to(message, "⚠️ Could not process that CSV. Please check the delimiter and columns.")

    @bot.message_handler(content_types=["text"])
    def text(message: Any) -> None:
        try:
            frame = parse_tabular_text(message.text)
            _analyze_and_send(bot, message, frame)
        except (DataParserError, ValueError) as exc:
            bot.reply_to(message, f"⚠️ {exc}")
        except Exception:
            bot.reply_to(message, "⚠️ Could not analyze that table. Please send headers followed by data rows.")

    return bot


def main() -> None:
    token = os.getenv("TELEGRAM_BOT_TOKEN") or os.getenv("BOT_TOKEN")
    if not token:
        raise RuntimeError("Set TELEGRAM_BOT_TOKEN before starting the Telegram client")
    _start_health_server()
    create_bot(token).infinity_polling(skip_pending=True)


if __name__ == "__main__":
    main()
