"""Thin Telegram client for the deterministic Rowfirst analysis service."""
from __future__ import annotations

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


UNSUPPORTED_MEDIA_MESSAGE = (
    "⚠️ Unsupported file type. Please send CSV, Excel, PDF, PNG, JPG, WEBP, TIFF, "
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


def _analyze_and_send(bot: Any, message: Any, frame: Any) -> None:
    factor_column, metric_column = _choose_columns(frame)
    engine = analyze_dataframe(frame, factor_column, metric_column)
    last_engines[message.chat.id] = engine
    with tempfile.TemporaryDirectory(prefix="rowfirst-results-") as directory:
        path = Path(directory) / "Rowfirst_Results.docx"
        generate_docx(engine, path)
        with path.open("rb") as report:
            bot.send_document(message.chat.id, report, caption="Rowfirst_Results.docx")


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
