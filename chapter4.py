"""Backward-compatible report imports; implementation lives in analysis_service."""

from analysis_service import to_markdown, write_docx, write_markdown, write_pdf

__all__ = ["to_markdown", "write_docx", "write_markdown", "write_pdf"]
