"""Logging configuration and custom exceptions."""

import logging

from rich.logging import RichHandler

# import sys


# ============================================================================
# Logger Setup
# ============================================================================


def setup_logger(name: str = "sequence_converter", level: int = logging.INFO) -> logging.Logger:
    """
    rich.logging.RichHandlerを使ったロガーを設定

    Args:
        name: ロガー名
        level: ログレベル（デフォルト: INFO）

    Returns:
        logging.Logger: 設定済みロガー
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # すでにハンドラが設定されている場合はスキップ
    if logger.handlers:
        return logger

    # Rich handlerを設定
    handler = RichHandler(
        rich_tracebacks=True,
        show_time=True,
        show_path=False,
    )
    handler.setFormatter(logging.Formatter(fmt="%(message)s", datefmt="[%X]"))

    logger.addHandler(handler)

    # 親ロガーへの伝播を防止
    logger.propagate = False

    return logger


# ============================================================================
# Custom Exceptions
# ============================================================================


class SequenceConverterError(Exception):
    """Base exception for sequence converter errors."""

    pass


class UnsupportedFormatError(SequenceConverterError):
    """Raised when input file format is not supported."""

    pass


class ImageProcessingError(SequenceConverterError):
    """Raised when image processing fails."""

    pass
