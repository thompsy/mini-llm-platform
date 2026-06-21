"""Centralised logging setup.

Application modules never configure logging themselves; they only obtain a
logger via ``logging.getLogger(__name__)``. The entry points (the ingest CLI and
the API lifespan) call :func:`setup_logging` once at startup to install handlers
and set the level.
"""

import logging

LOG_FORMAT = "%(asctime)s %(levelname)s %(name)s: %(message)s"

# Third-party loggers that are too chatty at DEBUG; keep them at WARNING so our
# own DEBUG output stays readable.
_NOISY_LOGGERS = ("httpx", "httpcore", "chromadb")


def setup_logging(level: str = "INFO") -> None:
    """Configure root logging once, with a consistent format and the given level.

    Safe to call from any entry point. ``force=True`` ensures the configuration
    is applied even if something (e.g. a library) already touched the root logger.
    Noisy third-party loggers are pinned to WARNING so app DEBUG logs are legible.
    """
    logging.basicConfig(
        level=level.upper(),
        format=LOG_FORMAT,
        force=True,
    )
    for name in _NOISY_LOGGERS:
        logging.getLogger(name).setLevel(logging.WARNING)
