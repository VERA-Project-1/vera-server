"""Terminal logging: one coloured format for app + uvicorn, with the current request id on every line."""
import logging
import logging.config
from contextvars import ContextVar

# Set per request by the middleware in main.py; anyio copies it into threadpool workers.
request_id_var: ContextVar[str] = ContextVar("request_id", default="-")


class RequestIdFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        record.request_id = request_id_var.get()
        if record.name == "uvicorn.error":  # uvicorn's server lifecycle logger, not errors
            record.name = "uvicorn"
        return True


# Chatty third-party loggers: HF hub checks, httpx request lines, file locks.
NOISY_LOGGERS = ("httpx", "httpcore", "huggingface_hub", "urllib3", "filelock", "numba", "transformers")


def setup_logging(level: str = "INFO") -> None:
    logging.config.dictConfig(
        {
            "version": 1,
            "disable_existing_loggers": False,
            "filters": {"request_id": {"()": RequestIdFilter}},
            "formatters": {
                "default": {
                    "()": "uvicorn.logging.DefaultFormatter",
                    "fmt": "%(asctime)s %(levelprefix)s %(name)-12s [%(request_id)s] %(message)s",
                    "datefmt": "%H:%M:%S",
                    "use_colors": None,  # auto: colours on a TTY, plain when piped to a file
                },
            },
            "handlers": {
                "console": {
                    "class": "logging.StreamHandler",
                    "formatter": "default",
                    "filters": ["request_id"],
                    "stream": "ext://sys.stderr",
                },
            },
            "loggers": {
                "vera": {"handlers": ["console"], "level": level, "propagate": False},
                "uvicorn": {"handlers": ["console"], "level": "INFO", "propagate": False},
                "uvicorn.error": {"level": "INFO"},
                # Replaced by the richer per-request line from the middleware.
                "uvicorn.access": {"handlers": [], "level": "WARNING", "propagate": False},
                **{name: {"level": "WARNING"} for name in NOISY_LOGGERS},
                # Only emits the "set HF_TOKEN" nag at WARNING.
                "huggingface_hub": {"level": "ERROR"},
            },
            "root": {"handlers": ["console"], "level": "WARNING"},
        }
    )
    # Route warnings.warn() (e.g. torch deprecations) through the same formatter.
    logging.captureWarnings(True)
