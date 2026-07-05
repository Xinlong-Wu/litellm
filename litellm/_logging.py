import ast
import logging
import os
import sys
import time
from datetime import datetime
from logging import Formatter
from logging.handlers import TimedRotatingFileHandler
from typing import Any, Dict, Optional

from litellm.litellm_core_utils.secret_redaction import redact_string
from litellm.litellm_core_utils.safe_json_dumps import safe_dumps
from litellm.litellm_core_utils.safe_json_loads import safe_json_loads

# OS-dispatched import for the advisory file lock used to serialize daily log
# rotation across processes. Each platform imports only the stdlib module it is
# guaranteed to have, so importing this module is safe everywhere.
if sys.platform == "win32":  # pragma: no cover - Windows-only
    import msvcrt
else:
    import fcntl


def _try_lock_nb(fd: int) -> bool:
    """Non-blocking exclusive advisory lock. True if acquired, False if contended."""
    try:
        if sys.platform == "win32":
            os.lseek(fd, 0, os.SEEK_SET)
            msvcrt.locking(fd, msvcrt.LK_NBLCK, 1)
        else:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        return True
    except OSError:  # BlockingIOError (contended) is an OSError subclass
        return False


def _unlock(fd: int) -> None:
    try:
        if sys.platform == "win32":
            os.lseek(fd, 0, os.SEEK_SET)
            msvcrt.locking(fd, msvcrt.LK_UNLCK, 1)
        else:
            fcntl.flock(fd, fcntl.LOCK_UN)
    except OSError:
        pass


set_verbose = False

if set_verbose is True:
    logging.warning(
        "`litellm.set_verbose` is deprecated. Please set `os.environ['LITELLM_LOG'] = 'DEBUG'` for debug logs."
    )

_ENABLE_SECRET_REDACTION = (
    os.getenv("LITELLM_DISABLE_REDACT_SECRETS", "").lower() != "true"
)


def _redact_string(value: str) -> str:
    if not _ENABLE_SECRET_REDACTION:
        return value
    return redact_string(value)


def redact_secrets(value: str) -> str:
    """Public API: redact known secret/credential patterns from an arbitrary string.

    Use this for code paths that bypass the logging system — e.g. Slack/Teams
    alerting, HTTP error response bodies, or any other string that may contain
    secrets and will be sent to an external sink.

    Not to be confused with redact_message_input_output_from_logging() in
    litellm_core_utils/redact_messages.py, which redacts LLM prompt/response
    content for privacy — this function redacts credential patterns (API keys,
    PEM blocks, tokens, etc.) by shape.
    """
    if not _ENABLE_SECRET_REDACTION:
        return value
    return _redact_string(value)


class SecretRedactionFilter(logging.Filter):
    """Scrubs known secret/credential patterns from log records."""

    _formatter = logging.Formatter()

    def filter(self, record: logging.LogRecord) -> bool:
        if not _ENABLE_SECRET_REDACTION:
            return True

        try:
            record.msg = _redact_string(record.getMessage())
            record.args = None
        except Exception:
            if isinstance(record.msg, str):
                record.msg = _redact_string(record.msg)

        # Redact exception tracebacks
        if record.exc_info and record.exc_info[1] is not None:
            try:
                record.exc_text = _redact_string(
                    self._formatter.formatException(record.exc_info)
                )
            except Exception:
                pass

        # Redact extra fields passed via logger.debug("msg", extra={...})
        for key, value in list(record.__dict__.items()):
            if key not in _STANDARD_RECORD_ATTRS and isinstance(value, str):
                setattr(record, key, _redact_string(value))

        return True


_secret_filter = SecretRedactionFilter()


json_logs = bool(os.getenv("JSON_LOGS", False))
# Create a handler for the logger (you may need to adapt this based on your needs)
log_level = os.getenv("LITELLM_LOG", "DEBUG")
numeric_level: str = getattr(logging, log_level.upper())
handler = logging.StreamHandler()
handler.setLevel(numeric_level)
handler.addFilter(_secret_filter)


def _try_parse_json_message(message: str) -> Optional[Dict[str, Any]]:
    """
    Try to parse a log message as JSON. Returns parsed dict if valid, else None.
    Handles messages that are entirely valid JSON (e.g. json.dumps output).
    Uses shared safe_json_loads for consistent error handling.
    """
    if not message or not isinstance(message, str):
        return None
    msg_stripped = message.strip()
    if not (msg_stripped.startswith("{") or msg_stripped.startswith("[")):
        return None
    parsed = safe_json_loads(message, default=None)
    if parsed is None or not isinstance(parsed, dict):
        return None
    return parsed


def _try_parse_embedded_python_dict(message: str) -> Optional[Dict[str, Any]]:
    """
    Try to find and parse a Python dict repr (e.g. str(d) or repr(d)) embedded in
    the message. Handles patterns like:
    "get_available_deployment for model: X, Selected deployment: {'model_name': '...', ...} for model: X"
    Uses ast.literal_eval for safe parsing. Returns the parsed dict or None.
    """
    if not message or not isinstance(message, str) or "{" not in message:
        return None
    i = 0
    while i < len(message):
        start = message.find("{", i)
        if start == -1:
            break
        depth = 0
        for j in range(start, len(message)):
            c = message[j]
            if c == "{":
                depth += 1
            elif c == "}":
                depth -= 1
                if depth == 0:
                    substr = message[start : j + 1]
                    try:
                        result = ast.literal_eval(substr)
                        if isinstance(result, dict) and len(result) > 0:
                            return result
                    except (ValueError, SyntaxError, TypeError):
                        pass
                    break
        i = start + 1
    return None


# Standard LogRecord attribute names - used to identify 'extra' fields.
# Derived at runtime so we automatically include version-specific attrs (e.g. taskName).
def _get_standard_record_attrs() -> frozenset:
    """Standard LogRecord attribute names - excludes extra keys from logger.debug(..., extra={...})."""
    return frozenset(logging.LogRecord("", 0, "", 0, "", (), None).__dict__.keys())


_STANDARD_RECORD_ATTRS = _get_standard_record_attrs()


class JsonFormatter(Formatter):
    def __init__(self):
        super(JsonFormatter, self).__init__()

    def formatTime(self, record, datefmt=None):
        # Use datetime to format the timestamp in ISO 8601 format
        dt = datetime.fromtimestamp(record.created)
        return dt.isoformat()

    def format(self, record):
        message_str = record.getMessage()
        json_record: Dict[str, Any] = {
            "message": message_str,
            "level": record.levelname,
            "timestamp": self.formatTime(record),
        }

        # Parse embedded JSON or Python dict repr in message so sub-fields become first-class properties
        parsed = _try_parse_json_message(message_str)
        if parsed is None:
            parsed = _try_parse_embedded_python_dict(message_str)
        if parsed is not None:
            for key, value in parsed.items():
                if key not in json_record:
                    json_record[key] = value

        # Include extra attributes passed via logger.debug("msg", extra={...})
        for key, value in record.__dict__.items():
            if key not in _STANDARD_RECORD_ATTRS and key not in json_record:
                json_record[key] = value

        # Set component/logger only if not already supplied via extra={...}
        if "component" not in json_record:
            json_record["component"] = record.name
        if "logger" not in json_record:
            json_record["logger"] = f"{record.filename}:{record.lineno}"

        if record.exc_info:
            json_record["stacktrace"] = record.exc_text or self.formatException(
                record.exc_info
            )

        return safe_dumps(json_record)


# Function to set up exception handlers for JSON logging
def _setup_json_exception_handlers(formatter):
    # Create a handler with JSON formatting for exceptions
    error_handler = logging.StreamHandler()
    error_handler.setFormatter(formatter)
    error_handler.addFilter(_secret_filter)

    # Setup excepthook for uncaught exceptions
    def json_excepthook(exc_type, exc_value, exc_traceback):
        record = logging.LogRecord(
            name="LiteLLM",
            level=logging.ERROR,
            pathname="",
            lineno=0,
            msg=str(exc_value),
            args=(),
            exc_info=(exc_type, exc_value, exc_traceback),
        )
        error_handler.handle(record)

    sys.excepthook = json_excepthook

    # Configure asyncio exception handler if possible
    try:
        import asyncio

        def async_json_exception_handler(loop, context):
            exception = context.get("exception")
            if exception:
                exc_type = type(exception)
                record = logging.LogRecord(
                    name="LiteLLM",
                    level=logging.ERROR,
                    pathname="",
                    lineno=0,
                    msg=str(exception),
                    args=(),
                    exc_info=(exc_type, exception, exception.__traceback__),
                )
                error_handler.handle(record)
            else:
                loop.default_exception_handler(context)

        asyncio.get_event_loop().set_exception_handler(async_json_exception_handler)
    except Exception:
        pass


# Create a formatter and set it for the handler
if json_logs:
    handler.setFormatter(JsonFormatter())
    _setup_json_exception_handlers(JsonFormatter())
else:
    formatter = logging.Formatter(
        "\033[92m%(asctime)s - %(name)s:%(levelname)s\033[0m: %(filename)s:%(lineno)s - %(message)s",
        datefmt="%H:%M:%S",
    )

    handler.setFormatter(formatter)

verbose_proxy_logger = logging.getLogger("LiteLLM Proxy")
verbose_router_logger = logging.getLogger("LiteLLM Router")
verbose_logger = logging.getLogger("LiteLLM")

# Add the handler to the loggers
verbose_router_logger.addHandler(handler)
verbose_proxy_logger.addHandler(handler)
verbose_logger.addHandler(handler)


def _suppress_loggers():
    """Suppress noisy loggers at INFO level"""
    # Suppress httpx request logging at INFO level
    httpx_logger = logging.getLogger("httpx")
    httpx_logger.setLevel(logging.WARNING)

    # Suppress APScheduler logging at INFO level
    apscheduler_executors_logger = logging.getLogger("apscheduler.executors.default")
    apscheduler_executors_logger.setLevel(logging.WARNING)
    apscheduler_scheduler_logger = logging.getLogger("apscheduler.scheduler")
    apscheduler_scheduler_logger.setLevel(logging.WARNING)


# Call the suppression function
_suppress_loggers()

ALL_LOGGERS = [
    logging.getLogger(),
    verbose_logger,
    verbose_router_logger,
    verbose_proxy_logger,
]


def _get_loggers_to_initialize():
    """
    Get all loggers that should be initialized with the JSON handler.

    Includes third-party integration loggers (like langfuse) if they are
    configured as callbacks.
    """
    import litellm

    loggers = list(ALL_LOGGERS)

    # Add langfuse logger if langfuse is being used as a callback
    langfuse_callbacks = {"langfuse", "langfuse_otel"}
    all_callbacks = set(litellm.success_callback + litellm.failure_callback)
    if langfuse_callbacks & all_callbacks:
        loggers.append(logging.getLogger("langfuse"))

    return loggers


# ----------------------------------------------------------------------------
# Optional file logging (write logs to a directory, rotated daily)
# ----------------------------------------------------------------------------

DEFAULT_LOG_FILENAME = "litellm.log"
UVICORN_LOG_FILENAME = "uvicorn.log"

# Tracks the file handler currently attached to the litellm loggers so re-init
# (e.g. toggling JSON) can rebuild it without leaking duplicate handlers.
_app_file_handler: Optional[logging.Handler] = None


def _get_log_retention_days() -> int:
    raw = os.getenv("LITELLM_LOG_RETENTION_DAYS", "14")
    try:
        return max(0, int(raw))
    except ValueError:
        return 14


def _resolve_app_log_file() -> Optional[str]:
    """Resolve the application log file path from env, or None if unconfigured.

    LITELLM_LOG_FILE (explicit path) wins; otherwise LITELLM_LOG_DIR/litellm.log.
    """
    log_file = os.getenv("LITELLM_LOG_FILE")
    if log_file:
        return log_file
    log_dir = os.getenv("LITELLM_LOG_DIR")
    if log_dir:
        return os.path.join(log_dir, DEFAULT_LOG_FILENAME)
    return None


def resolve_uvicorn_log_file() -> Optional[str]:
    """Resolve the uvicorn log file path (separate from the application log)."""
    log_dir = os.getenv("LITELLM_LOG_DIR")
    if log_dir:
        return os.path.join(log_dir, UVICORN_LOG_FILENAME)
    return None


def _get_file_text_formatter() -> logging.Formatter:
    """Plain (no ANSI color) formatter for log files."""
    return logging.Formatter(
        "%(asctime)s - %(name)s:%(levelname)s: %(filename)s:%(lineno)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def _lockfile_for(path: str) -> str:
    """Sidecar lock file path used to serialize rotation across processes."""
    parent = os.path.dirname(path) or "."
    return os.path.join(parent, "." + os.path.basename(path) + ".rotate.lock")


class _SharedRotatingFileHandler(TimedRotatingFileHandler):
    """Daily-rotating file handler safe for multiple processes sharing one file.

    All processes append to the same file. Rotation is emit-driven (no background
    threads, so it is fork/spawn-safe under multi-worker servers): the first
    process to log after local midnight tries a short-lived advisory ``flock`` on
    ``lockfile`` and, if it wins, renames the base file to a dated one. Peers that
    lose the race - or that already see the rotated file - detect the inode change
    on their next ``emit`` and reopen the new file (``WatchedFileHandler``
    behavior). Fork-safe because the lock fd is opened fresh per attempt and never
    held between rotations.
    """

    def __init__(
        self,
        filename: str,
        lockfile: str,
        backupCount: int = 14,
        encoding: str = "utf-8",
        delay: bool = True,
        use_json: bool = False,
    ) -> None:
        parent = os.path.dirname(filename)
        if parent:
            os.makedirs(parent, exist_ok=True)
        super().__init__(
            filename,
            when="midnight",
            backupCount=backupCount,
            encoding=encoding,
            delay=delay,
        )
        self._lockfile = lockfile
        self.addFilter(_secret_filter)
        self.setLevel(numeric_level)
        self.setFormatter(JsonFormatter() if use_json else _get_file_text_formatter())
        self._dev: Optional[int] = None
        self._ino: Optional[int] = None
        self._update_dev_ino()

    def _stat_base(self):
        try:
            st = os.stat(self.baseFilename)
            return st.st_dev, st.st_ino
        except FileNotFoundError:
            return None, None

    def _update_dev_ino(self) -> None:
        self._dev, self._ino = self._stat_base()

    def _reopen_if_changed(self) -> None:
        """Reopen the base file if a peer rotated it out from under us."""
        dev, ino = self._stat_base()
        if (
            dev is not None
            and self._dev is not None
            and (dev != self._dev or ino != self._ino)
            and self.stream is not None
        ):
            self.stream.flush()
            self.stream.close()
            self.stream = self._open()
            self._update_dev_ino()

    def emit(self, record: logging.LogRecord) -> None:
        try:
            self._reopen_if_changed()
        except Exception:
            pass
        super().emit(
            record
        )  # BaseRotatingHandler.emit -> shouldRollover -> doRollover, then write
        if self._dev is None and self.stream is not None:
            # First open under delay=True - record file identity now.
            self._update_dev_ino()

    def doRollover(self) -> None:
        """Rotate under a cross-process lock; only the lock winner renames.

        Invoked by the base ``emit`` when the period has elapsed (or directly by
        tests). Losers just adopt the peer's freshly rotated file and reschedule.
        """
        fd = os.open(self._lockfile, os.O_CREAT | os.O_RDWR, 0o644)
        try:
            if _try_lock_nb(fd):
                try:
                    _, ino = self._stat_base()
                    # Only rename if nobody has rotated this file already.
                    if ino is not None and (self._ino is None or ino == self._ino):
                        super().doRollover()
                        self._update_dev_ino()
                        return
                finally:
                    _unlock(fd)
        finally:
            os.close(fd)
        # Lost the race or already rotated: adopt the new file, defer next attempt.
        self._reopen_if_changed()
        self.rolloverAt = self.computeRollover(int(time.time()))

    # Test/utility hook: force a gated rotation attempt now.
    def _maybe_rollover(self) -> None:
        self.doRollover()


def _build_file_handler(path: str, use_json: bool) -> _SharedRotatingFileHandler:
    return _SharedRotatingFileHandler(
        filename=path,
        lockfile=_lockfile_for(path),
        backupCount=_get_log_retention_days(),
        use_json=use_json,
    )


def add_file_logging(
    path: Optional[str] = None, use_json: Optional[bool] = None
) -> Optional[str]:
    """Attach a daily-rotating file handler to the litellm loggers.

    Resolves ``path`` from the argument, then from LITELLM_LOG_FILE / LITELLM_LOG_DIR.
    Returns the resolved path (or None if file logging is not configured).
    Idempotent: an existing litellm-managed file handler is replaced, not duplicated.
    """
    global _app_file_handler

    resolved = path or _resolve_app_log_file()
    if not resolved:
        return None
    if use_json is None:
        use_json = json_logs

    # Attach to the same core loggers as the stream handler. Avoid
    # _get_loggers_to_initialize() here: it reads litellm.success_callback,
    # which is not defined yet during the import-time call.
    loggers = [verbose_logger, verbose_router_logger, verbose_proxy_logger]
    if _app_file_handler is not None:
        for lg in loggers:
            if _app_file_handler in lg.handlers:
                lg.removeHandler(_app_file_handler)

    _app_file_handler = _build_file_handler(resolved, use_json)
    for lg in loggers:
        lg.addHandler(_app_file_handler)
    return resolved


def _reattach_app_file_handler(use_json: bool) -> None:
    """Re-attach the file handler after loggers were cleared/re-initialized."""
    if _resolve_app_log_file() is not None:
        add_file_logging(use_json=use_json)


# Enable file logging at import time when LITELLM_LOG_DIR / LITELLM_LOG_FILE is set,
# so simply exporting the env var and starting the proxy is enough.
add_file_logging()


def _initialize_loggers_with_handler(handler: logging.Handler):
    """
    Initialize all loggers with a handler

    - Adds a handler to each logger
    - Prevents bubbling to parent/root (critical to prevent duplicate JSON logs)
    """
    global _app_file_handler
    handler.addFilter(_secret_filter)
    _app_file_handler = None  # handlers are cleared below; drop the stale reference
    for lg in _get_loggers_to_initialize():
        lg.handlers.clear()  # remove any existing handlers
        lg.addHandler(handler)  # add JSON formatter handler
        lg.propagate = False  # prevent bubbling to parent/root


def _get_uvicorn_log_config(use_json: bool):
    """
    Generate a uvicorn log_config dict.

    - In JSON mode all uvicorn loggers use the JSON formatter.
    - In text mode stdout keeps uvicorn's standard (optionally colored) formatters.
    - When LITELLM_LOG_DIR is set, a daily-rotating file handler writing to
      ``<dir>/uvicorn.log`` (separate from the application log) is added so
      uvicorn access/error logs also land on disk.
    """
    uvicorn_log_level = log_level.upper()
    json_formatter_class = "litellm._logging.JsonFormatter"
    default_fmt = "%(asctime)s %(levelprefix)s %(message)s"
    access_fmt = (
        "%(asctime)s %(levelprefix)s %(client_addr)s - "
        '"%(request_line)s" %(status_code)s'
    )

    formatters: Dict[str, Any] = {
        "json": {"()": json_formatter_class},
        "default": {
            "()": "uvicorn.logging.DefaultFormatter",
            "fmt": default_fmt,
            "use_colors": True,
        },
        "access": {
            "()": "uvicorn.logging.AccessFormatter",
            "fmt": access_fmt,
            "use_colors": True,
        },
    }

    stdout_default_formatter = "json" if use_json else "default"
    stdout_access_formatter = "json" if use_json else "access"
    handlers: Dict[str, Any] = {
        "default": {
            "formatter": stdout_default_formatter,
            "class": "logging.StreamHandler",
            "stream": "ext://sys.stdout",
        },
        "access": {
            "formatter": stdout_access_formatter,
            "class": "logging.StreamHandler",
            "stream": "ext://sys.stdout",
        },
    }

    default_handlers = ["default"]
    access_handlers = ["access"]

    uvicorn_log_file = resolve_uvicorn_log_file()
    if uvicorn_log_file:
        # One shared, multi-process-safe file handler for all uvicorn loggers.
        # It self-formats (JSON or plain text) and rotates daily via flock, so no
        # dictConfig formatter is attached here.
        handlers["file"] = {
            "()": "litellm._logging._SharedRotatingFileHandler",
            "filename": uvicorn_log_file,
            "lockfile": _lockfile_for(uvicorn_log_file),
            "backupCount": _get_log_retention_days(),
            "use_json": use_json,
        }
        default_handlers = ["default", "file"]
        access_handlers = ["access", "file"]

    return {
        "version": 1,
        "disable_existing_loggers": False,
        "formatters": formatters,
        "handlers": handlers,
        "loggers": {
            "uvicorn": {
                "handlers": default_handlers,
                "level": uvicorn_log_level,
                "propagate": False,
            },
            "uvicorn.error": {
                "handlers": default_handlers,
                "level": uvicorn_log_level,
                "propagate": False,
            },
            "uvicorn.access": {
                "handlers": access_handlers,
                "level": uvicorn_log_level,
                "propagate": False,
            },
        },
    }


def _get_uvicorn_json_log_config():
    """Backwards-compatible JSON uvicorn log config (delegates to _get_uvicorn_log_config)."""
    return _get_uvicorn_log_config(use_json=True)


def _turn_on_json():
    """
    Turn on JSON logging

    - Adds a JSON formatter to all loggers
    """
    handler = logging.StreamHandler()
    handler.setFormatter(JsonFormatter())
    _initialize_loggers_with_handler(handler)
    # Keep file logging (now JSON-formatted) after handlers were cleared above
    _reattach_app_file_handler(use_json=True)
    # Set up exception handlers
    _setup_json_exception_handlers(JsonFormatter())


def _turn_on_debug():
    verbose_logger.setLevel(level=logging.DEBUG)  # set package log to debug
    verbose_router_logger.setLevel(level=logging.DEBUG)  # set router logs to debug
    verbose_proxy_logger.setLevel(level=logging.DEBUG)  # set proxy logs to debug


def _disable_debugging():
    """Disable the package, router, and proxy verbose loggers."""
    verbose_logger.disabled = True
    verbose_router_logger.disabled = True
    verbose_proxy_logger.disabled = True


def _enable_debugging():
    verbose_logger.disabled = False
    verbose_router_logger.disabled = False
    verbose_proxy_logger.disabled = False


def print_verbose(print_statement):
    try:
        if set_verbose:
            print(redact_secrets(str(print_statement)))  # noqa: T201
    except Exception:
        pass


def _is_debugging_on() -> bool:
    """
    Returns True if debugging is on
    """
    return verbose_logger.isEnabledFor(logging.DEBUG) or set_verbose is True
