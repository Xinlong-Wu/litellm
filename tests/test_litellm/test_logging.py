import ast
import asyncio
import json
import os
import sys
from pathlib import Path
from typing import List

import pytest

sys.path.insert(0, os.path.abspath("../../.."))  # Adds the parent directory to the system-path
import logging
import sys

import litellm
from litellm._logging import (
    ALL_LOGGERS,
    JsonFormatter,
    _initialize_loggers_with_handler,
    _turn_on_json,
    verbose_logger,
    verbose_proxy_logger,
    verbose_router_logger,
)
from litellm.integrations.custom_logger import CustomLogger
from litellm.types.utils import StandardLoggingPayload


class CacheHitCustomLogger(CustomLogger):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.logged_standard_logging_payloads: List[StandardLoggingPayload] = []

    async def async_log_success_event(self, kwargs, response_obj, start_time, end_time):
        standard_logging_payload = kwargs.get("standard_logging_object", None)
        if standard_logging_payload:
            self.logged_standard_logging_payloads.append(standard_logging_payload)


def test_json_mode_emits_one_record_per_logger(capfd):
    # Turn on JSON logging
    _turn_on_json()
    # Make sure our loggers will emit INFO-level records
    for lg in (verbose_logger, verbose_router_logger, verbose_proxy_logger):
        lg.setLevel(logging.INFO)

    # Log one message from each logger at different levels
    verbose_logger.info("first info")
    verbose_router_logger.info("second info from router")
    verbose_proxy_logger.info("third info from proxy")

    # Capture stdout
    out, err = capfd.readouterr()
    print("out", out)
    print("err", err)
    lines = [l for l in err.splitlines() if l.strip()]

    # Expect exactly three JSON lines
    assert len(lines) == 3, f"got {len(lines)} lines, want 3: {lines!r}"

    # Each line must be valid JSON with the required fields
    for line in lines:
        obj = json.loads(line)
        assert "message" in obj, "`message` key missing"
        assert "level" in obj, "`level` key missing"
        assert "timestamp" in obj, "`timestamp` key missing"


def test_json_formatter_parses_embedded_json_message():
    """
    Test that JsonFormatter parses embedded JSON in the message field and promotes
    sub-fields to first-class JSON properties for downstream querying.
    """
    formatter = JsonFormatter()
    record = logging.LogRecord(
        name="LiteLLM",
        level=logging.DEBUG,
        pathname="",
        lineno=0,
        msg='{"event": "giveup", "exception": "Connection failed", "model_name": "gpt-4"}',
        args=(),
        exc_info=None,
    )
    output = formatter.format(record)
    obj = json.loads(output)
    # Standard fields preserved
    assert "message" in obj
    assert obj["level"] == "DEBUG"
    assert "timestamp" in obj
    # Embedded JSON fields promoted to top-level for querying
    assert obj["event"] == "giveup"
    assert obj["exception"] == "Connection failed"
    assert obj["model_name"] == "gpt-4"


def test_json_formatter_includes_extra_attributes():
    """
    Test that JsonFormatter includes extra attributes from logger.debug("msg", extra={...}).
    """
    formatter = JsonFormatter()
    record = logging.LogRecord(
        name="LiteLLM",
        level=logging.DEBUG,
        pathname="",
        lineno=0,
        msg="POST Request Sent from LiteLLM",
        args=(),
        exc_info=None,
    )
    record.api_base = "https://api.openai.com"
    record.authorization = "Bearer sk-***"
    output = formatter.format(record)
    obj = json.loads(output)
    assert obj["message"] == "POST Request Sent from LiteLLM"
    assert obj["api_base"] == "https://api.openai.com"
    assert obj["authorization"] == "Bearer sk-***"


def test_json_formatter_plain_message_unchanged():
    """
    Test that non-JSON messages are passed through as-is in the message field.
    """
    formatter = JsonFormatter()
    record = logging.LogRecord(
        name="LiteLLM",
        level=logging.INFO,
        pathname="",
        lineno=0,
        msg="Cache hit!",
        args=(),
        exc_info=None,
    )
    output = formatter.format(record)
    obj = json.loads(output)
    assert obj["message"] == "Cache hit!"
    assert "event" not in obj
    assert "exception" not in obj


def test_json_formatter_parses_embedded_python_dict_repr():
    """
    Test that JsonFormatter parses Python dict repr (str/deployment) embedded in
    plain text, e.g. from get_available_deployment logs.
    Reproduces Roni's reported case.
    """
    formatter = JsonFormatter()
    msg = (
        "get_available_deployment for model: text-embedding-3-large, "
        "Selected deployment: {'model_name': 'text-embedding-3-large', "
        "'litellm_params': {'api_key': 'sk**********', 'tpm': 1000000, 'rpm': 2000, "
        "'use_in_pass_through': False, 'use_litellm_proxy': False, "
        "'merge_reasoning_content_in_choices': False, 'model': 'text-embedding-3-large'}, "
        "'model_info': {'id': 'a624b057aec64ada48311', 'db_model': False}} "
        "for model: text-embedding-3-large"
    )
    record = logging.LogRecord(
        name="LiteLLM Router",
        level=logging.INFO,
        pathname="",
        lineno=0,
        msg=msg,
        args=(),
        exc_info=None,
    )
    output = formatter.format(record)
    obj = json.loads(output)
    assert "message" in obj
    assert obj["level"] == "INFO"
    # Python dict parsed and promoted to first-class properties
    assert obj["model_name"] == "text-embedding-3-large"
    assert "litellm_params" in obj
    assert obj["litellm_params"]["api_key"] == "sk**********"
    assert obj["litellm_params"]["tpm"] == 1000000
    assert obj["litellm_params"]["use_in_pass_through"] is False
    assert "model_info" in obj
    assert obj["model_info"]["id"] == "a624b057aec64ada48311"
    assert obj["model_info"]["db_model"] is False


def test_json_formatter_includes_component_field():
    """
    Test that JsonFormatter always emits a 'component' field equal to the logger name.
    This allows filtering by component (e.g. "LiteLLM Proxy") in Datadog / third-party log services.
    """
    formatter = JsonFormatter()
    for logger_name in ("LiteLLM Proxy", "LiteLLM Router", "LiteLLM"):
        record = logging.LogRecord(
            name=logger_name,
            level=logging.ERROR,
            pathname="proxy_server.py",
            lineno=42,
            msg="something went wrong",
            args=(),
            exc_info=None,
        )
        output = formatter.format(record)
        obj = json.loads(output)
        assert obj["component"] == logger_name, f"Expected component={logger_name!r}, got {obj.get('component')!r}"


def test_json_formatter_includes_logger_field():
    """
    Test that JsonFormatter always emits a 'logger' field with filename:lineno.
    This allows pinpointing the exact source of a log line in third-party services.
    """
    formatter = JsonFormatter()
    record = logging.LogRecord(
        name="LiteLLM Proxy",
        level=logging.INFO,
        pathname="/app/litellm/proxy/proxy_server.py",
        lineno=123,
        msg="request received",
        args=(),
        exc_info=None,
    )
    output = formatter.format(record)
    obj = json.loads(output)
    assert obj["logger"] == "proxy_server.py:123", f"Expected logger='proxy_server.py:123', got {obj['logger']!r}"


def test_json_formatter_extra_component_not_overwritten():
    """
    User-supplied extra={"component": "..."} must not be silently dropped.
    """
    formatter = JsonFormatter()
    record = logging.LogRecord(
        name="LiteLLM Proxy",
        level=logging.INFO,
        pathname="proxy_server.py",
        lineno=1,
        msg="event",
        args=(),
        exc_info=None,
    )
    record.component = "auth-service"
    obj = json.loads(formatter.format(record))
    assert obj["component"] == "auth-service", f"User-supplied component was overwritten, got {obj['component']!r}"


def test_initialize_loggers_with_handler_sets_propagate_false():
    """
    Test that the initialize_loggers_with_handler function sets propagate to False for all loggers
    """
    # Initialize loggers with the test handler
    _initialize_loggers_with_handler(logging.StreamHandler())

    # Check that propagate is set to False for all loggers
    for logger in ALL_LOGGERS:
        assert logger.propagate is False, (
            f"Logger {logger.name} has propagate set to {logger.propagate}, expected False"
        )


@pytest.mark.asyncio
async def test_cache_hit_includes_custom_llm_provider():
    """
    Test that when there's a cache hit, the standard logging payload includes the custom_llm_provider
    """
    # Set up caching and custom logger
    litellm.cache = litellm.Cache()
    test_custom_logger = CacheHitCustomLogger()
    original_callbacks = litellm.callbacks.copy() if litellm.callbacks else []
    litellm.callbacks = [test_custom_logger]

    try:
        # First call - should be a cache miss
        response1 = await litellm.acompletion(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": "test cache hit message"}],
            mock_response="test response",
            caching=True,
        )

        # Wait for logging to complete
        await asyncio.sleep(0.5)

        # Second identical call - should be a cache hit
        response2 = await litellm.acompletion(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": "test cache hit message"}],
            mock_response="test response",
            caching=True,
        )

        # Wait for logging to complete
        await asyncio.sleep(0.5)

        # Verify we have logged events
        assert len(test_custom_logger.logged_standard_logging_payloads) >= 2, (
            f"Expected at least 2 logged events, got {len(test_custom_logger.logged_standard_logging_payloads)}"
        )

        # Find the cache hit event (should be the second call)
        cache_hit_payload = None
        for payload in test_custom_logger.logged_standard_logging_payloads:
            if payload.get("cache_hit") is True:
                cache_hit_payload = payload
                break

        # Verify cache hit event was found
        assert cache_hit_payload is not None, "No cache hit event found in logged payloads"

        # Verify custom_llm_provider is included in the cache hit payload
        assert "custom_llm_provider" in cache_hit_payload, (
            "custom_llm_provider missing from cache hit standard logging payload"
        )

        # Verify custom_llm_provider has a valid value (should be "openai" for gpt-3.5-turbo)
        custom_llm_provider = cache_hit_payload["custom_llm_provider"]
        assert custom_llm_provider is not None and custom_llm_provider != "", (
            f"custom_llm_provider should not be None or empty, got: {custom_llm_provider}"
        )

        print(
            f"Cache hit standard logging payload with custom_llm_provider: {custom_llm_provider}",
            json.dumps(cache_hit_payload, indent=2),
        )

    finally:
        # Clean up
        litellm.callbacks = original_callbacks
        litellm.cache = None


LITELLM_LOGGER_NAMES = frozenset(
    {"verbose_logger", "verbose_proxy_logger", "verbose_router_logger", "logger", "logging"}
)
LOG_LEVEL_METHODS = frozenset({"debug", "info", "warning", "error", "exception", "critical"})
LITELLM_PACKAGE_ROOT = Path(__file__).resolve().parents[2] / "litellm"


def _receiver_name(node: ast.expr) -> str:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return node.attr
    return ""


def _is_logging_call(node: ast.AST) -> bool:
    return (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr in LOG_LEVEL_METHODS
        and _receiver_name(node.func.value) in LITELLM_LOGGER_NAMES
    )


def _has_format_spec(message: ast.JoinedStr) -> bool:
    return any(isinstance(value, ast.FormattedValue) and value.format_spec is not None for value in message.values)


def _eager_logging_calls(source: str, path: Path) -> tuple[str, ...]:
    return tuple(
        f"{path}:{node.lineno}"
        for node in ast.walk(ast.parse(source))
        if _is_logging_call(node)
        and node.args
        and isinstance(node.args[0], ast.JoinedStr)
        and not _has_format_spec(node.args[0])
    )


def test_logging_calls_do_not_build_their_message_eagerly():
    """A discarded log record must not have cost anything to build.

    `log.debug(f"payload: {body}")` interpolates before the call runs, so the message is
    built and thrown away on every request the level filters out; `log.debug("payload: %s", body)`
    defers that to `record.getMessage()`, which only runs once the record passes the level check.

    f-strings carrying a format spec are exempt: `%`-style has no faithful equivalent for
    specs like `{ratio:.1%}`, and those sites interpolate scalars rather than payloads.
    """
    offenders = tuple(
        offender
        for path in sorted(LITELLM_PACKAGE_ROOT.rglob("*.py"))
        for offender in _eager_logging_calls(
            path.read_text(encoding="utf-8"), path.relative_to(LITELLM_PACKAGE_ROOT.parent)
        )
    )

    assert offenders == (), (
        "these logging calls build their message eagerly; pass the values as %-style arguments instead:\n"
        + "\n".join(offenders)
    )
# ---------------------------------------------------------------------------
# File logging: write logs to a directory (daily rotation)
# ---------------------------------------------------------------------------
import glob
import logging.handlers

from litellm._logging import (
    add_file_logging,
    resolve_uvicorn_log_file,
    _get_app_file_handler,
    _get_uvicorn_log_config,
    _lockfile_for,
    _reattach_app_file_handler,
    _SharedRotatingFileHandler,
)

_FILE_LOG_LOGGERS = [verbose_logger, verbose_router_logger, verbose_proxy_logger]


@pytest.fixture
def clean_file_logging(monkeypatch):
    """Detach any litellm-managed file handler and clear log env after the test."""
    for var in ("LITELLM_LOG_DIR", "LITELLM_LOG_FILE", "LITELLM_LOG_RETENTION_DAYS"):
        monkeypatch.delenv(var, raising=False)
    yield
    handler = _get_app_file_handler()
    if handler is not None:
        for lg in _FILE_LOG_LOGGERS:
            if handler in lg.handlers:
                lg.removeHandler(handler)
        handler.close()


def test_add_file_logging_writes_marker_to_explicit_path(tmp_path, clean_file_logging):
    log_file = tmp_path / "litellm.log"
    resolved = add_file_logging(str(log_file))

    assert resolved == str(log_file)
    verbose_proxy_logger.error("hello-file-marker-123")

    assert log_file.exists()
    assert "hello-file-marker-123" in log_file.read_text()


def test_log_dir_env_resolves_default_filename(tmp_path, clean_file_logging, monkeypatch):
    monkeypatch.setenv("LITELLM_LOG_DIR", str(tmp_path))
    resolved = add_file_logging()

    assert resolved == str(tmp_path / "litellm.log")
    verbose_logger.error("dir-marker")
    assert (tmp_path / "litellm.log").exists()
    assert "dir-marker" in (tmp_path / "litellm.log").read_text()


def test_add_file_logging_is_idempotent(tmp_path, clean_file_logging):
    log_file = str(tmp_path / "litellm.log")
    add_file_logging(log_file)
    add_file_logging(log_file)

    for lg in _FILE_LOG_LOGGERS:
        file_handlers = [h for h in lg.handlers if isinstance(h, _SharedRotatingFileHandler)]
        assert len(file_handlers) == 1


def test_add_file_logging_no_op_without_config(clean_file_logging):
    assert add_file_logging() is None
    assert _get_app_file_handler() is None


def test_file_handler_is_daily_rotating_with_retention(tmp_path, clean_file_logging, monkeypatch):
    monkeypatch.setenv("LITELLM_LOG_RETENTION_DAYS", "7")
    add_file_logging(str(tmp_path / "litellm.log"))

    handler = _get_app_file_handler()
    assert isinstance(handler, _SharedRotatingFileHandler)
    assert handler.when == "MIDNIGHT"
    assert handler.backupCount == 7


@pytest.mark.skipif(sys.platform == "win32", reason="flock is POSIX-only")
def test_rollover_is_mutually_exclusive_across_holders(tmp_path, clean_file_logging):
    """While another holder owns the lock, _maybe_rollover must not rotate; once
    released, exactly one dated file is produced."""
    import fcntl

    log_file = tmp_path / "litellm.log"
    handler = add_file_logging(str(log_file)) and _get_app_file_handler()
    verbose_proxy_logger.error("before-rotate")

    def dated_files():
        return [p for p in glob.glob(str(log_file) + ".*") if not p.endswith(".rotate.lock")]

    # Hold the rotation lock from an independent fd -> handler cannot rotate.
    lock_fd = os.open(_lockfile_for(str(log_file)), os.O_CREAT | os.O_RDWR, 0o644)
    fcntl.flock(lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    handler._maybe_rollover()
    assert dated_files() == []  # contended -> skipped
    fcntl.flock(lock_fd, fcntl.LOCK_UN)
    os.close(lock_fd)

    # Now it wins the lock and rotates exactly once.
    handler._maybe_rollover()
    dated = dated_files()
    assert len(dated) == 1
    assert "before-rotate" in open(dated[0]).read()


def test_follower_reopens_new_base_after_rotation(tmp_path, clean_file_logging):
    """After rotation the base file is recreated on the next emit and old lines
    are not carried over."""
    log_file = tmp_path / "litellm.log"
    handler = add_file_logging(str(log_file)) and _get_app_file_handler()
    verbose_proxy_logger.error("old-line")

    handler._maybe_rollover()  # renames base -> dated (delay=True: base absent until next write)
    verbose_proxy_logger.error("new-line")

    assert log_file.exists()
    content = log_file.read_text()
    assert "new-line" in content
    assert "old-line" not in content


def test_file_log_is_redacted_and_has_no_ansi(tmp_path, clean_file_logging):
    log_file = tmp_path / "litellm.log"
    add_file_logging(str(log_file))

    verbose_proxy_logger.error("leaking sk-1234567890abcdefghij please")
    content = log_file.read_text()

    assert "sk-1234567890abcdefghij" not in content  # secret redaction filter ran
    assert "\033[" not in content  # no color escape codes in the file


def test_file_log_json_mode_writes_valid_json(tmp_path, clean_file_logging):
    log_file = tmp_path / "litellm.log"
    add_file_logging(str(log_file), use_json=True)

    verbose_proxy_logger.error("json-line-marker")
    lines = [l for l in log_file.read_text().splitlines() if "json-line-marker" in l]
    assert lines, "expected a json-formatted line containing the marker"
    parsed = json.loads(lines[-1])
    assert parsed["message"] == "json-line-marker"
    assert parsed["level"] == "ERROR"


def test_reattach_uses_json_formatter(tmp_path, clean_file_logging, monkeypatch):
    monkeypatch.setenv("LITELLM_LOG_DIR", str(tmp_path))
    add_file_logging()  # text formatter initially
    _reattach_app_file_handler(use_json=True)

    handler = _get_app_file_handler()
    assert isinstance(handler.formatter, JsonFormatter)


def test_uvicorn_log_config_adds_file_handler_when_dir_set(tmp_path, clean_file_logging, monkeypatch):
    monkeypatch.setenv("LITELLM_LOG_DIR", str(tmp_path))
    assert resolve_uvicorn_log_file() == str(tmp_path / "uvicorn.log")

    cfg = _get_uvicorn_log_config(use_json=False)
    file_handler = cfg["handlers"]["file"]
    assert file_handler["()"] == "litellm._logging._SharedRotatingFileHandler"
    assert file_handler["filename"] == str(tmp_path / "uvicorn.log")
    # the single shared file handler is attached to all uvicorn loggers
    assert "file" in cfg["loggers"]["uvicorn.access"]["handlers"]
    assert "file" in cfg["loggers"]["uvicorn.error"]["handlers"]


def test_uvicorn_log_config_no_file_handlers_without_dir(clean_file_logging):
    cfg = _get_uvicorn_log_config(use_json=True)
    assert "file" not in cfg["handlers"]
    assert cfg["loggers"]["uvicorn.access"]["handlers"] == ["access"]
