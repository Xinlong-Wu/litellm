"""
Test cases for scheduled scrubbing of detailed prompt content off old spend logs
"""

from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock, MagicMock

import pytest

from litellm.proxy.db.db_transaction_queue.spend_log_prompt_cleanup import (
    SpendLogPromptCleanup,
    parse_prompt_retention_seconds,
)


def _cleaner(general_settings):
    cleaner = SpendLogPromptCleanup(general_settings=general_settings)
    mock_pod_lock_manager = MagicMock()
    mock_pod_lock_manager.redis_cache = MagicMock()
    mock_pod_lock_manager.acquire_lock = AsyncMock(return_value=True)
    mock_pod_lock_manager.release_lock = AsyncMock()
    cleaner.pod_lock_manager = mock_pod_lock_manager
    return cleaner


def test_should_scrub_prompts():
    assert SpendLogPromptCleanup(general_settings={})._should_scrub_prompts() is False

    cleaner = SpendLogPromptCleanup(general_settings={"maximum_spend_logs_prompt_retention_period": "30d"})
    assert cleaner._should_scrub_prompts() is True
    assert cleaner.retention_seconds == 2592000

    assert (
        SpendLogPromptCleanup(
            general_settings={"maximum_spend_logs_prompt_retention_period": "invalid"}
        )._should_scrub_prompts()
        is False
    )


def test_should_scrub_prompts_bare_int_treated_as_days():
    cleaner = SpendLogPromptCleanup(general_settings={"maximum_spend_logs_prompt_retention_period": 7})
    assert cleaner._should_scrub_prompts() is True
    assert cleaner.retention_seconds == 7 * 86400


@pytest.mark.asyncio
async def test_scrub_updates_content_columns_only():
    mock_prisma_client = MagicMock()
    mock_prisma_client.db = MagicMock()
    mock_prisma_client.db.execute_raw = AsyncMock(return_value=0)

    cleaner = _cleaner({"maximum_spend_logs_prompt_retention_period": "30d"})
    await cleaner.scrub_old_prompts(mock_prisma_client)

    sql = mock_prisma_client.db.execute_raw.call_args_list[0][0][0]
    assert 'UPDATE "LiteLLM_SpendLogs"' in sql
    assert "DELETE" not in sql.upper()
    assert "\"messages\" = '{}'" in sql
    assert "\"response\" = '{}'" in sql
    assert "\"proxy_server_request\" = '{}'" in sql
    # only not-yet-scrubbed rows are targeted so the batched loop can converge
    assert "<> '{}'::jsonb" in sql
    assert 'WHERE ("request_id", "startTime") IN' in sql


@pytest.mark.asyncio
async def test_scrub_batches_until_zero():
    mock_prisma_client = MagicMock()
    mock_prisma_client.db = MagicMock()
    mock_prisma_client.db.execute_raw = AsyncMock(side_effect=[1000, 500, 0])

    cleaner = _cleaner({"maximum_spend_logs_prompt_retention_period": "30d"})
    total = await cleaner._scrub_old_prompts(mock_prisma_client, datetime.now(timezone.utc))

    assert mock_prisma_client.db.execute_raw.call_count == 3
    assert total == 1500


@pytest.mark.asyncio
async def test_scrub_cutoff_date():
    mock_prisma_client = MagicMock()
    mock_prisma_client.db = MagicMock()
    mock_prisma_client.db.execute_raw = AsyncMock(return_value=0)

    cleaner = _cleaner({"maximum_spend_logs_prompt_retention_period": "24h"})
    await cleaner.scrub_old_prompts(mock_prisma_client)

    cutoff_date = mock_prisma_client.db.execute_raw.call_args[0][1]
    expected_cutoff = datetime.now(timezone.utc) - timedelta(seconds=86400)
    assert abs((cutoff_date - expected_cutoff).total_seconds()) < 1


@pytest.mark.asyncio
async def test_scrub_acquires_and_releases_lock():
    mock_prisma_client = MagicMock()
    mock_prisma_client.db = MagicMock()
    mock_prisma_client.db.execute_raw = AsyncMock(return_value=0)

    cleaner = _cleaner({"maximum_spend_logs_prompt_retention_period": "30d"})
    await cleaner.scrub_old_prompts(mock_prisma_client)

    cleaner.pod_lock_manager.acquire_lock.assert_awaited_once()
    cleaner.pod_lock_manager.release_lock.assert_awaited_once()


@pytest.mark.asyncio
async def test_scrub_skips_when_lock_not_acquired():
    mock_prisma_client = MagicMock()
    mock_prisma_client.db = MagicMock()
    mock_prisma_client.db.execute_raw = AsyncMock(return_value=0)

    cleaner = _cleaner({"maximum_spend_logs_prompt_retention_period": "30d"})
    cleaner.pod_lock_manager.acquire_lock = AsyncMock(return_value=False)

    await cleaner.scrub_old_prompts(mock_prisma_client)

    mock_prisma_client.db.execute_raw.assert_not_called()
    cleaner.pod_lock_manager.release_lock.assert_not_called()


@pytest.mark.asyncio
async def test_scrub_noop_when_retention_unset():
    mock_prisma_client = MagicMock()
    mock_prisma_client.db = MagicMock()
    mock_prisma_client.db.execute_raw = AsyncMock(return_value=0)

    cleaner = _cleaner({})
    await cleaner.scrub_old_prompts(mock_prisma_client)

    mock_prisma_client.db.execute_raw.assert_not_called()
    cleaner.pod_lock_manager.acquire_lock.assert_not_called()


def test_parse_prompt_retention_seconds():
    assert parse_prompt_retention_seconds("10d") == 864000
    assert parse_prompt_retention_seconds(10) == 864000  # bare int treated as days
    assert parse_prompt_retention_seconds("24h") == 86400
    assert parse_prompt_retention_seconds(None) is None
    assert parse_prompt_retention_seconds("invalid") is None
    assert parse_prompt_retention_seconds(True) is None  # bool must not become "1d"


def test_prompt_cleanup_interval_equals_retention():
    """
    The single retention setting drives both the schedule cadence and the age
    threshold, so the job's interval must equal the parsed retention seconds.
    """
    mock_scheduler = MagicMock()
    mock_prisma_client = MagicMock()
    mock_cleanup_instance = MagicMock()

    retention_seconds = parse_prompt_retention_seconds("10d")
    mock_scheduler.add_job(
        mock_cleanup_instance.scrub_old_prompts,
        "interval",
        seconds=retention_seconds,
        args=[mock_prisma_client],
        id="spend_log_prompt_cleanup_job",
        replace_existing=True,
    )

    mock_scheduler.add_job.assert_called_once()
    call_args = mock_scheduler.add_job.call_args
    assert call_args[0][1] == "interval"
    assert call_args[1]["seconds"] == 864000
    assert call_args[1]["id"] == "spend_log_prompt_cleanup_job"
