from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone
from typing import Mapping, cast

from litellm._logging import verbose_proxy_logger
from litellm.constants import (
    SPEND_LOG_CLEANUP_BATCH_FAILURE_BACKOFF_SECONDS,
    SPEND_LOG_CLEANUP_BATCH_SIZE,
    SPEND_LOG_CLEANUP_MAX_CONSECUTIVE_BATCH_FAILURES,
    SPEND_LOG_PROMPT_CLEANUP_JOB_NAME,
    SPEND_LOG_RUN_LOOPS,
)
from litellm.litellm_core_utils.duration_parser import duration_in_seconds
from litellm.proxy.utils import PrismaClient

SCRUB_OLD_PROMPTS_SQL = """
UPDATE "LiteLLM_SpendLogs"
SET "messages" = '{}', "response" = '{}', "proxy_server_request" = '{}'
WHERE ("request_id", "startTime") IN (
    SELECT "request_id", "startTime" FROM "LiteLLM_SpendLogs"
    WHERE "startTime" < $1::timestamptz
      AND ("messages" <> '{}'::jsonb
           OR "response" <> '{}'::jsonb
           OR "proxy_server_request" <> '{}'::jsonb)
    LIMIT $2
)
"""


def parse_prompt_retention_seconds(value: object) -> int | None:
    """
    Parse maximum_spend_logs_prompt_retention_period into seconds.

    This single duration drives both the scrub schedule (how often the job runs)
    and the age threshold (how old a log must be for its prompt to be scrubbed).
    Accepts a duration string like "10d"; a bare int is treated as days. Returns
    None when unset or invalid.
    """
    if isinstance(value, bool) or value is None:
        return None
    duration = f"{value}d" if isinstance(value, int) else value
    if not isinstance(duration, str):
        return None
    try:
        return duration_in_seconds(duration)
    except ValueError:
        return None


class SpendLogPromptCleanup:
    """
    Scrubs the detailed prompt/response content off old spend logs while keeping
    the cost/token/model accounting on the row.

    For rows older than maximum_spend_logs_prompt_retention_period, the
    messages/response/proxy_server_request columns are reset to '{}' (their
    default, and the value written when store_prompts_in_spend_logs is off).
    Uses PodLockManager so only one pod runs the scrub in multi-pod deployments.
    """

    def __init__(self, general_settings: Mapping[str, object] | None = None) -> None:
        self.batch_size = SPEND_LOG_CLEANUP_BATCH_SIZE
        self.retention_seconds: int | None = None
        from litellm.proxy.proxy_server import general_settings as default_settings

        self.general_settings: Mapping[str, object] = (
            general_settings if general_settings is not None else cast(Mapping[str, object], default_settings)
        )
        from litellm.proxy.proxy_server import proxy_logging_obj

        self.pod_lock_manager = proxy_logging_obj.db_spend_update_writer.pod_lock_manager

    def _should_scrub_prompts(self) -> bool:
        retention_setting = self.general_settings.get("maximum_spend_logs_prompt_retention_period")
        retention_seconds = parse_prompt_retention_seconds(retention_setting)
        if retention_seconds is None:
            if retention_setting is not None:
                verbose_proxy_logger.warning(
                    f"Invalid maximum_spend_logs_prompt_retention_period value: {retention_setting}"
                )
            return False

        self.retention_seconds = retention_seconds
        return True

    async def _scrub_old_prompts(self, prisma_client: PrismaClient, cutoff_date: datetime) -> int:
        """
        Scrub prompt content off old logs in batches. Returns the number of rows scrubbed.

        The '<> {}' filter restricts each batch to not-yet-scrubbed rows so the
        loop converges to a 0-row batch; without it, already-scrubbed old rows
        keep matching the cutoff and the loop never terminates.
        """
        total_scrubbed = 0
        run_count = 0
        consecutive_failures = 0
        while True:
            if run_count > SPEND_LOG_RUN_LOOPS:
                verbose_proxy_logger.info("Max prompt-scrub batches reached; remaining logs will be scrubbed next run")
                break
            try:
                scrubbed_result: object = await prisma_client.db.execute_raw(
                    SCRUB_OLD_PROMPTS_SQL,
                    cutoff_date,
                    self.batch_size,
                )
            except Exception as batch_exc:
                consecutive_failures += 1
                verbose_proxy_logger.exception(
                    "Spend log prompt scrub batch failed "
                    "(run_count=%d, consecutive_failures=%d, batch_size=%d, "
                    "cutoff=%s, total_scrubbed_so_far=%d): %s: %s",
                    run_count,
                    consecutive_failures,
                    self.batch_size,
                    cutoff_date.isoformat(),
                    total_scrubbed,
                    type(batch_exc).__name__,
                    batch_exc,
                )
                if consecutive_failures >= SPEND_LOG_CLEANUP_MAX_CONSECUTIVE_BATCH_FAILURES:
                    verbose_proxy_logger.error(
                        "Aborting spend log prompt scrub after %d consecutive batch "
                        "failures; total scrubbed before abort: %d",
                        consecutive_failures,
                        total_scrubbed,
                    )
                    break
                await asyncio.sleep(SPEND_LOG_CLEANUP_BATCH_FAILURE_BACKOFF_SECONDS)
                continue

            consecutive_failures = 0

            if not isinstance(scrubbed_result, int):
                verbose_proxy_logger.error(
                    f"Unexpected execute_raw return type for prompt scrub: {type(scrubbed_result)}; "
                    "aborting to avoid infinite loop"
                )
                break

            verbose_proxy_logger.info(f"Scrubbed prompt content from {scrubbed_result} logs in this batch")

            if scrubbed_result == 0:
                break

            total_scrubbed += scrubbed_result
            run_count += 1
            await asyncio.sleep(0.1)

        return total_scrubbed

    async def scrub_old_prompts(self, prisma_client: PrismaClient) -> None:
        """
        Entry point registered as the scheduled job. Scrubs prompt content off
        logs older than the retention period, guarded by the pod lock.
        """
        lock_acquired = False
        try:
            if not self._should_scrub_prompts():
                return

            if self.retention_seconds is None:
                return

            if self.pod_lock_manager and self.pod_lock_manager.redis_cache:
                lock_acquired = (
                    await self.pod_lock_manager.acquire_lock(
                        cronjob_id=SPEND_LOG_PROMPT_CLEANUP_JOB_NAME,
                    )
                    or False
                )
                if not lock_acquired:
                    verbose_proxy_logger.info("Another pod is already running prompt scrub")
                    return

            cutoff_date = datetime.now(timezone.utc) - timedelta(seconds=float(self.retention_seconds))
            verbose_proxy_logger.info(f"Scrubbing prompt content from logs older than {cutoff_date.isoformat()}")
            total_scrubbed = await self._scrub_old_prompts(prisma_client, cutoff_date)
            verbose_proxy_logger.info(f"Scrubbed prompt content from {total_scrubbed} logs")
        except Exception as e:
            verbose_proxy_logger.exception(
                "Error during spend log prompt scrub: %s: %s",
                type(e).__name__,
                e,
            )
            return
        finally:
            if lock_acquired and self.pod_lock_manager and self.pod_lock_manager.redis_cache:
                await self.pod_lock_manager.release_lock(cronjob_id=SPEND_LOG_PROMPT_CLEANUP_JOB_NAME)
