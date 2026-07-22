import json
from datetime import datetime
from typing import List, Optional

import litellm
from litellm._logging import verbose_proxy_logger
from litellm.caching.caching import DualCache
from litellm.exceptions import RateLimitErrorCategory, RateLimitType
from litellm.integrations.custom_logger import Span
from litellm.proxy._types import UserAPIKeyAuth
from litellm.router_strategy.budget_limiter import RouterBudgetLimiting
from litellm.types.llms.openai import AllMessageValues
from litellm.types.utils import (
    BudgetConfig,
    GenericBudgetConfigType,
    StandardLoggingPayload,
)

VIRTUAL_KEY_SPEND_CACHE_KEY_PREFIX = "virtual_key_spend"
END_USER_SPEND_CACHE_KEY_PREFIX = "end_user_model_spend"
USER_GROUP_SPEND_CACHE_KEY_PREFIX = "user_model_group_spend"
TEAM_MEMBER_GROUP_SPEND_CACHE_KEY_PREFIX = "team_member_model_group_spend"
USER_GROUP_RATE_CACHE_KEY_PREFIX = "user_model_group_rate"
TEAM_MEMBER_GROUP_RATE_CACHE_KEY_PREFIX = "team_member_model_group_rate"


class _PROXY_VirtualKeyModelMaxBudgetLimiter(RouterBudgetLimiting):
    """
    Handles budgets for model + virtual key

    Example: key=sk-1234567890, model=gpt-4o, max_budget=100, time_period=1d
    """

    def __init__(self, dual_cache: DualCache):
        self.dual_cache = dual_cache
        self.redis_increment_operation_queue = []
        self.deployment_budget_config = None

    async def is_key_within_model_budget(
        self,
        user_api_key_dict: UserAPIKeyAuth,
        model: str,
    ) -> bool:
        """
        Check if the user_api_key_dict is within the model budget

        Raises:
            BudgetExceededError: If the user_api_key_dict has exceeded the model budget
        """
        _model_max_budget = user_api_key_dict.model_max_budget
        internal_model_max_budget: GenericBudgetConfigType = {}

        for _model, _budget_info in _model_max_budget.items():
            internal_model_max_budget[_model] = BudgetConfig(**_budget_info)

        verbose_proxy_logger.debug(
            "internal_model_max_budget %s",
            json.dumps(internal_model_max_budget, indent=4, default=str),
        )

        # check if current model is in internal_model_max_budget
        _current_model_budget_info = self._get_request_model_budget_config(
            model=model, internal_model_max_budget=internal_model_max_budget
        )
        if _current_model_budget_info is None:
            verbose_proxy_logger.debug(f"Model {model} not found in internal_model_max_budget")
            return True

        # check if current model is within budget
        if _current_model_budget_info.max_budget and _current_model_budget_info.max_budget > 0:
            _current_spend = await self._get_virtual_key_spend_for_model(
                user_api_key_hash=user_api_key_dict.token,
                model=model,
                key_budget_config=_current_model_budget_info,
            )
            if (
                _current_spend is not None
                and _current_model_budget_info.max_budget is not None
                and _current_spend > _current_model_budget_info.max_budget
            ):
                raise litellm.BudgetExceededError(
                    message=f"LiteLLM Virtual Key: {user_api_key_dict.token}, key_alias: {user_api_key_dict.key_alias}, exceeded budget for model={model}",
                    current_cost=_current_spend,
                    max_budget=_current_model_budget_info.max_budget,
                )

        return True

    async def get_fallback_model_within_budget(
        self,
        user_api_key_dict: UserAPIKeyAuth,
        model: str,
    ) -> Optional[str]:
        budget_fallbacks: dict[str, list[str]] = user_api_key_dict.budget_fallbacks or {}
        for fallback_model in budget_fallbacks.get(model, []):
            try:
                await self.is_key_within_model_budget(user_api_key_dict=user_api_key_dict, model=fallback_model)
                return fallback_model
            except litellm.BudgetExceededError:
                continue
        return None

    async def is_end_user_within_model_budget(
        self,
        end_user_id: str,
        end_user_model_max_budget: dict,
        model: str,
    ) -> bool:
        """
        Check if the end_user is within the model budget

        Raises:
            BudgetExceededError: If the end_user has exceeded the model budget
        """
        internal_model_max_budget: GenericBudgetConfigType = {}

        for _model, _budget_info in end_user_model_max_budget.items():
            internal_model_max_budget[_model] = BudgetConfig(**_budget_info)

        verbose_proxy_logger.debug(
            "end_user internal_model_max_budget %s",
            json.dumps(internal_model_max_budget, indent=4, default=str),
        )

        # check if current model is in internal_model_max_budget
        _current_model_budget_info = self._get_request_model_budget_config(
            model=model, internal_model_max_budget=internal_model_max_budget
        )
        if _current_model_budget_info is None:
            verbose_proxy_logger.debug(f"Model {model} not found in end_user_model_max_budget")
            return True

        # check if current model is within budget
        if _current_model_budget_info.max_budget and _current_model_budget_info.max_budget > 0:
            _current_spend = await self._get_end_user_spend_for_model(
                end_user_id=end_user_id,
                model=model,
                key_budget_config=_current_model_budget_info,
            )
            if (
                _current_spend is not None
                and _current_model_budget_info.max_budget is not None
                and _current_spend > _current_model_budget_info.max_budget
            ):
                raise litellm.BudgetExceededError(
                    message=f"LiteLLM End User: {end_user_id}, exceeded budget for model={model}",
                    current_cost=_current_spend,
                    max_budget=_current_model_budget_info.max_budget,
                )

        return True

    async def is_user_within_model_group_budget(
        self,
        user_id: str,
        user_model_group_max_budget: dict,
        groups: List[str],
    ) -> bool:
        """Check the user's aggregate spend (across all their keys) for each model group.

        Raises BudgetExceededError if any matching group is over budget, or
        RateLimitError if any matching group is over its TPM/RPM limit.
        """
        await self._is_within_model_group_budget(
            scope_prefix=USER_GROUP_SPEND_CACHE_KEY_PREFIX,
            scope_id=user_id,
            model_group_max_budget=user_model_group_max_budget,
            groups=groups,
            error_prefix=f"LiteLLM User={user_id}",
        )
        return await self._is_within_model_group_rate_limit(
            scope_prefix=USER_GROUP_RATE_CACHE_KEY_PREFIX,
            scope_id=user_id,
            model_group_max_budget=user_model_group_max_budget,
            groups=groups,
            error_prefix=f"LiteLLM User={user_id}",
        )

    async def is_team_member_within_model_group_budget(
        self,
        user_id: str,
        team_id: str,
        team_member_model_group_max_budget: dict,
        groups: List[str],
    ) -> bool:
        """Check a team member's per-group spend within their team.

        Raises BudgetExceededError if any matching group is over budget, or
        RateLimitError if any matching group is over its TPM/RPM limit.
        """
        await self._is_within_model_group_budget(
            scope_prefix=TEAM_MEMBER_GROUP_SPEND_CACHE_KEY_PREFIX,
            scope_id=f"{user_id}:{team_id}",
            model_group_max_budget=team_member_model_group_max_budget,
            groups=groups,
            error_prefix=f"LiteLLM User={user_id} in Team={team_id}",
        )
        return await self._is_within_model_group_rate_limit(
            scope_prefix=TEAM_MEMBER_GROUP_RATE_CACHE_KEY_PREFIX,
            scope_id=f"{user_id}:{team_id}",
            model_group_max_budget=team_member_model_group_max_budget,
            groups=groups,
            error_prefix=f"LiteLLM User={user_id} in Team={team_id}",
        )

    async def _is_within_model_group_budget(
        self,
        *,
        scope_prefix: str,
        scope_id: str,
        model_group_max_budget: dict,
        groups: List[str],
        error_prefix: str,
    ) -> bool:
        for group in groups:
            budget_config = self._group_budget_config(model_group_max_budget, group)
            if budget_config is None:
                continue
            spend_key = f"{scope_prefix}:{scope_id}:{group}:{budget_config.budget_duration}"
            current_spend = await self.dual_cache.async_get_cache(key=spend_key) or 0.0
            if budget_config.max_budget is not None and current_spend > budget_config.max_budget:
                raise litellm.BudgetExceededError(
                    message=f"{error_prefix}, exceeded budget for model group={group}",
                    current_cost=current_spend,
                    max_budget=budget_config.max_budget,
                )
        return True

    @staticmethod
    def _group_budget_config(model_group_max_budget: dict, group: str) -> Optional[BudgetConfig]:
        budget_info = model_group_max_budget.get(group)
        if budget_info is None:
            return None
        budget_config = BudgetConfig(**budget_info)
        if not budget_config.max_budget or budget_config.max_budget <= 0:
            return None
        return budget_config

    async def _increment_model_group_spend(
        self,
        *,
        scope_prefix: str,
        scope_id: str,
        model_group_max_budget: dict,
        groups: List[str],
        response_cost: float,
    ) -> None:
        for group in groups:
            budget_config = self._group_budget_config(model_group_max_budget, group)
            if budget_config is None or not budget_config.budget_duration:
                continue
            spend_key = f"{scope_prefix}:{scope_id}:{group}:{budget_config.budget_duration}"
            start_time_key = f"{scope_prefix}_start_time:{scope_id}:{group}"
            await self._increment_spend_for_key(
                budget_config=budget_config,
                spend_key=spend_key,
                start_time_key=start_time_key,
                response_cost=response_cost,
            )

    @staticmethod
    def _current_minute() -> str:
        return datetime.now().strftime("%Y-%m-%d-%H-%M")

    @staticmethod
    def _group_rate_config(model_group_max_budget: dict, group: str) -> Optional[BudgetConfig]:
        budget_info = model_group_max_budget.get(group)
        if budget_info is None:
            return None
        rate_config = BudgetConfig(**budget_info)
        has_tpm = rate_config.tpm_limit is not None and rate_config.tpm_limit > 0
        has_rpm = rate_config.rpm_limit is not None and rate_config.rpm_limit > 0
        if not has_tpm and not has_rpm:
            return None
        return rate_config

    async def _is_within_model_group_rate_limit(
        self,
        *,
        scope_prefix: str,
        scope_id: str,
        model_group_max_budget: dict,
        groups: List[str],
        error_prefix: str,
    ) -> bool:
        minute = self._current_minute()
        for group in groups:
            rate_config = self._group_rate_config(model_group_max_budget, group)
            if rate_config is None:
                continue
            tpm_key = f"{scope_prefix}_tpm:{scope_id}:{group}:{minute}"
            rpm_key = f"{scope_prefix}_rpm:{scope_id}:{group}:{minute}"
            current_tpm = await self.dual_cache.async_get_cache(key=tpm_key) or 0
            current_rpm = await self.dual_cache.async_get_cache(key=rpm_key) or 0
            if rate_config.rpm_limit is not None and current_rpm >= rate_config.rpm_limit:
                raise litellm.RateLimitError(
                    message=f"{error_prefix}, exceeded RPM limit for model group={group}. current rpm: {current_rpm}, rpm limit: {rate_config.rpm_limit}",
                    llm_provider="litellm",
                    model=group,
                    category=RateLimitErrorCategory.LITELLM_RATE_LIMIT,
                    rate_limit_type=RateLimitType.REQUESTS,
                )
            if rate_config.tpm_limit is not None and current_tpm >= rate_config.tpm_limit:
                raise litellm.RateLimitError(
                    message=f"{error_prefix}, exceeded TPM limit for model group={group}. current tpm: {current_tpm}, tpm limit: {rate_config.tpm_limit}",
                    llm_provider="litellm",
                    model=group,
                    category=RateLimitErrorCategory.LITELLM_RATE_LIMIT,
                    rate_limit_type=RateLimitType.TOKENS,
                )
        return True

    async def _increment_model_group_rate(
        self,
        *,
        scope_prefix: str,
        scope_id: str,
        model_group_max_budget: dict,
        groups: List[str],
        total_tokens: int,
    ) -> None:
        minute = self._current_minute()
        for group in groups:
            rate_config = self._group_rate_config(model_group_max_budget, group)
            if rate_config is None:
                continue
            rpm_key = f"{scope_prefix}_rpm:{scope_id}:{group}:{minute}"
            tpm_key = f"{scope_prefix}_tpm:{scope_id}:{group}:{minute}"
            await self.dual_cache.async_increment_cache(key=rpm_key, value=1, ttl=60)
            if total_tokens:
                await self.dual_cache.async_increment_cache(key=tpm_key, value=total_tokens, ttl=60)

    async def _get_end_user_spend_for_model(
        self,
        end_user_id: str,
        model: str,
        key_budget_config: BudgetConfig,
    ) -> Optional[float]:
        # 1. model: directly look up `model`
        end_user_model_spend_cache_key = (
            f"{END_USER_SPEND_CACHE_KEY_PREFIX}:{end_user_id}:{model}:{key_budget_config.budget_duration}"
        )
        _current_spend = await self.dual_cache.async_get_cache(
            key=end_user_model_spend_cache_key,
        )

        if _current_spend is None:
            # 2. If 1, does not exist, check if passed as {custom_llm_provider}/model
            end_user_model_spend_cache_key = f"{END_USER_SPEND_CACHE_KEY_PREFIX}:{end_user_id}:{self._get_model_without_custom_llm_provider(model)}:{key_budget_config.budget_duration}"
            _current_spend = await self.dual_cache.async_get_cache(
                key=end_user_model_spend_cache_key,
            )
        return _current_spend

    async def _get_virtual_key_spend_for_model(
        self,
        user_api_key_hash: Optional[str],
        model: str,
        key_budget_config: BudgetConfig,
    ) -> Optional[float]:
        """
        Get the current spend for a virtual key for a model

        Lookup model in this order:
            1. model: directly look up `model`
            2. If 1, does not exist, check if passed as {custom_llm_provider}/model
        """

        # 1. model: directly look up `model`
        virtual_key_model_spend_cache_key = (
            f"{VIRTUAL_KEY_SPEND_CACHE_KEY_PREFIX}:{user_api_key_hash}:{model}:{key_budget_config.budget_duration}"
        )
        _current_spend = await self.dual_cache.async_get_cache(
            key=virtual_key_model_spend_cache_key,
        )

        if _current_spend is None:
            # 2. If 1, does not exist, check if passed as {custom_llm_provider}/model
            # if "/" in model, remove first part before "/" - eg. openai/o1-preview -> o1-preview
            virtual_key_model_spend_cache_key = f"{VIRTUAL_KEY_SPEND_CACHE_KEY_PREFIX}:{user_api_key_hash}:{self._get_model_without_custom_llm_provider(model)}:{key_budget_config.budget_duration}"
            _current_spend = await self.dual_cache.async_get_cache(
                key=virtual_key_model_spend_cache_key,
            )
        return _current_spend

    def _get_request_model_budget_config(
        self, model: str, internal_model_max_budget: GenericBudgetConfigType
    ) -> Optional[BudgetConfig]:
        """
        Get the budget config for the request model

        1. Check if `model` is in `internal_model_max_budget`
        2. If not, check if `model` without custom llm provider is in `internal_model_max_budget`
        """
        return internal_model_max_budget.get(model, None) or internal_model_max_budget.get(
            self._get_model_without_custom_llm_provider(model), None
        )

    def _get_model_without_custom_llm_provider(self, model: str) -> str:
        if "/" in model:
            return model.split("/")[-1]
        return model

    async def async_filter_deployments(
        self,
        model: str,
        healthy_deployments: List,
        messages: Optional[List[AllMessageValues]],
        request_kwargs: Optional[dict] = None,
        parent_otel_span: Optional[Span] = None,  # type: ignore
    ) -> List[dict]:
        return healthy_deployments

    async def async_log_success_event(self, kwargs, response_obj, start_time, end_time):
        """
        Track spend for virtual key + model in DualCache

        Example: key=sk-1234567890, model=gpt-4o, max_budget=100, time_period=1d
        """
        verbose_proxy_logger.debug("in RouterBudgetLimiting.async_log_success_event")
        standard_logging_payload: Optional[StandardLoggingPayload] = kwargs.get("standard_logging_object", None)
        if standard_logging_payload is None:
            verbose_proxy_logger.debug(
                "Skipping _PROXY_VirtualKeyModelMaxBudgetLimiter.async_log_success_event: standard_logging_payload is None"
            )
            return

        _litellm_params: dict = kwargs.get("litellm_params", {}) or {}
        _metadata: dict = _litellm_params.get("metadata", {}) or {}
        user_api_key_model_max_budget: Optional[dict] = _metadata.get("user_api_key_model_max_budget", None)
        user_api_key_end_user_model_max_budget: Optional[dict] = _metadata.get(
            "user_api_key_end_user_model_max_budget", None
        )
        user_model_group_max_budget: Optional[dict] = _metadata.get("user_api_key_user_model_group_max_budget", None)
        team_member_model_group_max_budget: Optional[dict] = _metadata.get(
            "user_api_key_team_member_model_group_max_budget", None
        )
        if not any(
            (
                user_api_key_model_max_budget,
                user_api_key_end_user_model_max_budget,
                user_model_group_max_budget,
                team_member_model_group_max_budget,
            )
        ):
            verbose_proxy_logger.debug(
                "Not running _PROXY_VirtualKeyModelMaxBudgetLimiter.async_log_success_event because no model or model-group budgets are configured."
            )
            return

        response_cost: float = standard_logging_payload.get("response_cost", 0)
        # Use model_group (the user-facing model alias, e.g. "gpt-4o") when
        # available.  The enforcement path (is_key_within_model_budget) receives
        # the model name from request_data["model"] which is the model group
        # alias, so the spend tracking cache key must use the same name.
        # Falling back to the deployment-level "model" field preserves
        # behaviour for non-proxy or non-router deployments where model_group
        # is None.
        model = standard_logging_payload.get("model_group") or standard_logging_payload.get("model")
        virtual_key = standard_logging_payload.get("metadata", {}).get("user_api_key_hash")
        end_user_id = standard_logging_payload.get("end_user") or standard_logging_payload.get("metadata", {}).get(
            "user_api_key_end_user_id"
        )

        if model is None:
            return

        if (
            virtual_key is not None
            and user_api_key_model_max_budget is not None
            and len(user_api_key_model_max_budget) > 0
        ):
            internal_model_max_budget: GenericBudgetConfigType = {}
            for _model, _budget_info in user_api_key_model_max_budget.items():
                internal_model_max_budget[_model] = BudgetConfig(**_budget_info)
            key_budget_config = self._get_request_model_budget_config(
                model=model, internal_model_max_budget=internal_model_max_budget
            )
            if key_budget_config is not None and key_budget_config.budget_duration:
                virtual_spend_key = (
                    f"{VIRTUAL_KEY_SPEND_CACHE_KEY_PREFIX}:{virtual_key}:{model}:{key_budget_config.budget_duration}"
                )
                virtual_start_time_key = f"virtual_key_budget_start_time:{virtual_key}"
                await self._increment_spend_for_key(
                    budget_config=key_budget_config,
                    spend_key=virtual_spend_key,
                    start_time_key=virtual_start_time_key,
                    response_cost=response_cost,
                )

        if (
            end_user_id is not None
            and user_api_key_end_user_model_max_budget is not None
            and len(user_api_key_end_user_model_max_budget) > 0
        ):
            internal_model_max_budget: GenericBudgetConfigType = {}
            for _model, _budget_info in user_api_key_end_user_model_max_budget.items():
                internal_model_max_budget[_model] = BudgetConfig(**_budget_info)
            key_budget_config = self._get_request_model_budget_config(
                model=model, internal_model_max_budget=internal_model_max_budget
            )
            if key_budget_config is not None and key_budget_config.budget_duration:
                end_user_spend_key = (
                    f"{END_USER_SPEND_CACHE_KEY_PREFIX}:{end_user_id}:{model}:{key_budget_config.budget_duration}"
                )
                end_user_start_time_key = f"end_user_budget_start_time:{end_user_id}"
                await self._increment_spend_for_key(
                    budget_config=key_budget_config,
                    spend_key=end_user_spend_key,
                    start_time_key=end_user_start_time_key,
                    response_cost=response_cost,
                )

        if user_model_group_max_budget or team_member_model_group_max_budget:
            from litellm.proxy.auth.auth_checks import get_budgeted_groups_for_model
            from litellm.proxy.proxy_server import (
                llm_router,
                prisma_client,
                proxy_logging_obj,
                user_api_key_cache,
            )

            sl_metadata: dict = standard_logging_payload.get("metadata", {}) or {}
            user_id = sl_metadata.get("user_api_key_user_id")
            team_id = sl_metadata.get("user_api_key_team_id")
            total_tokens: int = standard_logging_payload.get("total_tokens", 0) or 0

            if user_id and user_model_group_max_budget:
                user_groups = await get_budgeted_groups_for_model(
                    model=model,
                    model_group_max_budget=user_model_group_max_budget,
                    llm_router=llm_router,
                    prisma_client=prisma_client,
                    user_api_key_cache=user_api_key_cache,
                    proxy_logging_obj=proxy_logging_obj,
                )
                if user_groups:
                    await self._increment_model_group_spend(
                        scope_prefix=USER_GROUP_SPEND_CACHE_KEY_PREFIX,
                        scope_id=user_id,
                        model_group_max_budget=user_model_group_max_budget,
                        groups=user_groups,
                        response_cost=response_cost,
                    )
                    await self._increment_model_group_rate(
                        scope_prefix=USER_GROUP_RATE_CACHE_KEY_PREFIX,
                        scope_id=user_id,
                        model_group_max_budget=user_model_group_max_budget,
                        groups=user_groups,
                        total_tokens=total_tokens,
                    )

            if user_id and team_id and team_member_model_group_max_budget:
                team_groups = await get_budgeted_groups_for_model(
                    model=model,
                    model_group_max_budget=team_member_model_group_max_budget,
                    llm_router=llm_router,
                    prisma_client=prisma_client,
                    user_api_key_cache=user_api_key_cache,
                    proxy_logging_obj=proxy_logging_obj,
                )
                if team_groups:
                    await self._increment_model_group_spend(
                        scope_prefix=TEAM_MEMBER_GROUP_SPEND_CACHE_KEY_PREFIX,
                        scope_id=f"{user_id}:{team_id}",
                        model_group_max_budget=team_member_model_group_max_budget,
                        groups=team_groups,
                        response_cost=response_cost,
                    )
                    await self._increment_model_group_rate(
                        scope_prefix=TEAM_MEMBER_GROUP_RATE_CACHE_KEY_PREFIX,
                        scope_id=f"{user_id}:{team_id}",
                        model_group_max_budget=team_member_model_group_max_budget,
                        groups=team_groups,
                        total_tokens=total_tokens,
                    )

        if self.dual_cache.redis_cache is not None:
            await self._push_in_memory_increments_to_redis()

        verbose_proxy_logger.debug(
            "current state of in memory cache %s",
            json.dumps(self.dual_cache.in_memory_cache.cache_dict, indent=4, default=str),
        )
