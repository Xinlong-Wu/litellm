"""
Unit tests for per-model-group budget enforcement helpers in auth_checks.

Covers the user-level (aggregate across keys) and team-member-level
(per-member override -> team default) enforcement wired into common_checks.

The "model group" is the UI-managed access group (LiteLLM_AccessGroupTable):
budgets are keyed by access_group_id and membership is resolved through
_get_models_from_access_groups + _can_object_call_model.
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

import litellm
from litellm.caching.caching import DualCache
from litellm.proxy._types import (
    LiteLLM_BudgetTable,
    LiteLLM_TeamMembership,
    LiteLLM_TeamTable,
    LiteLLM_UserTable,
    UserAPIKeyAuth,
)
from litellm.proxy.auth.auth_checks import (
    _check_team_member_model_group_budget,
    _check_user_model_group_budget,
)
from litellm.proxy.hooks.model_max_budget_limiter import (
    TEAM_MEMBER_GROUP_RATE_CACHE_KEY_PREFIX,
    USER_GROUP_RATE_CACHE_KEY_PREFIX,
    USER_GROUP_SPEND_CACHE_KEY_PREFIX,
    _PROXY_VirtualKeyModelMaxBudgetLimiter,
)

# budget keyed by access_group_id
GROUP_ID = "ag-premium"
GROUP_BUDGET = {GROUP_ID: {"max_budget": 0.01, "budget_duration": "30d"}}
# rate limit keyed by access_group_id (rides in the same map)
GROUP_RPM = {GROUP_ID: {"rpm_limit": 1}}
# models that belong to the access group (returned by the patched resolver)
GROUP_MODELS = ["gpt-4o"]


async def _seeded_limiter(scope_prefix, scope_id, cost) -> _PROXY_VirtualKeyModelMaxBudgetLimiter:
    limiter = _PROXY_VirtualKeyModelMaxBudgetLimiter(dual_cache=DualCache())
    await limiter._increment_model_group_spend(
        scope_prefix=scope_prefix,
        scope_id=scope_id,
        model_group_max_budget=GROUP_BUDGET,
        groups=[GROUP_ID],
        response_cost=cost,
    )
    return limiter


async def _rate_seeded_limiter(scope_prefix, scope_id) -> _PROXY_VirtualKeyModelMaxBudgetLimiter:
    limiter = _PROXY_VirtualKeyModelMaxBudgetLimiter(dual_cache=DualCache())
    await limiter._increment_model_group_rate(
        scope_prefix=scope_prefix,
        scope_id=scope_id,
        model_group_max_budget=GROUP_RPM,
        groups=[GROUP_ID],
        total_tokens=5,
    )
    return limiter


def _patch_group_models():
    """Resolve the access group to its member models (cache/DB stand-in)."""
    return patch(
        "litellm.proxy.auth.auth_checks._get_models_from_access_groups",
        new_callable=AsyncMock,
        return_value=GROUP_MODELS,
    )


@pytest.mark.asyncio
async def test_user_model_group_budget_blocks_when_over_cap():
    limiter = await _seeded_limiter(USER_GROUP_SPEND_CACHE_KEY_PREFIX, "user-1", cost=0.02)
    user_object = LiteLLM_UserTable(user_id="user-1", model_group_max_budget=GROUP_BUDGET)
    valid_token = UserAPIKeyAuth(token="tok", user_id="user-1")

    with (
        _patch_group_models(),
        patch("litellm.proxy.proxy_server.model_max_budget_limiter", limiter),
    ):
        with pytest.raises(litellm.BudgetExceededError) as exc:
            await _check_user_model_group_budget(
                user_object=user_object,
                valid_token=valid_token,
                model="gpt-4o",
                llm_router=None,
                prisma_client=MagicMock(),
                user_api_key_cache=MagicMock(),
                proxy_logging_obj=MagicMock(),
            )
    assert GROUP_ID in str(exc.value)
    # surfaced onto the token for the post-call logger
    assert valid_token.user_model_group_max_budget == GROUP_BUDGET


@pytest.mark.asyncio
async def test_user_model_group_rate_limit_blocks_when_over_cap():
    limiter = await _rate_seeded_limiter(USER_GROUP_RATE_CACHE_KEY_PREFIX, "user-1")
    user_object = LiteLLM_UserTable(user_id="user-1", model_group_max_budget=GROUP_RPM)
    valid_token = UserAPIKeyAuth(token="tok", user_id="user-1")

    with (
        _patch_group_models(),
        patch("litellm.proxy.proxy_server.model_max_budget_limiter", limiter),
    ):
        with pytest.raises(litellm.RateLimitError) as exc:
            await _check_user_model_group_budget(
                user_object=user_object,
                valid_token=valid_token,
                model="gpt-4o",
                llm_router=None,
                prisma_client=MagicMock(),
                user_api_key_cache=MagicMock(),
                proxy_logging_obj=MagicMock(),
            )
    assert GROUP_ID in str(exc.value)
    assert valid_token.user_model_group_max_budget == GROUP_RPM


@pytest.mark.asyncio
async def test_team_member_model_group_rate_limit_blocks_when_over_cap():
    limiter = await _rate_seeded_limiter(TEAM_MEMBER_GROUP_RATE_CACHE_KEY_PREFIX, "user-1:team-1")
    team_object = LiteLLM_TeamTable(team_id="team-1", team_alias="t")
    valid_token = UserAPIKeyAuth(token="tok", user_id="user-1", team_id="team-1")

    membership = LiteLLM_TeamMembership(
        user_id="user-1",
        team_id="team-1",
        litellm_budget_table=LiteLLM_BudgetTable(model_group_max_budget=GROUP_RPM),
    )

    with (
        _patch_group_models(),
        patch(
            "litellm.proxy.auth.auth_checks.get_team_membership",
            new_callable=AsyncMock,
            return_value=membership,
        ),
        patch("litellm.proxy.proxy_server.model_max_budget_limiter", limiter),
    ):
        with pytest.raises(litellm.RateLimitError):
            await _check_team_member_model_group_budget(
                team_object=team_object,
                valid_token=valid_token,
                model="gpt-4o",
                llm_router=None,
                prisma_client=MagicMock(),
                user_api_key_cache=MagicMock(),
                proxy_logging_obj=MagicMock(),
            )
    assert valid_token.team_member_model_group_max_budget == GROUP_RPM


@pytest.mark.asyncio
async def test_user_model_group_budget_passes_for_model_outside_group():
    limiter = await _seeded_limiter(USER_GROUP_SPEND_CACHE_KEY_PREFIX, "user-1", cost=0.02)
    user_object = LiteLLM_UserTable(user_id="user-1", model_group_max_budget=GROUP_BUDGET)
    valid_token = UserAPIKeyAuth(token="tok", user_id="user-1")

    # gpt-4o-mini is not in GROUP_MODELS -> _can_object_call_model raises -> no match
    with (
        _patch_group_models(),
        patch("litellm.proxy.proxy_server.model_max_budget_limiter", limiter),
    ):
        await _check_user_model_group_budget(
            user_object=user_object,
            valid_token=valid_token,
            model="gpt-4o-mini",
            llm_router=None,
            prisma_client=MagicMock(),
            user_api_key_cache=MagicMock(),
            proxy_logging_obj=MagicMock(),
        )


@pytest.mark.asyncio
async def test_user_model_group_budget_noop_without_config():
    user_object = LiteLLM_UserTable(user_id="user-1", model_group_max_budget={})
    valid_token = UserAPIKeyAuth(token="tok", user_id="user-1")
    failing_limiter = MagicMock()
    failing_limiter.is_user_within_model_group_budget = AsyncMock(side_effect=AssertionError("should not enforce"))
    with patch("litellm.proxy.proxy_server.model_max_budget_limiter", failing_limiter):
        await _check_user_model_group_budget(
            user_object=user_object,
            valid_token=valid_token,
            model="gpt-4o",
            llm_router=None,
            prisma_client=MagicMock(),
            user_api_key_cache=MagicMock(),
            proxy_logging_obj=MagicMock(),
        )


@pytest.mark.asyncio
async def test_team_member_group_budget_per_member_override():
    limiter = await _seeded_limiter("team_member_model_group_spend", "user-1:team-1", cost=0.02)
    team_object = LiteLLM_TeamTable(team_id="team-1", team_alias="t")
    valid_token = UserAPIKeyAuth(token="tok", user_id="user-1", team_id="team-1")

    membership = LiteLLM_TeamMembership(
        user_id="user-1",
        team_id="team-1",
        litellm_budget_table=LiteLLM_BudgetTable(model_group_max_budget=GROUP_BUDGET),
    )

    with (
        _patch_group_models(),
        patch(
            "litellm.proxy.auth.auth_checks.get_team_membership",
            new_callable=AsyncMock,
            return_value=membership,
        ),
        patch("litellm.proxy.proxy_server.model_max_budget_limiter", limiter),
    ):
        with pytest.raises(litellm.BudgetExceededError) as exc:
            await _check_team_member_model_group_budget(
                team_object=team_object,
                valid_token=valid_token,
                model="gpt-4o",
                llm_router=None,
                prisma_client=MagicMock(),
                user_api_key_cache=MagicMock(),
                proxy_logging_obj=MagicMock(),
            )
    assert "Team=team-1" in str(exc.value)


@pytest.mark.asyncio
async def test_team_member_group_budget_falls_back_to_team_default():
    limiter = await _seeded_limiter("team_member_model_group_spend", "user-1:team-1", cost=0.02)
    team_object = LiteLLM_TeamTable(
        team_id="team-1",
        team_alias="t",
        metadata={"team_member_budget_id": "budget-123"},
    )
    valid_token = UserAPIKeyAuth(token="tok", user_id="user-1", team_id="team-1")

    # membership has no per-member group budget -> team default applies
    membership = LiteLLM_TeamMembership(
        user_id="user-1",
        team_id="team-1",
        litellm_budget_table=LiteLLM_BudgetTable(max_budget=5.0),
    )
    default_budget = LiteLLM_BudgetTable(model_group_max_budget=GROUP_BUDGET)

    with (
        _patch_group_models(),
        patch(
            "litellm.proxy.auth.auth_checks.get_team_membership",
            new_callable=AsyncMock,
            return_value=membership,
        ),
        patch(
            "litellm.proxy.auth.auth_checks.get_team_member_default_budget",
            new_callable=AsyncMock,
            return_value=default_budget,
        ),
        patch("litellm.proxy.proxy_server.model_max_budget_limiter", limiter),
    ):
        with pytest.raises(litellm.BudgetExceededError):
            await _check_team_member_model_group_budget(
                team_object=team_object,
                valid_token=valid_token,
                model="gpt-4o",
                llm_router=None,
                prisma_client=MagicMock(),
                user_api_key_cache=MagicMock(),
                proxy_logging_obj=MagicMock(),
            )
    assert valid_token.team_member_model_group_max_budget == GROUP_BUDGET
