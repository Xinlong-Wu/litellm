"""
Unit tests for per-model-group dollar budgets in
``_PROXY_VirtualKeyModelMaxBudgetLimiter``.

A "model group" here is an access group: a named set of models whose combined
dollar spend is capped by one budget. These tests pin three behaviors that
would silently regress if spend were tracked per-model or per-key instead of
per-group/per-scope:

  * group aggregation - spend on every model in the group accrues to one cap
  * user aggregation across keys - the counter is keyed on user_id, so two
    different virtual keys for the same user share one budget
  * team-member scope - spend is keyed on (user_id, team_id)
"""

import pytest

from litellm.caching.caching import DualCache
from litellm.exceptions import (
    BudgetExceededError,
    RateLimitError,
    RateLimitType,
)
from litellm.proxy.hooks.model_max_budget_limiter import (
    TEAM_MEMBER_GROUP_RATE_CACHE_KEY_PREFIX,
    USER_GROUP_RATE_CACHE_KEY_PREFIX,
    USER_GROUP_SPEND_CACHE_KEY_PREFIX,
    _PROXY_VirtualKeyModelMaxBudgetLimiter,
)

GROUP_BUDGET = {"premium": {"max_budget": 0.02, "budget_duration": "30d"}}
GROUP_RPM = {"premium": {"rpm_limit": 2}}
GROUP_TPM = {"premium": {"tpm_limit": 100}}


def _make_limiter() -> _PROXY_VirtualKeyModelMaxBudgetLimiter:
    return _PROXY_VirtualKeyModelMaxBudgetLimiter(dual_cache=DualCache())


@pytest.mark.asyncio
async def test_user_within_group_budget_passes_when_under_cap():
    limiter = _make_limiter()
    await limiter._increment_model_group_spend(
        scope_prefix=USER_GROUP_SPEND_CACHE_KEY_PREFIX,
        scope_id="user-1",
        model_group_max_budget=GROUP_BUDGET,
        groups=["premium"],
        response_cost=0.005,
    )

    assert (
        await limiter.is_user_within_model_group_budget(
            user_id="user-1",
            user_model_group_max_budget=GROUP_BUDGET,
            groups=["premium"],
        )
        is True
    )


@pytest.mark.asyncio
async def test_user_group_budget_sums_across_models_in_group():
    """Two different models in the same group accrue to one cap."""
    limiter = _make_limiter()
    # model A request
    await limiter._increment_model_group_spend(
        scope_prefix=USER_GROUP_SPEND_CACHE_KEY_PREFIX,
        scope_id="user-1",
        model_group_max_budget=GROUP_BUDGET,
        groups=["premium"],
        response_cost=0.012,
    )
    # model B request (also resolves to the "premium" group)
    await limiter._increment_model_group_spend(
        scope_prefix=USER_GROUP_SPEND_CACHE_KEY_PREFIX,
        scope_id="user-1",
        model_group_max_budget=GROUP_BUDGET,
        groups=["premium"],
        response_cost=0.012,
    )

    with pytest.raises(BudgetExceededError):
        await limiter.is_user_within_model_group_budget(
            user_id="user-1",
            user_model_group_max_budget=GROUP_BUDGET,
            groups=["premium"],
        )


@pytest.mark.asyncio
async def test_user_group_budget_aggregates_across_keys():
    """Spend from two virtual keys for the same user shares one counter.

    The increment helper is key-agnostic - it is scoped by user_id - so the two
    requests below stand in for two different virtual keys. A per-key
    implementation would leave each key at 0.012 (under 0.02) and never trip.
    """
    limiter = _make_limiter()
    for _ in range(2):
        await limiter._increment_model_group_spend(
            scope_prefix=USER_GROUP_SPEND_CACHE_KEY_PREFIX,
            scope_id="user-1",
            model_group_max_budget=GROUP_BUDGET,
            groups=["premium"],
            response_cost=0.012,
        )

    with pytest.raises(BudgetExceededError) as exc:
        await limiter.is_user_within_model_group_budget(
            user_id="user-1",
            user_model_group_max_budget=GROUP_BUDGET,
            groups=["premium"],
        )
    assert exc.value.current_cost == pytest.approx(0.024)


@pytest.mark.asyncio
async def test_user_group_budget_isolated_per_user():
    limiter = _make_limiter()
    await limiter._increment_model_group_spend(
        scope_prefix=USER_GROUP_SPEND_CACHE_KEY_PREFIX,
        scope_id="user-1",
        model_group_max_budget=GROUP_BUDGET,
        groups=["premium"],
        response_cost=0.05,
    )

    # user-2 has spent nothing, so they are unaffected by user-1's overage
    assert (
        await limiter.is_user_within_model_group_budget(
            user_id="user-2",
            user_model_group_max_budget=GROUP_BUDGET,
            groups=["premium"],
        )
        is True
    )


@pytest.mark.asyncio
async def test_group_with_no_budget_is_not_enforced():
    limiter = _make_limiter()
    # requested model resolves to a group that has no configured budget
    assert (
        await limiter.is_user_within_model_group_budget(
            user_id="user-1",
            user_model_group_max_budget=GROUP_BUDGET,
            groups=["unbudgeted-group"],
        )
        is True
    )


@pytest.mark.asyncio
async def test_team_member_group_budget_keyed_on_user_and_team():
    limiter = _make_limiter()
    await limiter._increment_model_group_spend(
        scope_prefix="team_member_model_group_spend",
        scope_id="user-1:team-1",
        model_group_max_budget=GROUP_BUDGET,
        groups=["premium"],
        response_cost=0.03,
    )

    with pytest.raises(BudgetExceededError) as exc:
        await limiter.is_team_member_within_model_group_budget(
            user_id="user-1",
            team_id="team-1",
            team_member_model_group_max_budget=GROUP_BUDGET,
            groups=["premium"],
        )
    assert "Team=team-1" in str(exc.value)

    # same user in a different team is unaffected
    assert (
        await limiter.is_team_member_within_model_group_budget(
            user_id="user-1",
            team_id="team-2",
            team_member_model_group_max_budget=GROUP_BUDGET,
            groups=["premium"],
        )
        is True
    )


@pytest.mark.asyncio
async def test_user_group_rpm_limit_blocks_after_cap():
    """rpm_limit rides in the same model_group_max_budget map and caps requests/min."""
    limiter = _make_limiter()
    # one request under the cap of 2 still passes
    await limiter._increment_model_group_rate(
        scope_prefix=USER_GROUP_RATE_CACHE_KEY_PREFIX,
        scope_id="user-1",
        model_group_max_budget=GROUP_RPM,
        groups=["premium"],
        total_tokens=10,
    )
    assert (
        await limiter.is_user_within_model_group_budget(
            user_id="user-1",
            user_model_group_max_budget=GROUP_RPM,
            groups=["premium"],
        )
        is True
    )

    # a second request hits the cap
    await limiter._increment_model_group_rate(
        scope_prefix=USER_GROUP_RATE_CACHE_KEY_PREFIX,
        scope_id="user-1",
        model_group_max_budget=GROUP_RPM,
        groups=["premium"],
        total_tokens=10,
    )
    with pytest.raises(RateLimitError) as exc:
        await limiter.is_user_within_model_group_budget(
            user_id="user-1",
            user_model_group_max_budget=GROUP_RPM,
            groups=["premium"],
        )
    assert exc.value.rate_limit_type == RateLimitType.REQUESTS.value
    assert "model group=premium" in str(exc.value)


@pytest.mark.asyncio
async def test_user_group_tpm_limit_blocks_when_tokens_exceed():
    """tpm_limit caps summed tokens/min across the group, not request count."""
    limiter = _make_limiter()
    # a single request of 100 tokens reaches the tpm cap of 100
    await limiter._increment_model_group_rate(
        scope_prefix=USER_GROUP_RATE_CACHE_KEY_PREFIX,
        scope_id="user-1",
        model_group_max_budget=GROUP_TPM,
        groups=["premium"],
        total_tokens=100,
    )
    with pytest.raises(RateLimitError) as exc:
        await limiter.is_user_within_model_group_budget(
            user_id="user-1",
            user_model_group_max_budget=GROUP_TPM,
            groups=["premium"],
        )
    assert exc.value.rate_limit_type == RateLimitType.TOKENS.value

    # a different user with no token spend is unaffected
    assert (
        await limiter.is_user_within_model_group_budget(
            user_id="user-2",
            user_model_group_max_budget=GROUP_TPM,
            groups=["premium"],
        )
        is True
    )


@pytest.mark.asyncio
async def test_rate_only_group_enforced_without_max_budget():
    """A group carrying only a rate limit (no max_budget) is still enforced.

    Guards the ``_group_rate_config`` relaxation: ``_group_budget_config`` requires
    ``max_budget > 0``, so if rate enforcement reused it, rate-only groups would
    silently never trip.
    """
    limiter = _make_limiter()
    for _ in range(2):
        await limiter._increment_model_group_rate(
            scope_prefix=USER_GROUP_RATE_CACHE_KEY_PREFIX,
            scope_id="user-1",
            model_group_max_budget=GROUP_RPM,
            groups=["premium"],
            total_tokens=1,
        )

    with pytest.raises(RateLimitError):
        await limiter.is_user_within_model_group_budget(
            user_id="user-1",
            user_model_group_max_budget=GROUP_RPM,
            groups=["premium"],
        )


@pytest.mark.asyncio
async def test_user_and_team_member_rate_counters_isolated():
    """User-scope and team-member-scope rate counters use distinct cache keys."""
    limiter = _make_limiter()
    for _ in range(2):
        await limiter._increment_model_group_rate(
            scope_prefix=USER_GROUP_RATE_CACHE_KEY_PREFIX,
            scope_id="user-1",
            model_group_max_budget=GROUP_RPM,
            groups=["premium"],
            total_tokens=1,
        )

    # the team-member scope for the same user has its own counter at 0
    assert (
        await limiter.is_team_member_within_model_group_budget(
            user_id="user-1",
            team_id="team-1",
            team_member_model_group_max_budget=GROUP_RPM,
            groups=["premium"],
        )
        is True
    )

    await limiter._increment_model_group_rate(
        scope_prefix=TEAM_MEMBER_GROUP_RATE_CACHE_KEY_PREFIX,
        scope_id="user-1:team-1",
        model_group_max_budget=GROUP_RPM,
        groups=["premium"],
        total_tokens=1,
    )
    await limiter._increment_model_group_rate(
        scope_prefix=TEAM_MEMBER_GROUP_RATE_CACHE_KEY_PREFIX,
        scope_id="user-1:team-1",
        model_group_max_budget=GROUP_RPM,
        groups=["premium"],
        total_tokens=1,
    )
    with pytest.raises(RateLimitError):
        await limiter.is_team_member_within_model_group_budget(
            user_id="user-1",
            team_id="team-1",
            team_member_model_group_max_budget=GROUP_RPM,
            groups=["premium"],
        )
