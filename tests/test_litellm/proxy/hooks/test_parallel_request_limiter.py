"""
Unit Tests for the max parallel request limiter v1 for the proxy
"""

from datetime import datetime

import pytest

from litellm.caching.caching import DualCache
from litellm.proxy._types import UserAPIKeyAuth
from litellm.proxy.hooks.parallel_request_limiter import (
    _PROXY_MaxParallelRequestsHandler,
)
from litellm.proxy.utils import InternalUsageCache, hash_token
from litellm.types.utils import EmbeddingResponse, ModelResponse, TextCompletionResponse, Usage


@pytest.mark.parametrize(
    "response_obj",
    [
        EmbeddingResponse(
            model="text-embedding-3-small",
            usage=Usage(prompt_tokens=50, completion_tokens=0, total_tokens=50),
        ),
        TextCompletionResponse(
            model="gpt-3.5-turbo-instruct",
            usage=Usage(prompt_tokens=20, completion_tokens=30, total_tokens=50),
        ),
    ],
)
@pytest.mark.asyncio
async def test_async_log_success_event_counts_non_chat_response_tokens(response_obj):
    """
    Embedding and text completion responses must increment the per key, user,
    team, and end user TPM counters, not just chat completion ModelResponse
    objects.
    """
    _api_key = hash_token("sk-12345")
    user_id = "ishaan"
    team_id = "litellm-team"
    end_user_id = "customer-1"

    parallel_request_handler = _PROXY_MaxParallelRequestsHandler(
        internal_usage_cache=InternalUsageCache(DualCache())
    )

    current_date = datetime.now().strftime("%Y-%m-%d")
    current_hour = datetime.now().strftime("%H")
    current_minute = datetime.now().strftime("%M")
    precise_minute = f"{current_date}-{current_hour}-{current_minute}"

    scope_ids = [_api_key, user_id, team_id, end_user_id]
    for scope_id in scope_ids:
        await parallel_request_handler.internal_usage_cache.async_set_cache(
            key=f"{scope_id}::{precise_minute}::request_count",
            value={"current_requests": 1, "current_tpm": 0, "current_rpm": 1},
            litellm_parent_otel_span=None,
        )

    kwargs = {
        "litellm_params": {
            "metadata": {
                "user_api_key": _api_key,
                "user_api_key_user_id": user_id,
                "user_api_key_team_id": team_id,
                "user_api_key_model_max_budget": {},
            }
        },
        "user": end_user_id,
    }

    await parallel_request_handler.async_log_success_event(
        kwargs=kwargs,
        response_obj=response_obj,
        start_time=datetime.now(),
        end_time=datetime.now(),
    )

    for scope_id in scope_ids:
        current = await parallel_request_handler.internal_usage_cache.async_get_cache(
            key=f"{scope_id}::{precise_minute}::request_count",
            litellm_parent_otel_span=None,
        )
        assert current["current_tpm"] == 50, (
            f"expected 50 tokens counted for {scope_id}, "
            f"got {current['current_tpm']}"
        )


@pytest.mark.asyncio
async def test_model_specific_limits_keep_remaining_fields_out_of_provider_metadata():
    cache = DualCache()
    handler = _PROXY_MaxParallelRequestsHandler(internal_usage_cache=InternalUsageCache(cache))
    model = "restricted-model"
    provider_metadata = {"customer_id": "cust-123"}
    data = {
        "model": model,
        "messages": [{"role": "user", "content": "hello"}],
        "metadata": provider_metadata,
        "litellm_metadata": {"model_group": model},
    }
    user_api_key_dict = UserAPIKeyAuth(
        api_key=hash_token("sk-model-specific-metadata"),
        metadata={
            "model_tpm_limit": {model: 100},
            "model_rpm_limit": {model: 10},
        },
    )

    await handler.async_pre_call_hook(
        user_api_key_dict=user_api_key_dict,
        cache=cache,
        data=data,
        call_type="aresponses",
    )

    assert data["metadata"] == provider_metadata
    assert data["litellm_metadata"][f"litellm-key-remaining-tokens-{model}"] == 100
    assert data["litellm_metadata"][f"litellm-key-remaining-requests-{model}"] == 9
    assert f"litellm-key-remaining-tokens-{model}" not in data["metadata"]
    assert f"litellm-key-remaining-requests-{model}" not in data["metadata"]


@pytest.mark.asyncio
async def test_callbacks_read_nested_litellm_metadata_for_success_and_failure():
    cache = DualCache()
    handler = _PROXY_MaxParallelRequestsHandler(internal_usage_cache=InternalUsageCache(cache))
    api_key = hash_token("sk-nested-limiting-metadata")
    model = "restricted-model"
    now = datetime.now()
    precise_minute = now.strftime("%Y-%m-%d-%H-%M")
    key = f"{api_key}::{precise_minute}::request_count"
    model_key = f"{api_key}::{model}::{precise_minute}::request_count"
    provider_metadata = {"customer_id": "cust-123"}
    internal_metadata = {
        "model_group": model,
        "user_api_key": api_key,
        "user_api_key_model_max_budget": {},
        "user_api_key_metadata": {
            "model_tpm_limit": {model: 100},
            "model_rpm_limit": {model: 10},
        },
    }
    kwargs = {
        "litellm_params": {
            "metadata": provider_metadata,
            "litellm_metadata": internal_metadata,
        },
    }

    await handler.async_log_success_event(
        kwargs=kwargs,
        response_obj=ModelResponse(
            usage=Usage(prompt_tokens=4, completion_tokens=3, total_tokens=7),
        ),
        start_time=now,
        end_time=now,
    )

    current = await cache.async_get_cache(key=key, litellm_parent_otel_span=None)
    assert current["current_tpm"] == 7
    assert current["current_requests"] == 0
    model_current = await cache.async_get_cache(key=model_key, litellm_parent_otel_span=None)
    assert model_current["current_tpm"] == 7
    assert model_current["current_requests"] == 0
    assert provider_metadata == {"customer_id": "cust-123"}

    await cache.async_set_cache(
        key=key,
        value={"current_requests": 1, "current_tpm": 7, "current_rpm": 1},
        litellm_parent_otel_span=None,
    )
    failure_kwargs = {
        "litellm_params": {
            "metadata": provider_metadata,
            "litellm_metadata": internal_metadata,
        },
        "exception": Exception("upstream failure"),
    }

    await handler.async_log_failure_event(
        kwargs=failure_kwargs,
        response_obj=None,
        start_time=now,
        end_time=now,
    )

    current = await cache.async_get_cache(key=key, litellm_parent_otel_span=None)
    assert current["current_requests"] == 0
