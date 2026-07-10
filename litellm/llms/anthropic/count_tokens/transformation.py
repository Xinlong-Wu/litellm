"""
Anthropic CountTokens API transformation logic.

This module handles the transformation of requests to Anthropic's CountTokens API format.
"""

import os
from typing import Any, Dict, List, Optional

from litellm.constants import ANTHROPIC_TOKEN_COUNTING_BETA_VERSION


class AnthropicCountTokensConfig:
    """
    Configuration and transformation logic for Anthropic CountTokens API.

    Anthropic CountTokens API Specification:
    - Endpoint: POST https://api.anthropic.com/v1/messages/count_tokens
    - Beta header required: anthropic-beta: token-counting-2024-11-01
    - Response: {"input_tokens": <number>}
    """

    def get_anthropic_count_tokens_endpoint(self, api_base: Optional[str] = None) -> str:
        """
        Get the Anthropic CountTokens API endpoint.

        Resolution order:
        1. Explicit ``api_base`` (from the deployment's ``litellm_params``)
        2. ``ANTHROPIC_COUNT_TOKENS_API_BASE`` / ``ANTHROPIC_API_BASE`` env var
        3. Anthropic's public endpoint

        A bare base URL (e.g. ``http://host:8317`` or ``.../v1``) is normalized to
        the full ``/v1/messages/count_tokens`` path so an OpenAI-compatible upstream
        proxy can serve the request. A URL that already targets the count_tokens
        endpoint is used as-is.

        Returns:
            The endpoint URL for the CountTokens API
        """
        count_tokens_path = "/v1/messages/count_tokens"
        base = api_base or os.getenv("ANTHROPIC_COUNT_TOKENS_API_BASE") or os.getenv("ANTHROPIC_API_BASE")
        if not base:
            return "https://api.anthropic.com" + count_tokens_path

        base = base.rstrip("/")
        if base.endswith("/messages/count_tokens"):
            return base
        if base.endswith("/v1"):
            return base + "/messages/count_tokens"
        return base + count_tokens_path

    def transform_request_to_count_tokens(
        self,
        model: str,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
        system: Optional[Any] = None,
    ) -> Dict[str, Any]:
        """
        Transform request to Anthropic CountTokens format.

        Includes optional system and tools fields for accurate token counting.
        """
        request: Dict[str, Any] = {
            "model": model,
            "messages": messages,
        }

        if system is not None:
            request["system"] = system

        if tools is not None:
            request["tools"] = tools

        return request

    def get_required_headers(self, api_key: str) -> Dict[str, str]:
        """
        Get the required headers for the CountTokens API.

        Args:
            api_key: The Anthropic API key

        Returns:
            Dictionary of required headers
        """
        from litellm.llms.anthropic.common_utils import (
            optionally_handle_anthropic_oauth,
        )

        headers: Dict[str, str] = {
            "Content-Type": "application/json",
            "x-api-key": api_key,
            "anthropic-version": "2023-06-01",
            "anthropic-beta": ANTHROPIC_TOKEN_COUNTING_BETA_VERSION,
        }
        headers, _ = optionally_handle_anthropic_oauth(headers=headers, api_key=api_key)
        return headers

    def validate_request(self, model: str, messages: List[Dict[str, Any]]) -> None:
        """
        Validate the incoming count tokens request.

        Args:
            model: The model name
            messages: The messages to count tokens for

        Raises:
            ValueError: If the request is invalid
        """
        if not model:
            raise ValueError("model parameter is required")

        if not messages:
            raise ValueError("messages parameter is required")

        if not isinstance(messages, list):
            raise ValueError("messages must be a list")

        for i, message in enumerate(messages):
            if not isinstance(message, dict):
                raise ValueError(f"Message {i} must be a dictionary")

            if "role" not in message:
                raise ValueError(f"Message {i} must have a 'role' field")

            if "content" not in message:
                raise ValueError(f"Message {i} must have a 'content' field")
