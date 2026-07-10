"""Regression tests for the configurable MCP gateway identity.

``LITELLM_MCP_SERVER_NAME`` and ``LITELLM_MCP_SERVER_DESCRIPTION`` are read from
the environment at import time in
``litellm.proxy._experimental.mcp_server.utils`` and must flow through to every
consumer, including the well-known registry entry built in
``mcp_management_endpoints``. The env values are reloaded into the modules and
restored afterwards so the override does not leak into other tests.
"""

import contextlib
import importlib
import os

import pytest

pytest.importorskip("mcp")

UTILS_MODULE = "litellm.proxy._experimental.mcp_server.utils"
MGMT_MODULE = "litellm.proxy.management_endpoints.mcp_management_endpoints"


@contextlib.contextmanager
def _env_and_reload(**env):
    saved = {key: os.environ.get(key) for key in env}

    # Snapshot the modules' attributes so they can be restored to their ORIGINAL
    # class objects afterwards. Reloading to "undo" would mint brand-new classes
    # (e.g. MCPMissingUserEnvVarsError) that diverge from the references frozen
    # at import time by consumers such as mcp_server_manager, which then raises a
    # class that sibling tests' ``pytest.raises`` (resolving the current one) no
    # longer match — a cross-test failure on the same xdist worker.
    utils_module = importlib.import_module(UTILS_MODULE)
    mgmt_module = importlib.import_module(MGMT_MODULE)
    saved_utils = dict(utils_module.__dict__)
    saved_mgmt = dict(mgmt_module.__dict__)

    def _apply_env(values):
        for key, value in values.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value

    def _restore_module(module, snapshot):
        module.__dict__.clear()
        module.__dict__.update(snapshot)

    try:
        _apply_env(env)
        utils = importlib.reload(utils_module)
        mgmt = importlib.reload(mgmt_module)
        yield utils, mgmt
    finally:
        _apply_env(saved)
        _restore_module(utils_module, saved_utils)
        _restore_module(mgmt_module, saved_mgmt)


def test_defaults_used_when_env_unset():
    with _env_and_reload(LITELLM_MCP_SERVER_NAME=None, LITELLM_MCP_SERVER_DESCRIPTION=None) as (utils, _mgmt):
        assert utils.LITELLM_MCP_SERVER_NAME == "litellm-mcp-server"
        assert utils.LITELLM_MCP_SERVER_DESCRIPTION == "MCP Server for LiteLLM"


def test_env_overrides_server_identity():
    with _env_and_reload(
        LITELLM_MCP_SERVER_NAME="acme-gateway",
        LITELLM_MCP_SERVER_DESCRIPTION="Acme internal MCP gateway",
    ) as (utils, _mgmt):
        assert utils.LITELLM_MCP_SERVER_NAME == "acme-gateway"
        assert utils.LITELLM_MCP_SERVER_DESCRIPTION == "Acme internal MCP gateway"


def test_env_override_propagates_to_registry_entry():
    with _env_and_reload(
        LITELLM_MCP_SERVER_NAME="acme-gateway",
        LITELLM_MCP_SERVER_DESCRIPTION="Acme internal MCP gateway",
    ) as (_utils, mgmt):
        entry = mgmt._build_builtin_registry_entry("http://localhost:4000")

    assert entry["name"] == "acme-gateway"
    assert entry["title"] == "acme-gateway"
    assert entry["description"] == "Acme internal MCP gateway"
