from __future__ import annotations

import os

import pytest


OFFLINE_TEST_COHERE_KEY = "test-key-for-offline-tests"


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption(
        "--run-live",
        action="store_true",
        default=False,
        help="Run live Cohere/index integration tests marked with @pytest.mark.live.",
    )
    parser.addoption(
        "--run-slow-offline",
        action="store_true",
        default=False,
        help="Run local but expensive tests marked with @pytest.mark.slow_offline.",
    )


def pytest_configure(config: pytest.Config) -> None:
    if not config.getoption("--run-live"):
        os.environ.setdefault("COHERE_API_KEY", OFFLINE_TEST_COHERE_KEY)


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    skip_live = pytest.mark.skip(reason="live Cohere/index test; run with --run-live")
    skip_slow_offline = pytest.mark.skip(reason="slow local corpus/UI test; run with --run-slow-offline")
    for item in items:
        if "live" in item.keywords and not config.getoption("--run-live"):
            item.add_marker(skip_live)
        if "slow_offline" in item.keywords and not config.getoption("--run-slow-offline"):
            item.add_marker(skip_slow_offline)
