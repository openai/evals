import os
from unittest.mock import patch

import pytest

from evals.solvers.providers.minimax.minimax_solver import (
    MINIMAX_CHAT_MODELS,
    MiniMaxSolver,
    is_chat_model,
)


class TestIsChatModel:
    def test_m3_is_chat_model(self):
        assert is_chat_model("MiniMax-M3") is True

    def test_m27_is_chat_model(self):
        assert is_chat_model("MiniMax-M2.7") is True

    def test_m27_highspeed_is_chat_model(self):
        assert is_chat_model("MiniMax-M2.7-highspeed") is True

    def test_unknown_model_raises(self):
        with pytest.raises(NotImplementedError, match="not currently supported"):
            is_chat_model("unknown-model")

    def test_all_models_are_chat_models(self):
        for model in MINIMAX_CHAT_MODELS:
            assert is_chat_model(model) is True


class TestMiniMaxSolverProperties:
    @pytest.fixture
    def solver(self):
        return MiniMaxSolver(
            completion_fn_options={
                "model": "MiniMax-M3",
                "extra_options": {"temperature": 1, "max_tokens": 512},
            },
        )

    @pytest.fixture
    def solver_m27(self):
        return MiniMaxSolver(
            completion_fn_options={
                "model": "MiniMax-M2.7",
                "extra_options": {"temperature": 1, "max_tokens": 512},
            },
        )

    @pytest.fixture
    def solver_highspeed(self):
        return MiniMaxSolver(
            completion_fn_options={
                "model": "MiniMax-M2.7-highspeed",
                "extra_options": {"temperature": 1, "max_tokens": 512},
            },
        )

    def test_default_api_base(self, solver):
        assert solver._api_base == "https://api.minimax.io/v1"

    def test_custom_api_base(self, solver):
        with patch.dict(os.environ, {"MINIMAX_BASE_URL": "https://custom.api.io/v1"}):
            assert solver._api_base == "https://custom.api.io/v1"

    def test_api_key_from_env(self, solver):
        with patch.dict(os.environ, {"MINIMAX_API_KEY": "test-key-123"}):
            assert solver._api_key == "test-key-123"

    def test_api_key_none_when_unset(self, solver):
        with patch.dict(os.environ, {}, clear=True):
            assert solver._api_key is None

    def test_model_name(self, solver):
        assert solver.model == "MiniMax-M3"

    def test_m27_model_name(self, solver_m27):
        assert solver_m27.model == "MiniMax-M2.7"

    def test_highspeed_model_name(self, solver_highspeed):
        assert solver_highspeed.model == "MiniMax-M2.7-highspeed"

    def test_valid_answers_raises(self):
        with pytest.raises(NotImplementedError, match="valid_answers"):
            MiniMaxSolver(
                completion_fn_options={
                    "model": "MiniMax-M3",
                    "extra_options": {"temperature": 1},
                },
                valid_answers=["yes", "no"],
            )

    def test_prechecks_returns_none(self, solver):
        msgs = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Hello!"},
        ]
        assert solver._perform_prechecks(msgs) is None

    def test_preprocess_completion_fn_options_does_nothing(self, solver):
        assert solver._preprocess_completion_fn_options() is None
