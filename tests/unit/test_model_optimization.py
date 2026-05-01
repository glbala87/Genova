"""Tests for model optimization modules: quantization, export, and pruning."""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from genova.models.quantization import (
    _model_size_bytes,
    compare_model_sizes,
    quantize_dynamic,
)
from genova.models.export import export_torchscript
from genova.models.pruning import PruningSchedule


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def simple_model():
    model = nn.Sequential(
        nn.Linear(64, 32),
        nn.ReLU(),
        nn.Linear(32, 10),
    )
    model.eval()
    return model


@pytest.fixture
def simple_input():
    return torch.randn(4, 64)


# ---------------------------------------------------------------------------
# Quantization tests
# ---------------------------------------------------------------------------


class TestModelSizeBytes:

    def test_returns_positive_int(self, simple_model):
        size = _model_size_bytes(simple_model)
        assert isinstance(size, int)
        assert size > 0

    def test_larger_model_has_larger_size(self):
        small = nn.Linear(8, 4)
        large = nn.Linear(256, 128)
        assert _model_size_bytes(large) > _model_size_bytes(small)


class TestQuantizeDynamic:

    def test_returns_module(self, simple_model):
        q_model = quantize_dynamic(simple_model)
        assert isinstance(q_model, nn.Module)

    def test_quantized_model_produces_output(self, simple_model, simple_input):
        q_model = quantize_dynamic(simple_model)
        with torch.no_grad():
            output = q_model(simple_input)
        assert output.shape == (4, 10)

    def test_quantized_size_leq_original(self, simple_model):
        q_model = quantize_dynamic(simple_model)
        orig_size = _model_size_bytes(simple_model)
        quant_size = _model_size_bytes(q_model)
        assert quant_size <= orig_size

    def test_original_model_unchanged(self, simple_model, simple_input):
        with torch.no_grad():
            original_output = simple_model(simple_input).clone()
        _ = quantize_dynamic(simple_model)
        with torch.no_grad():
            after_output = simple_model(simple_input)
        assert torch.allclose(original_output, after_output)

    def test_custom_layer_types(self, simple_model):
        q_model = quantize_dynamic(simple_model, layer_types=[nn.Linear])
        assert isinstance(q_model, nn.Module)


class TestCompareModelSizes:

    def test_returns_expected_keys(self, simple_model):
        q_model = quantize_dynamic(simple_model)
        result = compare_model_sizes(simple_model, q_model)
        expected_keys = {
            "original_size_mb",
            "quantized_size_mb",
            "compression_ratio",
            "size_reduction_pct",
        }
        assert expected_keys == set(result.keys())

    def test_values_are_numeric(self, simple_model):
        q_model = quantize_dynamic(simple_model)
        result = compare_model_sizes(simple_model, q_model)
        for key, value in result.items():
            assert isinstance(value, (int, float))

    def test_compression_ratio_geq_one(self, simple_model):
        q_model = quantize_dynamic(simple_model)
        result = compare_model_sizes(simple_model, q_model)
        assert result["compression_ratio"] >= 1.0


# ---------------------------------------------------------------------------
# TorchScript export tests
# ---------------------------------------------------------------------------


class TestExportTorchScript:

    def test_creates_file(self, simple_model, simple_input, tmp_path):
        output_path = tmp_path / "model.pt"
        info = export_torchscript(simple_model, simple_input, output_path)
        assert output_path.exists()
        assert output_path.stat().st_size > 0

    def test_returns_metadata(self, simple_model, simple_input, tmp_path):
        output_path = tmp_path / "model.pt"
        info = export_torchscript(simple_model, simple_input, output_path)
        assert "file_size_bytes" in info
        assert "export_time_s" in info

    def test_exported_model_loads(self, simple_model, simple_input, tmp_path):
        output_path = tmp_path / "model.pt"
        export_torchscript(simple_model, simple_input, output_path)
        loaded = torch.jit.load(str(output_path))
        assert loaded is not None

    def test_exported_model_same_output(self, simple_model, simple_input, tmp_path):
        output_path = tmp_path / "model.pt"
        simple_model.eval()
        with torch.no_grad():
            expected = simple_model(simple_input)

        export_torchscript(simple_model, simple_input, output_path)
        loaded = torch.jit.load(str(output_path))
        loaded.eval()
        with torch.no_grad():
            actual = loaded(simple_input)

        assert torch.allclose(expected, actual, atol=1e-6)


# ---------------------------------------------------------------------------
# Pruning schedule tests
# ---------------------------------------------------------------------------


class TestPruningSchedule:

    def test_initial_step_gives_initial_ratio(self):
        schedule = PruningSchedule(initial_ratio=0.0, final_ratio=0.5, total_steps=1000)
        ratio = schedule.get_ratio(0)
        assert ratio == pytest.approx(0.0, abs=1e-9)

    def test_final_step_gives_final_ratio(self):
        schedule = PruningSchedule(initial_ratio=0.0, final_ratio=0.5, total_steps=1000)
        ratio = schedule.get_ratio(1000)
        assert ratio == pytest.approx(0.5, abs=1e-9)

    def test_beyond_total_steps_gives_final_ratio(self):
        schedule = PruningSchedule(initial_ratio=0.0, final_ratio=0.5, total_steps=1000)
        ratio = schedule.get_ratio(2000)
        assert ratio == pytest.approx(0.5, abs=1e-9)

    def test_monotonically_increasing(self):
        schedule = PruningSchedule(initial_ratio=0.0, final_ratio=0.5, total_steps=100)
        prev = schedule.get_ratio(0)
        for step in range(1, 101):
            current = schedule.get_ratio(step)
            assert current >= prev - 1e-12
            prev = current

    def test_cubic_interpolation_midpoint(self):
        schedule = PruningSchedule(initial_ratio=0.0, final_ratio=1.0, total_steps=100)
        ratio = schedule.get_ratio(50)
        # ratio = final - (final - initial) * (1 - t/total)^3 = 1.0 - 0.125 = 0.875
        assert ratio == pytest.approx(0.875, abs=1e-9)

    def test_nonzero_initial_ratio(self):
        schedule = PruningSchedule(initial_ratio=0.2, final_ratio=0.8, total_steps=100)
        assert schedule.get_ratio(0) == pytest.approx(0.2, abs=1e-9)
        assert schedule.get_ratio(100) == pytest.approx(0.8, abs=1e-9)

    def test_all_ratios_within_bounds(self):
        schedule = PruningSchedule(initial_ratio=0.1, final_ratio=0.9, total_steps=200)
        for step in range(0, 201):
            ratio = schedule.get_ratio(step)
            assert 0.1 - 1e-9 <= ratio <= 0.9 + 1e-9
