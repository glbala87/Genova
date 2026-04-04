"""Tests for Batch G and remaining untested modules.

Covers: ModelCard, BiasAuditor, DataQualityReporter, model export,
embedding caches, quantization, local registry, BPE tokenizer,
ensemble uncertainty, and conformal prediction.
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Tiny helper model used across many test groups
# ---------------------------------------------------------------------------

class _TinyModel(nn.Module):
    """Minimal model for testing."""

    def __init__(self, in_features: int = 16, out_features: int = 4):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)

    def forward(self, x):
        return self.linear(x)


class _TinyClassifier(nn.Module):
    """Minimal classifier that accepts (N, L) long tensors."""

    def __init__(self, vocab_size: int = 32, d_model: int = 8, n_classes: int = 3):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, d_model)
        self.head = nn.Linear(d_model, n_classes)

    def forward(self, x):
        e = self.emb(x).mean(dim=1)
        return self.head(e)


# ===================================================================
# 1. Model Card Tests
# ===================================================================

class TestModelCard:

    def test_from_model_creates_card(self):
        from genova.utils.model_card import ModelCard
        model = _TinyModel()
        card = ModelCard.from_model(model, metrics={"auroc": 0.95})
        assert card.architecture["total_parameters"] > 0
        assert card.metrics["auroc"] == 0.95

    def test_to_markdown_returns_string(self):
        from genova.utils.model_card import ModelCard
        model = _TinyModel()
        card = ModelCard.from_model(model)
        md = card.to_markdown()
        assert isinstance(md, str)
        assert len(md) > 0
        assert md.startswith("# Model Card")

    def test_markdown_contains_key_info(self):
        from genova.utils.model_card import ModelCard
        model = _TinyModel()
        card = ModelCard.from_model(model, model_name="TestGenova")
        md = card.to_markdown()
        assert "TestGenova" in md
        assert "Architecture" in md
        assert "Total Parameters" in md

    def test_save_writes_file(self, tmp_path):
        from genova.utils.model_card import ModelCard
        model = _TinyModel()
        card = ModelCard.from_model(model)
        out = tmp_path / "card.md"
        card.save(out)
        assert out.exists()
        content = out.read_text()
        assert "# Model Card" in content


# ===================================================================
# 2. Bias Audit Tests
# ===================================================================

class TestBiasAudit:

    @staticmethod
    def _score_fn(model, sequences):
        """Dummy score function returning random-ish scores."""
        return np.array([len(s) * 0.1 for s in sequences])

    def test_instantiation(self):
        from genova.evaluation.bias_audit import BiasAuditor
        auditor = BiasAuditor(score_fn=self._score_fn)
        assert auditor is not None

    def test_audit_gc_bias(self):
        from genova.evaluation.bias_audit import BiasAuditor
        auditor = BiasAuditor(score_fn=self._score_fn)
        seqs = ["GGCCGGCC", "AATTAATT", "GCGCATAT", "CCCCGGGG", "AAAATTTT"]
        report = auditor.audit_gc_bias(None, seqs)
        assert report.gc_bias
        # At least one bin should have samples
        has_samples = any(
            v.get("n_samples", 0) > 0
            for v in report.gc_bias.values()
            if isinstance(v, dict)
        )
        assert has_samples

    def test_audit_population_bias(self):
        from genova.evaluation.bias_audit import BiasAuditor
        auditor = BiasAuditor(score_fn=self._score_fn)
        data = {
            "EUR": ["ACGTACGT", "GCGCGCGC"],
            "AFR": ["ATATATATAT", "TTTTAAAA"],
            "EAS": ["CCCCGGGG"],
        }
        report = auditor.audit_population_bias(None, data)
        assert len(report.population_bias) == 3
        assert "EUR" in report.population_bias

    def test_generate_report_markdown(self):
        from genova.evaluation.bias_audit import BiasAuditor
        auditor = BiasAuditor(score_fn=self._score_fn)
        auditor.audit_gc_bias(None, ["ACGT", "GGCC"])
        report = auditor.generate_report()
        md = report.to_markdown()
        assert isinstance(md, str)
        assert "Bias Audit Report" in md


# ===================================================================
# 3. Data Quality Tests
# ===================================================================

class TestDataQuality:

    @staticmethod
    def _write_fasta(path: Path, sequences: dict):
        with open(path, "w") as fh:
            for name, seq in sequences.items():
                fh.write(f">{name}\n{seq}\n")

    def test_analyze_fasta(self, tmp_path):
        from genova.data.quality_report import DataQualityReporter
        fa = tmp_path / "test.fa"
        self._write_fasta(fa, {"chr1": "ACGTACGTNN", "chr2": "GGCCGGCC"})
        reporter = DataQualityReporter()
        report = reporter.analyze_fasta(fa)
        assert report.num_sequences == 2
        assert report.total_bases == 18

    def test_gc_content_stats(self, tmp_path):
        from genova.data.quality_report import DataQualityReporter
        fa = tmp_path / "gc.fa"
        self._write_fasta(fa, {"seq1": "GGCCGGCC"})  # 100% GC
        reporter = DataQualityReporter()
        report = reporter.analyze_fasta(fa)
        assert report.gc_content["overall_gc"] == pytest.approx(1.0)

    def test_n_content(self, tmp_path):
        from genova.data.quality_report import DataQualityReporter
        fa = tmp_path / "n.fa"
        self._write_fasta(fa, {"seq1": "ACGTNNNNN"})
        reporter = DataQualityReporter()
        report = reporter.analyze_fasta(fa)
        assert report.n_content["total_n"] == 5

    def test_per_chromosome_breakdown(self, tmp_path):
        from genova.data.quality_report import DataQualityReporter
        fa = tmp_path / "chr.fa"
        self._write_fasta(fa, {"chr1": "ACGTACGT", "chr2": "GGCCGGCC", "chr3": "AAAA"})
        reporter = DataQualityReporter()
        report = reporter.analyze_fasta(fa)
        assert "chr1" in report.per_chromosome
        assert "chr2" in report.per_chromosome
        assert "chr3" in report.per_chromosome
        assert report.per_chromosome["chr3"]["length"] == 4


# ===================================================================
# 4. Model Export Tests
# ===================================================================

class TestModelExport:

    def test_export_torchscript_creates_file(self, tmp_path):
        from genova.models.export import export_torchscript
        model = _TinyModel()
        sample = torch.randn(1, 16)
        out = tmp_path / "model.pt"
        info = export_torchscript(model, sample, out)
        assert out.exists()
        assert info["file_size_bytes"] > 0

    def test_exported_model_same_output(self, tmp_path):
        from genova.models.export import export_torchscript
        model = _TinyModel()
        model.eval()
        sample = torch.randn(2, 16)
        out = tmp_path / "model.pt"
        export_torchscript(model, sample, out)
        loaded = torch.jit.load(str(out))
        with torch.no_grad():
            orig = model(sample)
            exported = loaded(sample)
        assert torch.allclose(orig, exported, atol=1e-5)

    def test_compare_model_sizes(self, tmp_path):
        from genova.models.export import export_torchscript, compare_model_sizes
        model = _TinyModel()
        sample = torch.randn(1, 16)
        ts_path = tmp_path / "model.pt"
        export_torchscript(model, sample, ts_path)
        sizes = compare_model_sizes(model, {"torchscript": ts_path})
        assert "pytorch_memory_mb" in sizes
        assert "torchscript_file_mb" in sizes

    def test_export_onnx_or_torchscript(self, tmp_path):
        """Test ONNX export if available, otherwise fall back to TorchScript."""
        model = _TinyModel()
        sample = torch.randn(1, 16)

        try:
            import onnx  # noqa: F401
            import onnxruntime  # noqa: F401
            _has_onnx = True
        except ImportError:
            _has_onnx = False

        if _has_onnx:
            from genova.models.export import export_onnx
            out = tmp_path / "model.onnx"
            info = export_onnx(model, sample, out)
            assert out.exists()
            assert info["file_size_bytes"] > 0
        else:
            # Fall back to TorchScript export which always works
            from genova.models.export import export_torchscript
            out = tmp_path / "model_fallback.pt"
            info = export_torchscript(model, sample, out)
            assert out.exists()
            assert info["file_size_bytes"] > 0
            # Verify the exported model produces correct output
            loaded = torch.jit.load(str(out))
            model.eval()
            with torch.no_grad():
                orig = model(sample)
                exported = loaded(sample)
            assert torch.allclose(orig, exported, atol=1e-5)


# ===================================================================
# 5. Embedding Cache Tests
# ===================================================================

class TestMemoryCache:

    def test_put_then_get(self):
        from genova.utils.cache import MemoryCache
        cache = MemoryCache(max_size=10)
        emb = np.array([1.0, 2.0, 3.0])
        cache.put("ACGT", emb)
        result = cache.get("ACGT")
        assert result is not None
        np.testing.assert_array_equal(result, emb)

    def test_get_missing_returns_none(self):
        from genova.utils.cache import MemoryCache
        cache = MemoryCache(max_size=10)
        assert cache.get("MISSING") is None

    def test_lru_eviction(self):
        from genova.utils.cache import MemoryCache
        cache = MemoryCache(max_size=2)
        cache.put("SEQ1", np.array([1.0]))
        cache.put("SEQ2", np.array([2.0]))
        cache.put("SEQ3", np.array([3.0]))  # should evict SEQ1
        assert cache.get("SEQ1") is None
        assert cache.get("SEQ2") is not None
        assert cache.get("SEQ3") is not None

    def test_stats_tracks_hits_and_misses(self):
        from genova.utils.cache import MemoryCache
        cache = MemoryCache(max_size=10)
        cache.put("A", np.array([1.0]))
        cache.get("A")      # hit
        cache.get("B")      # miss
        cache.get("B")      # miss
        stats = cache.stats()
        assert stats["hits"] == 1
        assert stats["misses"] == 2


class TestDiskCache:

    def test_put_then_get_roundtrip(self, tmp_path):
        from genova.utils.cache import DiskCache
        db = tmp_path / "cache.db"
        cache = DiskCache(db_path=db, max_size=0)
        emb = np.array([10.0, 20.0, 30.0])
        cache.put("GCGC", emb)
        result = cache.get("GCGC")
        assert result is not None
        np.testing.assert_array_almost_equal(result, emb)
        cache.close()

    def test_persists_across_instances(self, tmp_path):
        from genova.utils.cache import DiskCache
        db = tmp_path / "persist.db"
        c1 = DiskCache(db_path=db)
        c1.put("SEQ", np.array([42.0]))
        c1.close()

        c2 = DiskCache(db_path=db)
        result = c2.get("SEQ")
        assert result is not None
        assert float(result[0]) == pytest.approx(42.0)
        c2.close()


# ===================================================================
# 6. Quantization Tests
# ===================================================================

def _has_quantization_engine():
    """Check if a quantization engine (fbgemm or qnnpack) is available."""
    import copy
    for engine in ("fbgemm", "qnnpack"):
        if engine in torch.backends.quantized.supported_engines:
            try:
                torch.backends.quantized.engine = engine
                m = nn.Sequential(nn.Linear(4, 4))
                m_copy = copy.deepcopy(m).cpu().eval()
                torch.quantization.quantize_dynamic(m_copy, {nn.Linear}, dtype=torch.qint8)
                return True
            except (RuntimeError, Exception):
                continue
    return False


_SKIP_QUANT = not _has_quantization_engine()
_QUANT_REASON = "No quantization engine available in this PyTorch build"


class TestQuantization:

    @pytest.mark.skipif(_SKIP_QUANT, reason=_QUANT_REASON)
    def test_quantize_dynamic_smaller(self):
        from genova.models.quantization import quantize_dynamic, _model_size_bytes
        model = nn.Sequential(nn.Linear(64, 128), nn.ReLU(), nn.Linear(128, 64))
        orig_size = _model_size_bytes(model)
        q_model = quantize_dynamic(model)
        q_size = _model_size_bytes(q_model)
        assert q_size < orig_size

    @pytest.mark.skipif(_SKIP_QUANT, reason=_QUANT_REASON)
    def test_quantized_forward_pass(self):
        from genova.models.quantization import quantize_dynamic
        model = nn.Sequential(nn.Linear(16, 32), nn.ReLU(), nn.Linear(32, 4))
        q_model = quantize_dynamic(model)
        x = torch.randn(2, 16)
        out = q_model(x)
        assert out.shape == (2, 4)

    @pytest.mark.skipif(_SKIP_QUANT, reason=_QUANT_REASON)
    def test_compare_model_sizes(self):
        from genova.models.quantization import quantize_dynamic, compare_model_sizes
        model = nn.Sequential(nn.Linear(64, 128), nn.ReLU(), nn.Linear(128, 64))
        q_model = quantize_dynamic(model)
        sizes = compare_model_sizes(model, q_model)
        assert sizes["compression_ratio"] > 1.0
        assert sizes["size_reduction_pct"] > 0

    def test_benchmark_inference(self):
        from genova.models.quantization import benchmark_inference
        model = nn.Sequential(nn.Linear(16, 8))
        model.eval()
        data = {"input_ids": torch.randn(4, 16)}
        # Patch forward to accept kwargs
        orig_fwd = model.forward
        model.forward = lambda **kw: orig_fwd(kw["input_ids"])
        result = benchmark_inference(model, data, num_runs=5, warmup_runs=2)
        assert "mean_latency_ms" in result
        assert result["mean_latency_ms"] > 0


# ===================================================================
# 7. Local Registry Tests
# ===================================================================

class TestLocalRegistry:

    def test_register_model_saves(self, tmp_path):
        from genova.utils.registry import LocalRegistry, register_model
        registry = LocalRegistry(root=tmp_path / "reg")
        model = _TinyModel()
        meta = register_model(
            registry, model, config={"lr": 1e-4},
            metrics={"acc": 0.9}, name="tiny", version="1.0",
        )
        assert meta["name"] == "tiny"
        assert meta["version"] == "1.0"

    def test_load_model_retrieves(self, tmp_path):
        from genova.utils.registry import LocalRegistry, register_model, load_model
        registry = LocalRegistry(root=tmp_path / "reg")
        model = _TinyModel()
        register_model(registry, model, config={}, metrics={}, name="m1", version="1.0")
        loaded = load_model(registry, "m1", "1.0")
        assert loaded["name"] == "m1"
        assert Path(loaded["model_path"]).exists()

    def test_list_models(self, tmp_path):
        from genova.utils.registry import LocalRegistry, register_model, list_models
        registry = LocalRegistry(root=tmp_path / "reg")
        model = _TinyModel()
        register_model(registry, model, config={}, metrics={}, name="a", version="1.0")
        register_model(registry, model, config={}, metrics={}, name="b", version="1.0")
        models = list_models(registry)
        names = {m["name"] for m in models}
        assert "a" in names
        assert "b" in names

    def test_multiple_versions(self, tmp_path):
        from genova.utils.registry import LocalRegistry, register_model, list_models
        registry = LocalRegistry(root=tmp_path / "reg")
        model = _TinyModel()
        register_model(registry, model, config={}, metrics={}, name="m", version="1.0")
        register_model(registry, model, config={}, metrics={}, name="m", version="2.0")
        models = list_models(registry)
        versions = {m["version"] for m in models if m["name"] == "m"}
        assert "1.0" in versions
        assert "2.0" in versions


# ===================================================================
# 8. BPE Tokenizer Tests
# ===================================================================

class TestBPETokenizer:

    @pytest.fixture()
    def trained_tokenizer(self):
        from genova.data.bpe_tokenizer import GenomicBPETokenizer
        tok = GenomicBPETokenizer(add_special_tokens=False)
        corpus = ["ACGTACGTACGT", "NNACGTNN", "GCGCGCGC", "AAAAACCCC"] * 5
        tok.train(corpus, vocab_size=30)
        return tok

    def test_train_builds_vocab(self, trained_tokenizer):
        assert trained_tokenizer.vocab_size > 10  # base (5 nucs + 5 special)
        assert len(trained_tokenizer.merges) > 0

    def test_encode_returns_ints(self, trained_tokenizer):
        ids = trained_tokenizer.encode("ACGTAC")
        assert isinstance(ids, list)
        assert all(isinstance(i, int) for i in ids)

    def test_decode_returns_string(self, trained_tokenizer):
        ids = trained_tokenizer.encode("GCGC")
        seq = trained_tokenizer.decode(ids)
        assert isinstance(seq, str)

    def test_encode_decode_roundtrip(self, trained_tokenizer):
        original = "ACGTACGT"
        ids = trained_tokenizer.encode(original)
        decoded = trained_tokenizer.decode(ids)
        assert decoded == original

    def test_save_load_preserves_vocab(self, trained_tokenizer, tmp_path):
        out = tmp_path / "tok.json"
        trained_tokenizer.save(out)
        from genova.data.bpe_tokenizer import GenomicBPETokenizer
        loaded = GenomicBPETokenizer.load(out)
        assert loaded.vocab_size == trained_tokenizer.vocab_size
        seq = "ACGTNN"
        assert loaded.encode(seq) == trained_tokenizer.encode(seq)

    def test_create_tokenizer_bpe(self):
        from genova.data.tokenizer import create_tokenizer
        tok = create_tokenizer("bpe")
        from genova.data.bpe_tokenizer import GenomicBPETokenizer
        assert isinstance(tok, GenomicBPETokenizer)


# ===================================================================
# 9. Ensemble Tests
# ===================================================================

class TestDeepEnsemble:

    def test_add_multiple_models(self):
        from genova.uncertainty.ensemble import DeepEnsemble
        ens = DeepEnsemble(device="cpu")
        for _ in range(3):
            ens.add_model(_TinyClassifier())
        assert ens.n_models == 3

    def test_predict_with_uncertainty(self):
        from genova.uncertainty.ensemble import DeepEnsemble
        ens = DeepEnsemble(device="cpu", batch_size=4)
        for _ in range(2):
            ens.add_model(_TinyClassifier())
        x = torch.randint(0, 32, (4, 10))
        result = ens.predict_with_uncertainty(x)
        assert "mean" in result
        assert "variance" in result
        assert result["mean"].shape[0] == 4

    def test_variance_non_negative(self):
        from genova.uncertainty.ensemble import DeepEnsemble
        ens = DeepEnsemble(device="cpu")
        for _ in range(3):
            ens.add_model(_TinyClassifier())
        x = torch.randint(0, 32, (8, 10))
        result = ens.predict_with_uncertainty(x)
        assert np.all(result["variance"] >= 0)


class TestSnapshotEnsemble:

    def test_collect_snapshot(self):
        from genova.uncertainty.ensemble import SnapshotEnsemble
        model = _TinyClassifier()
        snap = SnapshotEnsemble(model, device="cpu")
        n = snap.collect_snapshot()
        assert n == 1
        assert snap.n_snapshots == 1

    def test_prediction_uses_all_snapshots(self):
        from genova.uncertainty.ensemble import SnapshotEnsemble
        model = _TinyClassifier()
        snap = SnapshotEnsemble(model, device="cpu", batch_size=4)
        # Collect two snapshots with different weights
        snap.collect_snapshot()
        # Perturb weights for a different snapshot
        with torch.no_grad():
            for p in model.parameters():
                p.add_(torch.randn_like(p) * 0.5)
        snap.collect_snapshot()
        assert snap.n_snapshots == 2
        x = torch.randint(0, 32, (4, 10))
        result = snap.predict_with_uncertainty(x)
        assert "mean" in result
        assert result["mean"].shape[0] == 4


# ===================================================================
# 10. Conformal Prediction Tests
# ===================================================================

class TestConformalPredictor:

    @staticmethod
    def _make_calibration_data(n=200, n_classes=3):
        """Synthetic softmax probs and labels."""
        np.random.seed(42)
        labels = np.random.randint(0, n_classes, size=n)
        # Create probabilities biased toward correct label
        raw = np.random.dirichlet(np.ones(n_classes), size=n)
        for i in range(n):
            raw[i, labels[i]] += 1.0
        probs = raw / raw.sum(axis=1, keepdims=True)
        return probs, labels

    def test_calibrate_sets_threshold(self):
        from genova.uncertainty.conformal import ConformalPredictor
        probs, labels = self._make_calibration_data()
        cp = ConformalPredictor()
        q = cp.calibrate(probs, labels, alpha=0.1)
        assert cp.is_calibrated
        assert q is not None
        assert q >= 0

    def test_predict_set_non_empty(self):
        from genova.uncertainty.conformal import ConformalPredictor
        probs, labels = self._make_calibration_data()
        cp = ConformalPredictor()
        cp.calibrate(probs, labels, alpha=0.1)
        sets = cp.predict_set(probs[:10])
        assert all(len(s) > 0 for s in sets)

    def test_coverage_meets_target(self):
        from genova.uncertainty.conformal import ConformalPredictor
        probs, labels = self._make_calibration_data(n=500)
        cp = ConformalPredictor()
        cp.calibrate(probs[:300], labels[:300], alpha=0.1)
        result = cp.evaluate_coverage(probs[300:], labels[300:])
        # Coverage should be >= 1 - alpha (0.9), allow small slack
        assert result["coverage"] >= 0.85


class TestConformalRegressor:

    def test_predict_interval_returns_bounds(self):
        from genova.uncertainty.conformal import ConformalRegressor
        np.random.seed(42)
        residuals = np.random.randn(100)
        cr = ConformalRegressor()
        cr.calibrate(residuals, alpha=0.1)
        preds = np.linspace(0, 10, 20)
        lower, upper = cr.predict_interval(preds)
        assert len(lower) == 20
        assert len(upper) == 20

    def test_lower_less_than_upper(self):
        from genova.uncertainty.conformal import ConformalRegressor
        np.random.seed(42)
        residuals = np.random.randn(100)
        cr = ConformalRegressor()
        cr.calibrate(residuals, alpha=0.1)
        preds = np.array([1.0, 2.0, 3.0])
        lower, upper = cr.predict_interval(preds)
        assert np.all(lower < upper)

    def test_interval_coverage_meets_target(self):
        from genova.uncertainty.conformal import ConformalRegressor
        np.random.seed(42)
        n = 500
        true_vals = np.random.randn(n) * 2 + 5
        predictions = true_vals + np.random.randn(n) * 0.5
        residuals_cal = true_vals[:300] - predictions[:300]
        cr = ConformalRegressor()
        cr.calibrate(residuals_cal, alpha=0.1)
        result = cr.evaluate_coverage(predictions[300:], true_vals[300:])
        assert result["coverage"] >= 0.85
