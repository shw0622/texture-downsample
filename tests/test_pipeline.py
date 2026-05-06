"""pipeline.py 集成测试。

测试都跑在 CPU 上，避免 GPU 资源争用与 CI 不可用。
"""
import numpy as np
import pytest

from pbr_compress.metrics import evaluate_render_quality
from pbr_compress.pipeline import compress_pbr_textures


def make_synthetic(H=16, W=16, seed=0):
    rng = np.random.default_rng(seed)
    A = (rng.random((H, W, 3), dtype=np.float64) * 0.8 + 0.1).astype(np.float32)

    n = rng.standard_normal((H, W, 3)).astype(np.float32)
    n[..., 2] = np.abs(n[..., 2]) + 0.5
    n = n / np.linalg.norm(n, axis=-1, keepdims=True)
    N = n

    R = (rng.random((H, W, 1), dtype=np.float64) * 0.7 + 0.1).astype(np.float32)
    M = (rng.random((H, W, 1)) > 0.7).astype(np.float32)
    AO = (rng.random((H, W, 1), dtype=np.float64) * 0.5 + 0.5).astype(np.float32)

    return A, N, R, M, AO


def test_pipeline_shapes_and_ranges():
    A, N, R, M, AO = make_synthetic(16, 16)
    A_lr, N_lr, R_lr, M_lr, AO_lr = compress_pbr_textures(
        A, N, R, M, AO,
        n_iter=20, batch_views=8, device="cpu", seed=0, verbose=False,
    )
    assert A_lr.shape == (8, 8, 3)
    assert N_lr.shape == (8, 8, 3)
    assert R_lr.shape == (8, 8, 1)
    assert M_lr.shape == (8, 8, 1)
    assert AO_lr.shape == (8, 8, 1)

    # 法线归一化
    lengths = np.linalg.norm(N_lr, axis=-1)
    assert np.allclose(lengths, 1.0, atol=1e-5)

    # 数值范围
    assert (A_lr >= 0).all() and (A_lr <= 1).all()
    assert (R_lr >= 0.04 - 1e-6).all() and (R_lr <= 1.0 + 1e-6).all()
    assert set(np.unique(M_lr).tolist()) <= {0.0, 1.0}
    assert (AO_lr >= 0).all() and (AO_lr <= 1).all()


def test_pipeline_repeatable_with_seed():
    A, N, R, M, AO = make_synthetic(16, 16, seed=1)
    out1 = compress_pbr_textures(
        A, N, R, M, AO,
        n_iter=10, batch_views=4, device="cpu", seed=42, verbose=False,
    )
    out2 = compress_pbr_textures(
        A, N, R, M, AO,
        n_iter=10, batch_views=4, device="cpu", seed=42, verbose=False,
    )
    for a, b in zip(out1, out2):
        assert np.allclose(a, b)


def test_pipeline_metrics_reasonable():
    A, N, R, M, AO = make_synthetic(32, 32, seed=2)
    A_lr, N_lr, R_lr, M_lr, AO_lr = compress_pbr_textures(
        A, N, R, M, AO,
        n_iter=80, batch_views=16, device="cpu", seed=0, verbose=False,
    )
    m = evaluate_render_quality(
        A, N, R, M, AO,
        A_lr, N_lr, R_lr, M_lr, AO_lr,
        n_pairs=16, device="cpu", seed=0, use_flip=False,
    )
    assert m["render_l1"] < 0.5
    assert m["render_psnr_ldr"] > 12.0


def test_pipeline_rejects_odd_resolution():
    A = np.zeros((5, 4, 3), dtype=np.float32)
    N = np.zeros((5, 4, 3), dtype=np.float32); N[..., 2] = 1.0
    R = np.zeros((5, 4, 1), dtype=np.float32)
    M = np.zeros((5, 4, 1), dtype=np.float32)
    AO = np.zeros((5, 4, 1), dtype=np.float32)
    with pytest.raises(ValueError):
        compress_pbr_textures(
            A, N, R, M, AO,
            n_iter=1, batch_views=2, device="cpu", verbose=False,
        )


def test_pipeline_rejects_mismatched_resolution():
    A, N, R, M, AO = make_synthetic(16, 16)
    AO_bad = np.zeros((8, 8, 1), dtype=np.float32)
    with pytest.raises(ValueError):
        compress_pbr_textures(
            A, N, R, M, AO_bad,
            n_iter=1, batch_views=2, device="cpu", verbose=False,
        )
