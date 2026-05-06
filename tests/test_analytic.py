"""analytic.py 单元测试。"""
import numpy as np
import pytest

from pbr_compress.analytic import (
    downsample_ao,
    downsample_metallic,
    downsample_normal,
    downsample_roughness_lean,
)


# ------------------------- normal -------------------------


def test_normal_unit_length():
    rng = np.random.default_rng(0)
    n = rng.standard_normal((8, 8, 3)).astype(np.float32)
    n = n / np.linalg.norm(n, axis=-1, keepdims=True)
    out = downsample_normal(n)
    assert out.shape == (4, 4, 3)
    assert np.allclose(np.linalg.norm(out, axis=-1), 1.0, atol=1e-5)


def test_normal_smooth_returns_input():
    n = np.zeros((4, 4, 3), dtype=np.float32)
    n[..., 2] = 1.0  # 全部指向 +z
    out = downsample_normal(n)
    expected = np.zeros((2, 2, 3), dtype=np.float32)
    expected[..., 2] = 1.0
    assert np.allclose(out, expected, atol=1e-6)


def test_normal_opposing_directions():
    """对法线 +z 与 -z 平均后归一化必须仍是单位长度（不能 NaN）。"""
    n = np.zeros((2, 2, 3), dtype=np.float32)
    n[0, 0, 2] = 1.0
    n[0, 1, 2] = 1.0
    n[1, 0, 2] = 1.0
    n[1, 1, 0] = 1.0  # 一个像素指向 +x，破坏对称
    out = downsample_normal(n)
    assert np.all(np.isfinite(out))
    assert np.isclose(np.linalg.norm(out[0, 0]), 1.0, atol=1e-5)


# ------------------------- roughness (LEAN) -------------------------


def test_roughness_lean_smooth_normal_is_box_in_alpha2():
    """法线无方差时，LEAN 退化为 alpha² 域的 box average。"""
    R = np.array([[0.1, 0.3], [0.5, 0.7]], dtype=np.float64).reshape(2, 2, 1)
    n = np.zeros((2, 2, 3), dtype=np.float64)
    n[..., 2] = 1.0
    R_lr = downsample_roughness_lean(R, n)
    expected = ((0.1 ** 4 + 0.3 ** 4 + 0.5 ** 4 + 0.7 ** 4) / 4.0) ** 0.25
    assert R_lr.shape == (1, 1, 1)
    assert np.isclose(R_lr[0, 0, 0], expected, atol=1e-6)


def test_roughness_lean_increases_with_normal_variance():
    """法线方差越大，输出 R 应该越大。"""
    R = np.full((4, 4, 1), 0.2, dtype=np.float64)

    n_smooth = np.zeros((4, 4, 3), dtype=np.float64)
    n_smooth[..., 2] = 1.0
    R_smooth = downsample_roughness_lean(R, n_smooth)

    # 在 footprint 内交替方向 → 高法线方差
    n_rough = np.zeros((4, 4, 3), dtype=np.float64)
    n_rough[..., 2] = np.sqrt(1.0 - 0.49)
    n_rough[0::2, 0::2, 0] = 0.7
    n_rough[1::2, 1::2, 0] = -0.7
    R_rough = downsample_roughness_lean(R, n_rough)

    assert R_rough.mean() > R_smooth.mean() + 0.01


def test_roughness_lean_clamped_to_unit():
    """合成 alpha² 不会让 R 超过 1。"""
    R = np.full((2, 2, 1), 1.0, dtype=np.float64)
    n = np.zeros((2, 2, 3), dtype=np.float64)
    n[0, 0, 0] = 1.0
    n[0, 1, 1] = 1.0
    n[1, 0, 0] = -1.0
    n[1, 1, 1] = -1.0
    R_lr = downsample_roughness_lean(R, n)
    assert R_lr.max() <= 1.0


# ------------------------- metallic -------------------------


def test_metallic_threshold_majority():
    M = np.array([[1.0, 1.0], [1.0, 0.0]], dtype=np.float32).reshape(2, 2, 1)
    out = downsample_metallic(M, threshold=0.5)
    assert out.shape == (1, 1, 1)
    assert out[0, 0, 0] == 1.0  # avg=0.75 ≥ 0.5


def test_metallic_threshold_minority():
    M = np.array([[1.0, 0.0], [0.0, 0.0]], dtype=np.float32).reshape(2, 2, 1)
    out = downsample_metallic(M, threshold=0.5)
    assert out[0, 0, 0] == 0.0  # avg=0.25 < 0.5


def test_metallic_dtype_preserved():
    M = np.zeros((2, 2, 1), dtype=np.float32)
    out = downsample_metallic(M)
    assert out.dtype == np.float32


# ------------------------- AO -------------------------


def test_ao_box():
    AO = np.full((4, 4, 1), 0.5, dtype=np.float32)
    out = downsample_ao(AO)
    assert out.shape == (2, 2, 1)
    assert np.allclose(out, 0.5)
