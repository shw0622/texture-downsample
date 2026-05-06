"""io.py 单元测试（颜色空间编解码 + roundtrip）。"""
import numpy as np
import pytest

from pbr_compress.io import (
    decode_normal,
    encode_normal,
    linear_to_srgb,
    load_albedo,
    load_normal,
    load_scalar,
    save_albedo,
    save_normal,
    save_scalar,
    srgb_to_linear,
)


def test_srgb_linear_roundtrip():
    rng = np.random.default_rng(0)
    x = rng.random((32, 32, 3)).astype(np.float32)
    y = linear_to_srgb(srgb_to_linear(x))
    assert np.allclose(x, y, atol=2e-3)


def test_srgb_known_anchor_points():
    # 0、1 必须保持
    x = np.array([0.0, 1.0], dtype=np.float32)
    assert np.allclose(srgb_to_linear(x), [0.0, 1.0], atol=1e-6)
    assert np.allclose(linear_to_srgb(x), [0.0, 1.0], atol=1e-6)
    # 0.5 sRGB -> 约 0.214 linear（标准曲线）
    assert np.isclose(srgb_to_linear(np.array([0.5]))[0], 0.21404114, atol=1e-4)


def test_normal_encode_decode_roundtrip():
    rng = np.random.default_rng(1)
    n = rng.standard_normal((4, 4, 3)).astype(np.float32)
    n = n / np.linalg.norm(n, axis=-1, keepdims=True)
    enc = encode_normal(n)
    assert (enc >= 0).all() and (enc <= 1).all()
    dec = decode_normal(enc)
    assert np.allclose(dec, n, atol=1e-3)


def test_albedo_tif_roundtrip(tmp_path):
    """16-bit TIFF 走 tifffile 后端，可保留 albedo 精度。"""
    rng = np.random.default_rng(2)
    A = (rng.random((8, 8, 3)) * 0.8 + 0.1).astype(np.float32)
    p = tmp_path / "albedo.tif"
    save_albedo(p, A, srgb=True, bit_depth=16)
    A2 = load_albedo(p, srgb=True)
    assert A2.shape == A.shape
    assert np.allclose(A, A2, atol=2e-3)


def test_albedo_png_8bit_roundtrip(tmp_path):
    """8-bit sRGB PNG 是最常用的 albedo 工作流，量化误差需可接受。"""
    rng = np.random.default_rng(20)
    A = (rng.random((16, 16, 3)) * 0.8 + 0.1).astype(np.float32)
    p = tmp_path / "albedo.png"
    save_albedo(p, A, srgb=True, bit_depth=8)
    A2 = load_albedo(p, srgb=True)
    # sRGB 在暗部量化较粗，留 8e-3 容忍
    assert np.allclose(A, A2, atol=8e-3)


def test_normal_tif_roundtrip(tmp_path):
    rng = np.random.default_rng(3)
    n = rng.standard_normal((8, 8, 3)).astype(np.float32)
    n = n / np.linalg.norm(n, axis=-1, keepdims=True)
    p = tmp_path / "normal.tif"
    save_normal(p, n, bit_depth=16)
    n2 = load_normal(p)
    assert np.allclose(n, n2, atol=5e-3)


def test_normal_png_8bit_roundtrip(tmp_path):
    rng = np.random.default_rng(30)
    n = rng.standard_normal((8, 8, 3)).astype(np.float32)
    n = n / np.linalg.norm(n, axis=-1, keepdims=True)
    p = tmp_path / "normal.png"
    save_normal(p, n, bit_depth=8)
    n2 = load_normal(p)
    # 8-bit 量化导致最大约 1/127 误差
    assert np.allclose(n, n2, atol=2e-2)


def test_scalar_png_roundtrip(tmp_path):
    rng = np.random.default_rng(4)
    R = rng.random((8, 8, 1)).astype(np.float32)
    p = tmp_path / "rough.png"
    save_scalar(p, R, bit_depth=16)
    R2 = load_scalar(p)
    assert R2.shape == R.shape
    assert np.allclose(R, R2, atol=2e-4)
