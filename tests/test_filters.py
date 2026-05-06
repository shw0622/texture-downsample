"""filters.py 单元测试。"""
import numpy as np
import pytest

from pbr_compress.filters import box_downsample_2x, dpid_downsample_2x


def test_box_downsample_constant():
    x = np.full((4, 4, 3), 0.7, dtype=np.float32)
    y = box_downsample_2x(x)
    assert y.shape == (2, 2, 3)
    assert np.allclose(y, 0.7)


def test_box_downsample_average():
    x = np.zeros((2, 2, 1), dtype=np.float32)
    x[0, 0, 0] = 1.0  # 单像素 1.0，2x2 平均 = 0.25
    y = box_downsample_2x(x)
    assert y.shape == (1, 1, 1)
    assert np.isclose(y[0, 0, 0], 0.25)


def test_box_downsample_against_strided_mean():
    """与显式 stride 平均一致。"""
    rng = np.random.default_rng(0)
    x = rng.random((8, 6, 3)).astype(np.float32)
    y = box_downsample_2x(x)
    expected = (
        x[0::2, 0::2] + x[0::2, 1::2] + x[1::2, 0::2] + x[1::2, 1::2]
    ) / 4.0
    assert np.allclose(y, expected)


def test_box_downsample_odd_raises():
    with pytest.raises(ValueError):
        box_downsample_2x(np.zeros((3, 4, 1)))


def test_box_downsample_2d_raises():
    with pytest.raises(ValueError):
        box_downsample_2x(np.zeros((4, 4)))


def test_dpid_constant_equals_box():
    x = np.full((8, 8, 3), 0.4, dtype=np.float32)
    y = dpid_downsample_2x(x)
    assert np.allclose(y, 0.4, atol=1e-6)


def test_dpid_lambda_increases_detail():
    """加大 ``lam`` 时锐利特征保留更多（与 box average 的差距更大）。"""
    rng = np.random.default_rng(0)
    x = rng.random((16, 16, 3)).astype(np.float32)
    box = box_downsample_2x(x)
    y_low = dpid_downsample_2x(x, lam=0.1)
    y_high = dpid_downsample_2x(x, lam=2.0)
    assert np.abs(y_high - box).mean() > np.abs(y_low - box).mean()


def test_dpid_shape():
    rng = np.random.default_rng(1)
    x = rng.random((10, 8, 4)).astype(np.float32)
    y = dpid_downsample_2x(x)
    assert y.shape == (5, 4, 4)
