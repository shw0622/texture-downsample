"""空间下采样滤波器：``box_downsample`` 与 ``dpid_downsample``。

所有函数都假设输入是 numpy float 数组，形状 ``(H, W, C)``，且 ``H``、``W``
能被 ``factor`` 整除。``factor`` 必须是 ≥ 2 的整数（不要求 2 的幂，
但实践中只用 2/4/8）。

历史 API ``box_downsample_2x`` / ``dpid_downsample_2x`` 仍保留，
等价于 ``box_downsample(..., factor=2)`` / ``dpid_downsample(..., factor=2)``，
以保证旧调用方零回归。
"""
from __future__ import annotations

import numpy as np
from scipy.ndimage import uniform_filter

__all__ = [
    "box_downsample",
    "dpid_downsample",
    "box_downsample_2x",   # 向后兼容
    "dpid_downsample_2x",  # 向后兼容
]


def box_downsample(x: np.ndarray, factor: int = 2) -> np.ndarray:
    """逐 ``factor × factor`` 块求平均下采样。

    Parameters
    ----------
    x : ndarray, shape ``(H, W, C)``
        输入张量；``H``、``W`` 必须能被 ``factor`` 整除。
    factor : int, 默认 2
        块边长（同时是下采倍率）。

    Returns
    -------
    ndarray, shape ``(H/factor, W/factor, C)``
        与 ``x`` 同 dtype。
    """
    if x.ndim != 3:
        raise ValueError(
            f"box_downsample: 期望 3D 输入 (H, W, C)，实际 ndim={x.ndim}"
        )
    if not isinstance(factor, int) or factor < 2:
        raise ValueError(f"box_downsample: factor 必须是 >=2 的整数，实际 {factor}")
    H, W, C = x.shape
    if H % factor != 0 or W % factor != 0:
        raise ValueError(
            f"box_downsample: H、W 必须能被 factor={factor} 整除，实际 H={H}, W={W}"
        )
    f = factor
    return x.reshape(H // f, f, W // f, f, C).mean(axis=(1, 3))


def dpid_downsample(
    img: np.ndarray,
    lam: float = 1.0,
    support: int = 4,
    factor: int = 2,
) -> np.ndarray:
    """DPID（Detail-Preserving Image Downscaling）``factor×`` 下采样。

    与局部均值距离越远的像素在下采样时权重越大，可显著保留细节。

    Parameters
    ----------
    img : ndarray, shape ``(H, W, C)``, float
        输入图像；``H``、``W`` 必须能被 ``factor`` 整除。
    lam : float
        锐化强度。常规材质 0.8-1.0；含细线/印花 1.5-2.0；纯渐变 0.5。
    support : int
        局部均值窗口大小（在原分辨率上，与 ``factor`` 无关）。
    factor : int, 默认 2
        下采倍率。

    Returns
    -------
    ndarray, shape ``(H/factor, W/factor, C)``
    """
    if img.ndim != 3:
        raise ValueError("dpid_downsample: 期望 3D 输入 (H, W, C)")
    if not isinstance(factor, int) or factor < 2:
        raise ValueError(f"dpid_downsample: factor 必须是 >=2 的整数，实际 {factor}")
    H, W, C = img.shape
    if H % factor != 0 or W % factor != 0:
        raise ValueError(
            f"dpid_downsample: H、W 必须能被 factor={factor} 整除，实际 H={H}, W={W}"
        )

    # 局部均值（在原分辨率上，逐通道求）
    mu = np.stack(
        [
            uniform_filter(img[..., c], size=support, mode="reflect")
            for c in range(C)
        ],
        axis=-1,
    )

    # 与局部均值的距离作为权重；epsilon 防止全零块除 0
    dist = np.linalg.norm(img - mu, axis=-1, keepdims=True)
    weights = (dist ** lam) + 1e-8                        # (H, W, 1)

    # 加权下采样：分子分母分别做 box 平均，再相除。
    weighted = img * weights                              # (H, W, C)
    num = box_downsample(weighted, factor=factor)         # (H/f, W/f, C)
    den = box_downsample(weights, factor=factor)          # (H/f, W/f, 1)
    return num / np.maximum(den, 1e-12)


# ---------------------------------------------------------------------------
# 向后兼容别名（旧调用方零回归）
# ---------------------------------------------------------------------------


def box_downsample_2x(x: np.ndarray) -> np.ndarray:
    """``box_downsample(x, factor=2)`` 的向后兼容别名。"""
    return box_downsample(x, factor=2)


def dpid_downsample_2x(
    img: np.ndarray, lam: float = 1.0, support: int = 4
) -> np.ndarray:
    """``dpid_downsample(img, lam, support, factor=2)`` 的向后兼容别名。"""
    return dpid_downsample(img, lam=lam, support=support, factor=2)
