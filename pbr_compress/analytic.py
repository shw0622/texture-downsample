"""解析与启发式通道下采样：Normal / Roughness(LEAN) / Metallic / AO。

对应 doc.md 第 3.1 ~ 3.4 节。所有函数纯 numpy，不依赖 PyTorch。

所有函数都接受 ``factor`` 参数（默认 2）以支持任意 ``factor × factor`` 块下采。
``factor`` 必须是 ≥ 2 的整数。LEAN 中的方差吸收系数 ``2 * σ²`` 由 GGX
各向同性 Toksvig 推导而来，与 ``factor`` 无关——只要在更大块上正确估计
``σ²`` 即可保持物理一致。
"""
from __future__ import annotations

import numpy as np

from .filters import box_downsample

__all__ = [
    "downsample_normal",
    "downsample_roughness_lean",
    "downsample_metallic",
    "downsample_ao",
]


def downsample_normal(N_hr: np.ndarray, factor: int = 2) -> np.ndarray:
    """法线 ``factor×`` 下采样：``box average`` + 重归一化。

    Parameters
    ----------
    N_hr : ndarray, shape ``(H, W, 3)``
        切空间法线，单位向量，分量在 ``[-1, 1]``。
    factor : int, 默认 2
        下采倍率。

    Returns
    -------
    ndarray, shape ``(H/factor, W/factor, 3)``
        归一化后的低分法线。
    """
    if N_hr.ndim != 3 or N_hr.shape[-1] != 3:
        raise ValueError("downsample_normal: 输入 shape 必须为 (H, W, 3)")
    N_avg = box_downsample(N_hr, factor=factor)
    length = np.linalg.norm(N_avg, axis=-1, keepdims=True)
    return N_avg / np.maximum(length, 1e-8)


def downsample_roughness_lean(
    R_hr: np.ndarray, N_hr: np.ndarray, factor: int = 2
) -> np.ndarray:
    """LEAN 方差补偿的 roughness 下采样。

    把 footprint 内法线方差吸收进粗糙度（基于 GGX 的各向同性 Toksvig 式近似）::

        alpha_hr      = R_hr ** 2                        # GGX 粗糙度参数
        alpha2_avg    = box_downsample(alpha_hr ** 2, factor)
        sigma2_normal = (1 - |N_avg|^2) / 2              # footprint 内损失的方差
        alpha2_lr     = alpha2_avg + 2 * sigma2_normal
        R_lr          = sqrt(sqrt(alpha2_lr))

    ``sigma2_normal`` 必须独立用 ``box_downsample(N_hr, factor)`` 计算，
    以保证逻辑解耦（即便未来修改 Normal 通道的下采样方法也不受影响）。

    Parameters
    ----------
    R_hr : ndarray, shape ``(H, W, 1)``, ``[0, 1]``
    N_hr : ndarray, shape ``(H, W, 3)``, 单位向量
    factor : int, 默认 2
        下采倍率；更大 factor 会让 ``σ²`` 估计更激进（吸收更大块的方差），
        因此对镜面金属（α 极小）仍可能过度模糊——是否启用应由上游路由决定。

    Returns
    -------
    ndarray, shape ``(H/factor, W/factor, 1)``, ``[0, 1]``
    """
    if R_hr.ndim != 3 or R_hr.shape[-1] != 1:
        raise ValueError("downsample_roughness_lean: R_hr 末维必须为 1")
    if N_hr.ndim != 3 or N_hr.shape[-1] != 3:
        raise ValueError("downsample_roughness_lean: N_hr 末维必须为 3")
    if R_hr.shape[:2] != N_hr.shape[:2]:
        raise ValueError(
            f"downsample_roughness_lean: R_hr {R_hr.shape[:2]} 与 "
            f"N_hr {N_hr.shape[:2]} 空间分辨率必须一致"
        )

    alpha_sq_hr = (R_hr ** 2) ** 2                                       # (H, W, 1)
    alpha_sq_avg = box_downsample(alpha_sq_hr, factor=factor)            # (H/f, W/f, 1)

    N_avg = box_downsample(N_hr, factor=factor)                          # (H/f, W/f, 3)
    N_avg_len_sq = np.sum(N_avg ** 2, axis=-1, keepdims=True)
    sigma_sq_normal = np.clip((1.0 - N_avg_len_sq) / 2.0, 0.0, None)

    alpha_sq_lr = alpha_sq_avg + 2.0 * sigma_sq_normal
    R_lr = np.sqrt(np.sqrt(alpha_sq_lr))
    return np.clip(R_lr, 0.0, 1.0)


def downsample_metallic(
    M_hr: np.ndarray, threshold: float = 0.5, factor: int = 2
) -> np.ndarray:
    """Metallic 通道 ``factor×`` 下采样：``box average`` + 二值阈值。

    .. note::
       不要做软插值。金属/非金属的 F0 差异（0.04 vs 0.7-1.0）在物理上是
       跳变的，任何中间值都对应非物理材质。

    Parameters
    ----------
    M_hr : ndarray, shape ``(H, W, 1)``, 近似二值
    threshold : float
        判定为金属的阈值，默认 0.5。
    factor : int, 默认 2
        下采倍率。

    Returns
    -------
    ndarray, shape ``(H/factor, W/factor, 1)``, ``{0, 1}``，dtype 与 ``M_hr`` 相同
    """
    if M_hr.ndim != 3 or M_hr.shape[-1] != 1:
        raise ValueError("downsample_metallic: M_hr 末维必须为 1")
    M_avg = box_downsample(M_hr, factor=factor)
    return (M_avg >= threshold).astype(M_hr.dtype)


def downsample_ao(AO_hr: np.ndarray, factor: int = 2) -> np.ndarray:
    """AO 通道 ``factor×`` 下采样：直接 box average。"""
    if AO_hr.ndim != 3 or AO_hr.shape[-1] != 1:
        raise ValueError("downsample_ao: AO_hr 末维必须为 1")
    return box_downsample(AO_hr, factor=factor)
