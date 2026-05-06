"""4 个 baseline + 主方法 + 路由化主方法，用于消融对比。

所有 baseline 接口签名一致：

    method(A, N, R, M, AO) -> (A_lr, N_lr, R_lr, M_lr, AO_lr)

便于在外层脚本里写一个 ``methods`` 字典统一循环。

| key                  | 方案                                                   |
|----------------------|--------------------------------------------------------|
| ``all_box``          | A1：所有 5 通道 box，normal 加重归一化                 |
| ``lanczos``          | B1：PIL Lanczos-3，normal 加重归一化                   |
| ``box_lean``         | A2：A1 + Roughness 用 LEAN                             |
| ``box_lean_dpid``    | A3：A2 + Albedo 用 DPID（**不优化**）                  |
| ``ours``             | A4：A2 + 自适应起点 (box/DPID) + Adam 优化 + 早停      |
| ``ours_routed``      | A5：A4 + 材质路由 (auto_route=True)，工程化「不输」版  |
"""
from __future__ import annotations

from typing import Tuple

import numpy as np
from PIL import Image

from .analytic import (
    downsample_ao,
    downsample_metallic,
    downsample_normal,
    downsample_roughness_lean,
)
from .filters import box_downsample_2x, dpid_downsample_2x
from .pipeline import (
    ROUGHNESS_LOWER_BOUND,
    _adaptive_roughness_floor,
    compress_pbr_textures,
)

__all__ = [
    "baseline_all_box",
    "baseline_lanczos",
    "baseline_box_lean",
    "baseline_box_lean_dpid",
    "baseline_ours",
    "baseline_ours_routed",
    "BASELINES",
    "apply_levels",
]


PBRTuple = Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]


def apply_levels(method, A, N, R, M, AO, levels: int = 1, **kwargs) -> PBRTuple:
    """把任意 ``method(A,N,R,M,AO,**kwargs) -> PBRTuple`` 链式执行 ``levels`` 次。

    每次的输出是下一次的输入，等价于 mip 链生成的天然口径：
    - ``levels=1`` → 1× 调用 → 总下采样 2×
    - ``levels=2`` → 2× 调用 → 总下采样 4×
    - ``levels=3`` → 3× 调用 → 总下采样 8×

    与「直接做一次 4×/8× 下采样」的差异在于：
    每一级 LEAN/DPID 都会重新基于上一级 (R, N) 估计 footprint 内方差，
    更接近 GPU 运行时 mip 采样的链式行为。
    """
    if levels < 1:
        raise ValueError(f"apply_levels: levels 必须 ≥ 1，实际 {levels}")
    cur = (A, N, R, M, AO)
    for _ in range(levels):
        cur = method(*cur, **kwargs)
    return cur


# ---------------------------------------------------------------------------
# 内部工具
# ---------------------------------------------------------------------------


def _lanczos_2x(x: np.ndarray) -> np.ndarray:
    """逐通道 PIL Lanczos-3 下采样到 1/2，保持 float32 精度。"""
    H, W = x.shape[:2]
    new_w, new_h = W // 2, H // 2
    out = np.empty((new_h, new_w, x.shape[-1]), dtype=np.float32)
    for c in range(x.shape[-1]):
        ch = x[..., c].astype(np.float32)
        im = Image.fromarray(ch, mode="F")
        im_lr = im.resize((new_w, new_h), Image.LANCZOS)
        out[..., c] = np.asarray(im_lr, dtype=np.float32)
    return out


def _renormalize(N: np.ndarray) -> np.ndarray:
    return N / np.maximum(np.linalg.norm(N, axis=-1, keepdims=True), 1e-8)


# ---------------------------------------------------------------------------
# Baseline 实现
# ---------------------------------------------------------------------------


def baseline_all_box(A, N, R, M, AO) -> PBRTuple:
    """A1：所有 5 通道 box 平均；normal 重归一化保持单位长度。"""
    A_lr = box_downsample_2x(A)
    N_lr = downsample_normal(N)            # = box + renormalize
    R_lr = box_downsample_2x(R)
    M_lr = box_downsample_2x(M)
    AO_lr = box_downsample_2x(AO)
    return A_lr, N_lr, R_lr, M_lr, AO_lr


def baseline_lanczos(A, N, R, M, AO) -> PBRTuple:
    """B1：所有 5 通道走 PIL Lanczos-3；normal 重归一化。

    这是「业界 mip 默认」做法：高质量低通滤波器，但不感知 BRDF 的非线性。
    """
    A_lr = np.clip(_lanczos_2x(A), 0.0, 1.0)
    N_lr = _renormalize(_lanczos_2x(N))
    R_lr = np.clip(_lanczos_2x(R), 0.0, 1.0)
    M_lr = np.clip(_lanczos_2x(M), 0.0, 1.0)
    AO_lr = np.clip(_lanczos_2x(AO), 0.0, 1.0)
    return A_lr, N_lr, R_lr, M_lr, AO_lr


def baseline_box_lean(
    A, N, R, M, AO,
    metallic_threshold: float = 0.5,
    roughness_floor: float | None = None,
) -> PBRTuple:
    """A2：A1 基础上把 Roughness 换成 LEAN，Metallic 用阈值。

    ``roughness_floor=None`` → 自适应 ``max(0.02, R.min())``，与主管线一致。
    """
    A_lr = box_downsample_2x(A)
    N_lr = downsample_normal(N)
    floor = _adaptive_roughness_floor(R) if roughness_floor is None else roughness_floor
    R_lr = np.clip(downsample_roughness_lean(R, N), floor, 1.0)
    M_lr = downsample_metallic(M, threshold=metallic_threshold)
    AO_lr = downsample_ao(AO)
    return A_lr, N_lr, R_lr, M_lr, AO_lr


def baseline_box_lean_dpid(
    A, N, R, M, AO,
    dpid_lam: float = 1.0,
    dpid_support: int = 4,
    metallic_threshold: float = 0.5,
    roughness_floor: float | None = None,
) -> PBRTuple:
    """A3：A2 基础上把 Albedo 换成 DPID（**不做可微优化**）。

    用于隔离评估「DPID 初始化」相比 box 平均给 Albedo 带来多少提升。
    ``roughness_floor=None`` → 自适应，与主管线一致。
    """
    A_lr = np.clip(
        dpid_downsample_2x(A, lam=dpid_lam, support=dpid_support), 0.0, 1.0
    )
    N_lr = downsample_normal(N)
    floor = _adaptive_roughness_floor(R) if roughness_floor is None else roughness_floor
    R_lr = np.clip(downsample_roughness_lean(R, N), floor, 1.0)
    M_lr = downsample_metallic(M, threshold=metallic_threshold)
    AO_lr = downsample_ao(AO)
    return A_lr, N_lr, R_lr, M_lr, AO_lr


def baseline_ours(A, N, R, M, AO, **kwargs) -> PBRTuple:
    """A4：本项目完整管线。

    与 A3 (`box_lean_dpid`) 的差异：
    1. **自适应起点**：normal 几何方差小则用 box（A1），反之用 DPID（A3）。
    2. **Adam 可微优化**：在 holdout 视角上做 best-snapshot 早停，
       若所有步都不如起点则回退到起点（不会反向劣化）。

    .. note::
       baseline 模式下**强制** ``auto_route=False`` 以保证消融完整：
       此处希望测的是「完整 LEAN+DPID+Opt 路径」的纯方法效果，
       而不是「auto 路由后落到 all-box」的工程化版本。
       生产/落地请直接调 ``compress_pbr_textures(...)``，享受路由收益。
    """
    kwargs.setdefault("auto_route", False)
    return compress_pbr_textures(A, N, R, M, AO, **kwargs)


def baseline_ours_routed(A, N, R, M, AO, **kwargs) -> PBRTuple:
    """A5：本项目工程化版（启用 auto_route）。

    与 ``baseline_ours`` (A4) 的差异：``auto_route=True``，按材质类型路由：
    - GEOMETRIC 纹理：完整 LEAN+DPID+Adam（与 A4 等价）
    - CONSTANT / METAL_FLAT 纹理：直接 all-box，跳过 LEAN/Opt

    这是「全材质支持」的工程化形态：在能赢的材质上赢，在赢不了的材质上不输。
    """
    kwargs.setdefault("auto_route", True)
    return compress_pbr_textures(A, N, R, M, AO, **kwargs)


# 名字 -> 函数；外部脚本可直接 ``for name, fn in BASELINES.items(): ...``
BASELINES = {
    "all_box":         baseline_all_box,
    "lanczos":         baseline_lanczos,
    "box_lean":        baseline_box_lean,
    "box_lean_dpid":   baseline_box_lean_dpid,
    "ours":            baseline_ours,
    "ours_routed":     baseline_ours_routed,
}
