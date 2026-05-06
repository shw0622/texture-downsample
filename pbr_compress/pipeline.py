"""顶层管线：把 5 个通道一起 2× 下采样。

对应 doc.md 第 5 节。处理顺序严格按 doc.md 第 2 节::

    Step 1: Normal           ->  N_lr
    Step 2: Roughness (LEAN) ->  R_lr      (依赖 N_hr)
    Step 3: Metallic         ->  M_lr
    Step 4: AO               ->  AO_lr
    Step 5: Albedo (优化)    ->  A_lr      (依赖 N_lr / R_lr / M_lr / AO_lr)

材质自适应路由（v2）:
    examples/verify_box_ceiling.py 在 5 张真实纹理上的实测显示，
    LEAN+DPID+Adam 完整管线只在「normal 真有几何变化」的纹理上才能
    显著优于 box；在 normal 近常数的纹理上 box 已是天花板，复杂管线
    反而引入噪声。因此 ``compress_pbr_textures`` 默认按下面三类路由::

        | 类型         | 判据                                  | 路径                  |
        | CONSTANT     | normal_var<5e-4 且 roughness_std<2e-2 | all-box，零优化       |
        | METAL_FLAT   | normal_var<5e-4 但 R/A 有变化          | all-box，跳过 LEAN/Opt|
        | GEOMETRIC    | normal_var≥5e-4                       | 完整 LEAN+DPID+Opt    |

    传 ``auto_route=False`` 强制走完整管线（向后兼容旧脚本）。
"""
from __future__ import annotations

from enum import Enum
from typing import Callable, Literal, Optional, Tuple

import numpy as np

from .analytic import (
    downsample_ao,
    downsample_metallic,
    downsample_normal,
    downsample_roughness_lean,
)
from .filters import box_downsample, dpid_downsample
from .optimize import optimize_albedo

__all__ = [
    "compress_pbr_textures",
    "ROUGHNESS_LOWER_BOUND",
    "ROUGHNESS_NUMERIC_FLOOR",
    "MaterialClass",
    "classify_material",
    "_scaled_normal_var_threshold",
]

# doc.md 第 8 节注意事项 9：避免渲染时高光过尖产生数值问题。
# 历史上是硬 clamp 0.04，但对镜面金属（HR roughness < 0.04）会无中生有
# 把 LR 拉高、产生人为 HR/LR 不一致。新版改为「自适应」：
#   floor = max(NUMERIC_FLOOR, R_hr.min())
# 这样既保住 GGX D 函数的数值安全下限，又不会比 HR 真实最小值更高。
ROUGHNESS_LOWER_BOUND = 0.04         # 兼容旧 API；新版默认不再使用
ROUGHNESS_NUMERIC_FLOOR = 0.02       # GGX D 函数 fp16/fp32 数值安全下限

# 自适应 Albedo 起点的判据：
# - normal 几何方差 σ² 很小 → 渲染对 albedo 高频敏感度低，
#   DPID 保下来的边缘往往是噪声，box 反而更平稳。
# - σ² > 阈值（5e-4，factor=2 标定）→ 启用 DPID 边缘保持。
# 同样按 (factor/2)² 缩放（与 GEOMETRIC 阈值一致，量纲同源）。
NORMAL_VAR_DPID_THRESHOLD = 5e-4

# 材质分类阈值（在 **factor=2 块**上标定；其他 factor 由
# ``_scaled_normal_var_threshold`` 自动按 (factor/2)² 缩放）。
#
# 量纲推导：normal_var = 1 - |⟨n⟩|²。当块面积变大时，块内法线方向的
# 离散度上升，⟨n⟩ 的模长下降，σ² 大致按"块边长平方"线性增长（小扰动假设
# 下 ~ 块面积）。因此 σ²(factor) ≈ σ²(2) × (factor/2)²；阈值同步放大
# 才能保持"判 GEOMETRIC = 真有几何细节"的语义。
NORMAL_VAR_GEOMETRIC_THRESHOLD = 5e-4    # factor=2 标定值；≥ 此值 → 走完整管线
ROUGHNESS_STD_CONSTANT_THRESHOLD = 2e-2  # < 此值 → 视为 R 近常数（与 factor 无关）


def _scaled_normal_var_threshold(threshold_2x: float, factor: int) -> float:
    """按 ``(factor/2)²`` 缩放在 factor=2 上标定的 normal_var 阈值。

    Examples
    --------
    >>> _scaled_normal_var_threshold(5e-4, factor=2)
    0.0005
    >>> _scaled_normal_var_threshold(5e-4, factor=4)
    0.002
    >>> _scaled_normal_var_threshold(5e-4, factor=8)
    0.008
    """
    return float(threshold_2x) * (float(factor) / 2.0) ** 2


class MaterialClass(str, Enum):
    """材质类型，决定 ``compress_pbr_textures`` 的处理路径。"""
    CONSTANT = "CONSTANT"        # 全图近常数 → all-box
    METAL_FLAT = "METAL_FLAT"    # normal 平但 R/A 有变化 → all-box（LEAN 失衡）
    GEOMETRIC = "GEOMETRIC"      # normal 真有几何 → 完整 LEAN+DPID+Opt


def _normal_geometric_variance(N_hr: np.ndarray, factor: int = 2) -> float:
    """逐 ``factor × factor`` 块计算 normal 几何方差并取均值（与 LEAN 同源）。

    数值意义：``σ²_geom = 1 - |⟨n⟩|²``。``factor`` 越大，
    被块内吸收的法线扰动越多，``σ²`` 也越大；这与 LEAN 中
    ``alpha²_lr = alpha²_avg + 2σ²`` 是同一个 σ。
    """
    H, W = N_hr.shape[:2]
    f = int(factor)
    Hf, Wf = H // f, W // f
    blk = N_hr[: f * Hf, : f * Wf].reshape(Hf, f, Wf, f, 3)
    n_mean = blk.mean(axis=(1, 3))
    return float(1.0 - np.sum(n_mean * n_mean, axis=-1).mean())


def _adaptive_roughness_floor(
    R_hr: np.ndarray,
    numeric_floor: float = ROUGHNESS_NUMERIC_FLOOR,
) -> float:
    """LR 端 roughness 的自适应下限。

    取 ``max(numeric_floor, R_hr.min())``：
    - 当 R_hr 全部 ≥ numeric_floor → floor=R_hr.min()，clamp 不会主动抬高 LR；
    - 当 R_hr 含有 < numeric_floor 的极尖镜面像素 → floor=numeric_floor，
      只把这部分像素抬到数值安全区，其余保留；
    - **永远不会把 R 抬到比 HR 真实最小值还高的位置**，从而避免 Metal049A
      这类「整图镜面金属」被人为拖坏。
    """
    return max(float(numeric_floor), float(R_hr.min()))


def classify_material(
    N_hr: np.ndarray,
    R_hr: np.ndarray,
    normal_var_threshold: Optional[float] = None,
    roughness_std_threshold: float = ROUGHNESS_STD_CONSTANT_THRESHOLD,
    factor: int = 2,
) -> Tuple["MaterialClass", dict]:
    """根据 HR 信号粗略判定材质类型。

    Parameters
    ----------
    N_hr, R_hr : ndarray
        高分辨率 normal、roughness。
    normal_var_threshold : float | None
        normal 几何方差的"足以让 LEAN 起作用"阈值。
        默认 ``None`` → 自动按 ``_scaled_normal_var_threshold(5e-4, factor)``，
        因为 σ² 与块面积成正比，必须随 ``factor`` 同步缩放才能保持语义。
        显式传入数值则使用该固定阈值（绕过自适应，调试用）。
    roughness_std_threshold : float
        roughness 空间方差的"近常数"阈值，默认 0.02（与 factor 无关）。
    factor : int, 默认 2
        ``normal_var`` 在 ``factor × factor`` 块上计算；与下采倍率保持一致
        才能反映「该尺度下 LEAN 真正会吸收多少方差」。

    Returns
    -------
    cls : MaterialClass
    stats : dict
        统计量，便于调试 / 日志。
    """
    if normal_var_threshold is None:
        threshold_used = _scaled_normal_var_threshold(
            NORMAL_VAR_GEOMETRIC_THRESHOLD, factor
        )
    else:
        threshold_used = float(normal_var_threshold)

    nvar = _normal_geometric_variance(N_hr, factor=factor)
    rstd = float(R_hr.std())
    stats = {
        "normal_var": nvar,
        "roughness_std": rstd,
        "normal_var_threshold": threshold_used,
        "roughness_std_threshold": float(roughness_std_threshold),
        "factor": int(factor),
    }
    if nvar >= threshold_used:
        cls = MaterialClass.GEOMETRIC
    elif rstd < roughness_std_threshold:
        cls = MaterialClass.CONSTANT
    else:
        cls = MaterialClass.METAL_FLAT
    return cls, stats


def _all_box_compress(
    A_hr: np.ndarray,
    N_hr: np.ndarray,
    R_hr: np.ndarray,
    M_hr: np.ndarray,
    AO_hr: np.ndarray,
    metallic_threshold: float,
    factor: int = 2,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """CONSTANT / METAL_FLAT 路径：所有 5 通道纯 box 平均，无优化。

    Normal 仍然 box+normalize（保持单位向量），Metallic 仍然阈值化
    （避免中间值非物理），其余 box。
    """
    A_lr = box_downsample(A_hr, factor=factor)
    N_avg = box_downsample(N_hr, factor=factor)
    length = np.linalg.norm(N_avg, axis=-1, keepdims=True)
    N_lr = N_avg / np.maximum(length, 1e-8)
    R_lr = box_downsample(R_hr, factor=factor)
    M_avg = box_downsample(M_hr, factor=factor)
    M_lr = (M_avg >= metallic_threshold).astype(M_hr.dtype)
    AO_lr = box_downsample(AO_hr, factor=factor)
    return A_lr, N_lr, R_lr, M_lr, AO_lr


def compress_pbr_textures(
    A_hr: np.ndarray,
    N_hr: np.ndarray,
    R_hr: np.ndarray,
    M_hr: np.ndarray,
    AO_hr: np.ndarray,
    factor: int = 2,
    n_iter: int = 300,
    batch_views: int = 24,
    dpid_lam: float = 1.0,
    dpid_support: int = 4,
    metallic_threshold: float = 0.5,
    roughness_floor: Optional[float] = None,
    color_weight: float = 0.1,
    lr: float = 5e-3,
    device: Optional[str] = None,
    seed: Optional[int] = None,
    verbose: bool = True,
    on_step: Optional[Callable[[int, float], None]] = None,
    albedo_init: Literal["auto", "dpid", "box"] = "auto",
    auto_route: bool = True,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """完整 ``factor×`` 下采样管线（默认 2×；传 ``factor=4`` 即一次性 4×）。

    输入：所有 numpy float, **linear 空间**, normal 已解码到 ``[-1, 1]``。

    Parameters
    ----------
    factor : int, 默认 2
        下采倍率。``2`` 走经典 2× 链路；``4`` 一次性 4× 下采（所有 op 在
        4×4 块上做）。其他 2 的幂理论上也支持。

        .. note::
           ``factor`` 越大，LEAN 在块内吸收的法线方差越多，对镜面金属（α 极小）
           的过度模糊问题越严重。建议配合 ``auto_route`` 让 ``METAL_FLAT``
           自动回退到 box。
    n_iter : int
        Adam 迭代步数。``factor`` 增大时搜索空间更大，建议适度增加（300→400）。
    on_step : callable(it, loss) | None
        在 albedo 优化阶段每个迭代结束后回调一次，可用于绘 loss 曲线。
    roughness_floor : float | None
        若为 ``None``（默认）→ 用 ``_adaptive_roughness_floor(R_hr)``，
        即 ``max(0.02, R_hr.min())``，避免镜面金属（HR R<0.04）被人为抬高。
        显式传入数值则使用该固定下限（向后兼容旧调用方）。
    albedo_init : {"auto", "dpid", "box"}
        Albedo 优化起点。``auto``（默认）按 normal 几何方差自适应选择：
        几何近平的纹理用 box 起点（避免 DPID 把噪声高频放进来），
        几何丰富的纹理用 DPID。
    auto_route : bool, default True
        启用材质自适应路由：
        - ``CONSTANT`` / ``METAL_FLAT`` → 直接 all-box，跳过 LEAN/Opt（毫秒级）；
        - ``GEOMETRIC`` → 完整 LEAN+DPID+Adam 管线。

        显式传 ``False`` 时强制走完整管线（向后兼容旧脚本/baseline 对比）。

    Returns
    -------
    (A_lr, N_lr, R_lr, M_lr, AO_lr)
        分辨率 ``(H/factor, W/factor, ...)``。
    """
    _validate_inputs(A_hr, N_hr, R_hr, M_hr, AO_hr, factor=factor)

    # 路由决策（在与 factor 同尺度的块上算 normal_var）
    if auto_route:
        cls, stats = classify_material(N_hr, R_hr, factor=factor)
        if verbose:
            print(
                f"[route] cls={cls.value}  factor={factor}  "
                f"normal_var={stats['normal_var']:.2e}  "
                f"roughness_std={stats['roughness_std']:.3f}"
            )
        if cls in (MaterialClass.CONSTANT, MaterialClass.METAL_FLAT):
            if verbose:
                print(f"[route] -> all-box（跳过 LEAN/DPID/Opt）")
            A_lr, N_lr, R_lr, M_lr, AO_lr = _all_box_compress(
                A_hr, N_hr, R_hr, M_hr, AO_hr, metallic_threshold, factor=factor
            )
            return (
                A_lr.astype(A_hr.dtype, copy=False),
                N_lr.astype(N_hr.dtype, copy=False),
                R_lr.astype(R_hr.dtype, copy=False),
                M_lr.astype(M_hr.dtype, copy=False),
                AO_lr.astype(AO_hr.dtype, copy=False),
            )
        if verbose:
            print(f"[route] -> 完整 LEAN+DPID+Opt")

    # ===== 完整管线（GEOMETRIC 或 auto_route=False）=====

    # Step 1-4: 解析与启发式通道
    N_lr = downsample_normal(N_hr, factor=factor)
    R_lr = downsample_roughness_lean(R_hr, N_hr, factor=factor)
    M_lr = downsample_metallic(M_hr, threshold=metallic_threshold, factor=factor)
    AO_lr = downsample_ao(AO_hr, factor=factor)

    # 注意事项 9：clamp roughness 下界（自适应或显式）
    if roughness_floor is None:
        floor_used = _adaptive_roughness_floor(R_hr)
        if verbose:
            print(
                f"[init  rfloor] R_hr.min()={float(R_hr.min()):.4f} "
                f"-> 自适应 floor={floor_used:.4f}"
            )
    else:
        floor_used = float(roughness_floor)
        if verbose:
            print(f"[init  rfloor] 显式 floor={floor_used:.4f}")
    R_lr = np.clip(R_lr, floor_used, 1.0)

    # Step 5a: Albedo 初始化（自适应）
    if albedo_init == "auto":
        nvar = _normal_geometric_variance(N_hr, factor=factor)
        dpid_threshold = _scaled_normal_var_threshold(
            NORMAL_VAR_DPID_THRESHOLD, factor
        )
        if nvar < dpid_threshold:
            init_choice = "box"
        else:
            init_choice = "dpid"
        if verbose:
            print(
                f"[init  auto] normal_var={nvar:.2e} "
                f"(threshold {dpid_threshold:.2e}, factor={factor}) "
                f"-> 选择 {init_choice} 起点"
            )
    else:
        init_choice = albedo_init

    if init_choice == "box":
        A_lr_init = box_downsample(A_hr, factor=factor)
    elif init_choice == "dpid":
        A_lr_init = dpid_downsample(
            A_hr, lam=dpid_lam, support=dpid_support, factor=factor
        )
    else:
        raise ValueError(f"未知的 albedo_init={init_choice!r}")
    A_lr_init = np.clip(A_lr_init, 0.0, 1.0)

    # Step 5b: Albedo 可微优化（best-snapshot 早停内置）
    A_lr = optimize_albedo(
        A_hr, N_hr, R_hr, M_hr, AO_hr,
        A_lr_init, N_lr, R_lr, M_lr, AO_lr,
        factor=factor,
        n_iter=n_iter,
        batch_views=batch_views,
        lr=lr,
        color_weight=color_weight,
        device=device,
        seed=seed,
        verbose=verbose,
        on_step=on_step,
    )

    # 还原到输入 dtype，避免外部存图时类型错乱
    return (
        A_lr.astype(A_hr.dtype, copy=False),
        N_lr.astype(N_hr.dtype, copy=False),
        R_lr.astype(R_hr.dtype, copy=False),
        M_lr.astype(M_hr.dtype, copy=False),
        AO_lr.astype(AO_hr.dtype, copy=False),
    )


def _validate_inputs(
    A_hr: np.ndarray,
    N_hr: np.ndarray,
    R_hr: np.ndarray,
    M_hr: np.ndarray,
    AO_hr: np.ndarray,
    factor: int = 2,
) -> None:
    """检查所有 5 个通道分辨率一致且 H、W 能被 ``factor`` 整除。"""
    if A_hr.shape[-1] != 3:
        raise ValueError(f"A_hr 末维必须为 3，实际 {A_hr.shape}")
    if N_hr.shape[-1] != 3:
        raise ValueError(f"N_hr 末维必须为 3，实际 {N_hr.shape}")
    for name, x in [("R_hr", R_hr), ("M_hr", M_hr), ("AO_hr", AO_hr)]:
        if x.shape[-1] != 1:
            raise ValueError(f"{name} 末维必须为 1，实际 {x.shape}")

    H, W = A_hr.shape[:2]
    for name, x in [("N_hr", N_hr), ("R_hr", R_hr), ("M_hr", M_hr), ("AO_hr", AO_hr)]:
        if x.shape[:2] != (H, W):
            raise ValueError(
                f"{name} 分辨率 {x.shape[:2]} 与 A_hr {(H, W)} 不一致"
            )
    if not isinstance(factor, int) or factor < 2:
        raise ValueError(f"factor 必须是 >=2 的整数，实际 {factor}")
    if H % factor != 0 or W % factor != 0:
        raise ValueError(
            f"H、W 必须能被 factor={factor} 整除，实际 ({H}, {W})"
        )
