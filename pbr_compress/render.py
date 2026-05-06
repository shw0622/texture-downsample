"""高分 footprint 平均渲染 + 低分直接渲染。

对应 doc.md 第 4.3 节。所有张量假设 shape ``(H, W, C)``，wi/wo 为
``(B, 3)``，输出 shape 为 ``(H/2, W/2, B, 3)``。
"""
from __future__ import annotations

import torch

from .brdf import evaluate_brdf

__all__ = ["render_hr_footprint_avg", "render_lr"]


def _expand_view_axis(t: torch.Tensor) -> torch.Tensor:
    """把纹理 ``(H, W, C)`` 扩展到 ``(H, W, 1, C)``，便于与 view 维广播。"""
    return t.unsqueeze(-2)


def render_hr_footprint_avg(
    A_hr: torch.Tensor,
    N_hr: torch.Tensor,
    R_hr: torch.Tensor,
    M_hr: torch.Tensor,
    AO_hr: torch.Tensor,
    wi: torch.Tensor,
    wo: torch.Tensor,
    factor: int = 2,
) -> torch.Tensor:
    """对每个低分 texel × 每对 ``(wi, wo)``，返回其 ``factor×factor``
    footprint 内 BRDF 响应的平均。

    Parameters
    ----------
    factor : int
        下采样倍数。``2`` 表示 2x2 footprint；``4`` 表示 4x4 footprint，
        用于评估 4× 端到端 (mip2) 的端到端口径。

    Returns
    -------
    Tensor, shape ``(H/factor, W/factor, B, 3)``

    Notes
    -----
    中间张量为 ``(H, W, B, 3)``，对 1K 纹理 + ``B=24`` 约 300 MB；显存紧张时
    应在调用层按 tile 分块（doc.md 第 8 节注意事项 3）。
    """
    if factor < 1:
        raise ValueError(f"render_hr_footprint_avg: factor 必须 ≥ 1，实际 {factor}")
    if wi.ndim != 2 or wi.shape[-1] != 3:
        raise ValueError("render_hr_footprint_avg: wi 必须是 (B, 3)")
    if wo.shape != wi.shape:
        raise ValueError("render_hr_footprint_avg: wi 与 wo shape 不一致")

    B = wi.shape[0]
    wi_b = wi.view(1, 1, B, 3)
    wo_b = wo.view(1, 1, B, 3)

    L_hr = evaluate_brdf(
        _expand_view_axis(A_hr),
        _expand_view_axis(N_hr),
        _expand_view_axis(R_hr),
        _expand_view_axis(M_hr),
        _expand_view_axis(AO_hr),
        wi_b,
        wo_b,
    )                                                # (H, W, B, 3)

    if factor == 1:
        return L_hr

    H, W = L_hr.shape[:2]
    if H % factor != 0 or W % factor != 0:
        raise ValueError(
            f"render_hr_footprint_avg: H、W 必须能被 factor={factor} 整除，"
            f"实际 ({H}, {W})"
        )
    Hl, Wl = H // factor, W // factor
    L_hr = L_hr.reshape(Hl, factor, Wl, factor, B, 3).mean(dim=(1, 3))
    return L_hr                                      # (H/factor, W/factor, B, 3)


def render_lr(
    A_lr: torch.Tensor,
    N_lr: torch.Tensor,
    R_lr: torch.Tensor,
    M_lr: torch.Tensor,
    AO_lr: torch.Tensor,
    wi: torch.Tensor,
    wo: torch.Tensor,
) -> torch.Tensor:
    """逐低分 texel × 视-光对，直接评估 BRDF。

    Returns
    -------
    Tensor, shape ``(H/2, W/2, B, 3)``
    """
    if wi.ndim != 2 or wi.shape[-1] != 3:
        raise ValueError("render_lr: wi 必须是 (B, 3)")
    B = wi.shape[0]
    wi_b = wi.view(1, 1, B, 3)
    wo_b = wo.view(1, 1, B, 3)
    return evaluate_brdf(
        _expand_view_axis(A_lr),
        _expand_view_axis(N_lr),
        _expand_view_axis(R_lr),
        _expand_view_axis(M_lr),
        _expand_view_axis(AO_lr),
        wi_b,
        wo_b,
    )
