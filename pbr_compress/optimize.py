"""Albedo 可微优化：log-空间 L1 重建 + 全局色调一致性。

对应 doc.md 第 4.3 后半节与 4.4 节。Loss 选择 ``log L1`` 是为了在
diffuse / specular 数量级差异巨大时仍然平衡（高光不至于主导）。

为了避免「训练 loss 在降但 holdout 渲染 L1 在涨」（典型如金属、几何近常数
的纹理），优化循环额外维护一组**固定的 holdout 视角**，每 ``eval_every``
步计算一次 holdout 渲染 L1，并保留 L1 最低时的 albedo 快照作为最终输出
（即 best-snapshot 早停）。
"""
from __future__ import annotations

from typing import Callable, Optional

import numpy as np
import torch

from .brdf import sample_directions
from .render import render_hr_footprint_avg, render_lr

__all__ = ["compute_loss", "optimize_albedo"]


def compute_loss(
    pred: torch.Tensor, target: torch.Tensor, color_weight: float = 0.1
) -> torch.Tensor:
    """``log`` 空间 L1 重建 + 全局色调一致性。

    Parameters
    ----------
    pred, target : ``(..., 3)`` tensor
        通常 shape 为 ``(H/2, W/2, B, 3)``。
    color_weight : float
        全局色调一致性的权重，doc.md 推荐 ``0.05 ~ 0.2``，默认 ``0.1``。
    """
    eps = 1e-3
    log_pred = torch.log(pred.clamp_min(0.0) + eps)
    log_target = torch.log(target.clamp_min(0.0) + eps)
    L_recon = (log_pred - log_target).abs().mean()

    # 全局色调一致性：除最后 channel 维外全部求平均
    spatial_view_dims = tuple(range(pred.ndim - 1))
    mean_pred = pred.mean(dim=spatial_view_dims)
    mean_target = target.mean(dim=spatial_view_dims)
    L_color = (mean_pred - mean_target).abs().mean()

    return L_recon + color_weight * L_color


def _eval_l1_chunked(
    A_lr: torch.Tensor,
    A_hr_t: torch.Tensor,
    N_hr_t: torch.Tensor,
    R_hr_t: torch.Tensor,
    M_hr_t: torch.Tensor,
    AO_hr_t: torch.Tensor,
    N_lr_t: torch.Tensor,
    R_lr_t: torch.Tensor,
    M_lr_t: torch.Tensor,
    AO_lr_t: torch.Tensor,
    wi_eval: torch.Tensor,
    wo_eval: torch.Tensor,
    chunk: int,
    factor: int = 2,
) -> float:
    """在 holdout 视角上分块计算 linear L1，避免 2K 贴图 OOM。"""
    total, npx = 0.0, 0
    n_pairs = wi_eval.shape[0]
    for s in range(0, n_pairs, chunk):
        wi_c = wi_eval[s:s + chunk]
        wo_c = wo_eval[s:s + chunk]
        ref_c = render_hr_footprint_avg(
            A_hr_t, N_hr_t, R_hr_t, M_hr_t, AO_hr_t, wi_c, wo_c, factor=factor
        )
        cmp_c = render_lr(A_lr, N_lr_t, R_lr_t, M_lr_t, AO_lr_t, wi_c, wo_c)
        diff = (ref_c - cmp_c).abs()
        total += float(diff.sum().item())
        npx += int(diff.numel())
        del ref_c, cmp_c, diff
    return total / max(1, npx)


def optimize_albedo(
    A_hr: np.ndarray,
    N_hr: np.ndarray,
    R_hr: np.ndarray,
    M_hr: np.ndarray,
    AO_hr: np.ndarray,
    A_lr_init: np.ndarray,
    N_lr: np.ndarray,
    R_lr: np.ndarray,
    M_lr: np.ndarray,
    AO_lr: np.ndarray,
    factor: int = 2,
    n_iter: int = 300,
    batch_views: int = 24,
    lr: float = 5e-3,
    color_weight: float = 0.1,
    device: Optional[str | torch.device] = None,
    seed: Optional[int] = None,
    verbose: bool = True,
    log_every: int = 50,
    on_step: Optional[Callable[[int, float], None]] = None,
    holdout_pairs: int = 32,
    holdout_chunk: int = 8,
    eval_every: int = 5,
    holdout_seed: int = 12345,
) -> np.ndarray:
    """优化低分 albedo，使其与高分 footprint 平均渲染一致。

    所有 numpy 输入会自动 ``to_t`` 到 ``device``，最终返回 numpy。

    Parameters
    ----------
    A_lr_init : ndarray
        DPID 或其他方法给出的初始低分 albedo。
    factor : int, 默认 2
        HR / LR 的倍率比；HR footprint 渲染按 ``factor × factor`` 块平均。
        ``A_lr_init.shape[:2]`` 必须等于 ``(H_hr // factor, W_hr // factor)``。
    seed : int | None
        若提供，则用专属 RNG 控制**训练**视角采样，方便复现。
    on_step : callable(it, loss) | None
        每个迭代结束后的回调，可用于记录 loss 历史、绘制曲线等。
    holdout_pairs : int
        用作 best-snapshot 早停的 holdout 视角对数（与训练视角不重叠）。
    holdout_chunk : int
        holdout 渲染分块大小，避免 2K+ 贴图 OOM。
    eval_every : int
        每多少步评估一次 holdout L1。
    holdout_seed : int
        holdout 视角的固定种子（与 ``seed`` 不同，确保不重叠）。

    Returns
    -------
    A_lr : ndarray
        ``argmin_t holdout_L1(t)`` 对应的 albedo 快照（best-snapshot），
        若所有步都比 init 差，则原样返回 init。
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)

    def to_t(x: np.ndarray) -> torch.Tensor:
        return torch.from_numpy(np.ascontiguousarray(x)).float().to(device)

    A_hr_t = to_t(A_hr)
    N_hr_t = to_t(N_hr)
    R_hr_t = to_t(R_hr)
    M_hr_t = to_t(M_hr)
    AO_hr_t = to_t(AO_hr)

    N_lr_t = to_t(N_lr)
    R_lr_t = to_t(R_lr)
    M_lr_t = to_t(M_lr)
    AO_lr_t = to_t(AO_lr)

    A_lr = torch.nn.Parameter(to_t(A_lr_init))
    optim = torch.optim.Adam([A_lr], lr=lr)

    train_gen: Optional[torch.Generator] = None
    if seed is not None:
        train_gen = torch.Generator(device=device)
        train_gen.manual_seed(int(seed))

    # holdout 视角：固定，独立于训练 RNG
    eval_gen = torch.Generator(device=device)
    eval_gen.manual_seed(int(holdout_seed))
    wi_eval, wo_eval = sample_directions(holdout_pairs, device, generator=eval_gen)

    def eval_holdout(p: torch.Tensor) -> float:
        with torch.no_grad():
            return _eval_l1_chunked(
                p, A_hr_t, N_hr_t, R_hr_t, M_hr_t, AO_hr_t,
                N_lr_t, R_lr_t, M_lr_t, AO_lr_t,
                wi_eval, wo_eval, chunk=holdout_chunk, factor=factor,
            )

    init_holdout = eval_holdout(A_lr)
    best_holdout = init_holdout
    best_snapshot = A_lr.detach().clone()
    best_iter = -1   # -1 表示「最优是 init」

    if verbose:
        print(f"[init      ] holdout_L1 = {init_holdout:.6f}")

    last_loss = float("nan")
    for it in range(n_iter):
        wi, wo = sample_directions(batch_views, device, generator=train_gen)

        with torch.no_grad():
            target = render_hr_footprint_avg(
                A_hr_t, N_hr_t, R_hr_t, M_hr_t, AO_hr_t, wi, wo, factor=factor
            )

        pred = render_lr(A_lr, N_lr_t, R_lr_t, M_lr_t, AO_lr_t, wi, wo)
        loss = compute_loss(pred, target, color_weight=color_weight)

        optim.zero_grad()
        loss.backward()
        optim.step()

        with torch.no_grad():
            A_lr.clamp_(0.0, 1.0)

        last_loss = float(loss.detach().item())
        if on_step is not None:
            on_step(it, last_loss)

        if (it + 1) % eval_every == 0 or it == n_iter - 1:
            cur_holdout = eval_holdout(A_lr)
            if cur_holdout < best_holdout:
                best_holdout = cur_holdout
                best_snapshot = A_lr.detach().clone()
                best_iter = it
            if verbose and (it % log_every == 0 or it == n_iter - 1):
                print(
                    f"[iter {it:4d}] loss={last_loss:.5f}  "
                    f"holdout_L1={cur_holdout:.6f}  "
                    f"best={best_holdout:.6f}@it{best_iter}"
                )
        elif verbose and it % log_every == 0:
            print(f"[iter {it:4d}] loss={last_loss:.5f}")

    if verbose:
        if best_iter < 0:
            print(
                f"[final     ] init 始终最优 ({init_holdout:.6f})，"
                f"回退到初始化（无效优化）"
            )
        else:
            improvement = (init_holdout - best_holdout) / max(init_holdout, 1e-12)
            print(
                f"[final     ] best holdout_L1={best_holdout:.6f} "
                f"@iter {best_iter} (init {init_holdout:.6f}, "
                f"提升 {improvement * 100:.1f}%)"
            )

    return best_snapshot.detach().cpu().numpy()
