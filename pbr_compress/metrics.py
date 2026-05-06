"""验证与质量控制：贴图层指标 + 渲染层指标。

对应 doc.md 第 7 节。

- 贴图层（弱指标，仅参考）：
    - albedo PSNR vs 高分 box-down
    - normal 角度误差（mean / median / max，单位：度）
- 渲染层（强指标）：
    - 固定 N 对随机视角，计算 reference vs compressed 的 L1、PSNR、SSIM、FLIP
"""
from __future__ import annotations

import math
from typing import Dict, Optional

import numpy as np
import torch

from .brdf import sample_directions
from .filters import box_downsample_2x

__all__ = [
    "psnr",
    "ssim_simple",
    "normal_angle_error",
    "evaluate_render_quality",
]


def psnr(a: np.ndarray, b: np.ndarray, data_range: float = 1.0) -> float:
    """标准 PSNR（dB）。"""
    diff = a.astype(np.float64) - b.astype(np.float64)
    mse = float(np.mean(diff * diff))
    if mse <= 0.0:
        return float("inf")
    return 20.0 * math.log10(data_range / math.sqrt(mse))


def ssim_simple(a: np.ndarray, b: np.ndarray, data_range: float = 1.0) -> float:
    """简化 SSIM（基于全图均值/方差/协方差）。仅用于参考性比较。"""
    a = a.astype(np.float64)
    b = b.astype(np.float64)
    mu_a, mu_b = a.mean(), b.mean()
    va, vb = a.var(), b.var()
    cov = ((a - mu_a) * (b - mu_b)).mean()
    c1 = (0.01 * data_range) ** 2
    c2 = (0.03 * data_range) ** 2
    num = (2 * mu_a * mu_b + c1) * (2 * cov + c2)
    den = (mu_a * mu_a + mu_b * mu_b + c1) * (va + vb + c2)
    return float(num / max(den, 1e-12))


def normal_angle_error(N_ref: np.ndarray, N_lr: np.ndarray) -> Dict[str, float]:
    """逐 texel 法线角度误差（度），返回 mean / median / max。"""
    cos = np.clip((N_ref * N_lr).sum(axis=-1), -1.0, 1.0)
    err = np.degrees(np.arccos(cos))
    return {
        "mean": float(err.mean()),
        "median": float(np.median(err)),
        "max": float(err.max()),
    }


def evaluate_render_quality(
    A_hr: np.ndarray,
    N_hr: np.ndarray,
    R_hr: np.ndarray,
    M_hr: np.ndarray,
    AO_hr: np.ndarray,
    A_lr: np.ndarray,
    N_lr: np.ndarray,
    R_lr: np.ndarray,
    M_lr: np.ndarray,
    AO_lr: np.ndarray,
    n_pairs: int = 64,
    device: Optional[str] = None,
    seed: Optional[int] = 0,
    use_flip: bool = True,
) -> Dict[str, object]:
    """对一组固定视-光对计算渲染层 reference vs compressed 的差异。

    Returns
    -------
    dict
        ``render_l1`` (linear)，``render_psnr_ldr`` / ``render_ssim_ldr``
        （Reinhard tone-map 后），``render_flip_mean``（若 ``use_flip``），
        以及 ``albedo_psnr_vs_box``、``normal_angle_error_deg``。
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)

    def to_t(x: np.ndarray) -> torch.Tensor:
        return torch.from_numpy(np.ascontiguousarray(x)).float().to(device)

    A_hr_t = to_t(A_hr); N_hr_t = to_t(N_hr); R_hr_t = to_t(R_hr)
    M_hr_t = to_t(M_hr); AO_hr_t = to_t(AO_hr)
    A_lr_t = to_t(A_lr); N_lr_t = to_t(N_lr); R_lr_t = to_t(R_lr)
    M_lr_t = to_t(M_lr); AO_lr_t = to_t(AO_lr)

    gen = torch.Generator(device=device)
    if seed is not None:
        gen.manual_seed(int(seed))
    wi, wo = sample_directions(n_pairs, device, generator=gen)

    # 延迟 import 避免循环引用
    from .render import render_hr_footprint_avg, render_lr

    with torch.no_grad():
        ref = render_hr_footprint_avg(
            A_hr_t, N_hr_t, R_hr_t, M_hr_t, AO_hr_t, wi, wo
        )                                              # (H/2, W/2, B, 3)
        cmp = render_lr(
            A_lr_t, N_lr_t, R_lr_t, M_lr_t, AO_lr_t, wi, wo
        )

    ref_np = ref.cpu().numpy()
    cmp_np = cmp.cpu().numpy()

    # Reinhard tone-map 到 [0, 1] 用于 LDR 指标
    def reinhard(x: np.ndarray) -> np.ndarray:
        return x / (1.0 + x)

    ref_ldr = np.clip(reinhard(ref_np), 0.0, 1.0)
    cmp_ldr = np.clip(reinhard(cmp_np), 0.0, 1.0)

    out: Dict[str, object] = {
        "render_l1": float(np.abs(ref_np - cmp_np).mean()),
        "render_psnr_ldr": psnr(ref_ldr, cmp_ldr, data_range=1.0),
        "render_ssim_ldr": ssim_simple(ref_ldr, cmp_ldr, data_range=1.0),
    }

    if use_flip:
        try:
            import flip_evaluator                         # noqa: WPS433

            B = ref_ldr.shape[2]
            flips = []
            for i in range(B):
                r = ref_ldr[:, :, i, :].astype(np.float32)
                c = cmp_ldr[:, :, i, :].astype(np.float32)
                # flip_evaluator.evaluate(reference, test, "LDR")
                _, mean_flip, _ = flip_evaluator.evaluate(r, c, "LDR")
                flips.append(float(mean_flip))
            out["render_flip_mean"] = float(np.mean(flips))
        except Exception as exc:  # noqa: BLE001
            out["render_flip_mean"] = None
            out["render_flip_error"] = repr(exc)

    # 贴图层补充指标
    out["albedo_psnr_vs_box"] = psnr(box_downsample_2x(A_hr), A_lr)

    # 法线参考：用 box + 归一化做基线（与 downsample_normal 等价，但不引入循环 import）
    N_box = box_downsample_2x(N_hr)
    N_box = N_box / np.maximum(np.linalg.norm(N_box, axis=-1, keepdims=True), 1e-8)
    out["normal_angle_error_deg"] = normal_angle_error(N_box, N_lr)

    return out
