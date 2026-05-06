"""端到端合成纹理 demo：生成 64×64 PBR 贴图，下采样到 32×32 并打印指标。

直接运行::

    .venv\\Scripts\\python.exe examples/synthetic_demo.py
"""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np

from pbr_compress import compress_pbr_textures, evaluate_render_quality
from pbr_compress.io import save_albedo, save_normal, save_scalar


def make_demo_textures(H: int = 64, W: int = 64, seed: int = 42):
    """生成具有空间结构的合成 PBR 纹理。"""
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32) / np.array([H, W]).reshape(2, 1, 1)

    # Albedo：低频条纹 + 噪声
    base = 0.4 + 0.4 * np.stack(
        [
            np.sin(8 * np.pi * xx),
            np.cos(6 * np.pi * yy),
            np.sin(10 * np.pi * (xx + yy)),
        ],
        axis=-1,
    )
    A = np.clip(base + 0.05 * rng.standard_normal(base.shape).astype(np.float32),
                0.0, 1.0).astype(np.float32)

    # Normal：轻度凹凸（保持 z>0）
    bump = 0.4 * np.stack([np.sin(6 * np.pi * xx), np.cos(8 * np.pi * yy)], axis=-1)
    nz = np.sqrt(np.clip(1.0 - bump[..., 0] ** 2 - bump[..., 1] ** 2, 1e-6, 1.0))
    N = np.concatenate([bump, nz[..., None]], axis=-1).astype(np.float32)
    N = N / np.linalg.norm(N, axis=-1, keepdims=True)

    # Roughness
    R = (0.4 + 0.3 * np.sin(4 * np.pi * xx) + 0.1 * rng.random((H, W))).astype(np.float32)
    R = np.clip(R, 0.05, 1.0)[..., None]

    # Metallic：二值带状
    M = ((np.sin(3 * np.pi * yy) > 0.3).astype(np.float32))[..., None]

    # AO：平滑暗纹
    AO = (0.7 + 0.3 * np.cos(2 * np.pi * (xx + yy))).astype(np.float32)
    AO = np.clip(AO, 0.0, 1.0)[..., None]

    return A, N, R, M, AO


def main() -> None:
    out_dir = Path(__file__).parent / "out"
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=== 合成 64×64 PBR 纹理 ===")
    A_hr, N_hr, R_hr, M_hr, AO_hr = make_demo_textures(64, 64)

    print("=== 运行压缩管线 (n_iter=200, batch_views=24) ===")
    t0 = time.time()
    A_lr, N_lr, R_lr, M_lr, AO_lr = compress_pbr_textures(
        A_hr, N_hr, R_hr, M_hr, AO_hr,
        n_iter=200, batch_views=24, seed=0, verbose=True,
    )
    print(f"耗时: {time.time() - t0:.2f}s")

    print(f"=== 保存输出到 {out_dir} ===")
    save_albedo(out_dir / "albedo_hr.png", A_hr)
    save_albedo(out_dir / "albedo_lr.png", A_lr)
    save_normal(out_dir / "normal_hr.png", N_hr)
    save_normal(out_dir / "normal_lr.png", N_lr)
    save_scalar(out_dir / "rough_hr.png",  R_hr)
    save_scalar(out_dir / "rough_lr.png",  R_lr)
    save_scalar(out_dir / "metal_hr.png",  M_hr)
    save_scalar(out_dir / "metal_lr.png",  M_lr)
    save_scalar(out_dir / "ao_hr.png",     AO_hr)
    save_scalar(out_dir / "ao_lr.png",     AO_lr)

    print("=== 渲染层指标 ===")
    m = evaluate_render_quality(
        A_hr, N_hr, R_hr, M_hr, AO_hr,
        A_lr, N_lr, R_lr, M_lr, AO_lr,
        n_pairs=64, seed=0,
    )
    for k, v in m.items():
        print(f"  {k:32s} = {v}")


if __name__ == "__main__":
    main()
