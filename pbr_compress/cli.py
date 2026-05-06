"""命令行入口：批量压缩一组 PBR 纹理。

用法示例（在虚拟环境内）::

    .venv\\Scripts\\python.exe -m pbr_compress.cli ^
        --albedo  input/albedo.png ^
        --normal  input/normal.png ^
        --rough   input/roughness.png ^
        --metal   input/metallic.png ^
        --ao      input/ao.png ^
        --out_dir output/ ^
        --n_iter  300

Albedo 默认按 sRGB 读取并解码到 linear，输出再编码回 sRGB。法线磁盘存储
``(n+1)/2 ∈ [0, 1]``，处理时会自动解码到 ``[-1, 1]``。
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from .io import (
    load_albedo,
    load_normal,
    load_scalar,
    save_albedo,
    save_normal,
    save_scalar,
)
from .metrics import evaluate_render_quality
from .pipeline import compress_pbr_textures


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="pbr-compress",
        description="2× 下采样 PBR 纹理离线压缩管线（参见 doc.md）",
    )
    p.add_argument("--albedo", required=True, help="albedo 贴图路径")
    p.add_argument("--normal", required=True, help="normal 贴图路径")
    p.add_argument("--rough",  required=True, help="roughness 单通道贴图")
    p.add_argument("--metal",  required=True, help="metallic 单通道贴图")
    p.add_argument("--ao",     required=True, help="AO 单通道贴图")
    p.add_argument("--out_dir", required=True, help="输出目录")

    p.add_argument(
        "--no_srgb_albedo",
        action="store_true",
        help="若 albedo 已是 linear 则加此 flag（默认按 sRGB 处理）",
    )
    p.add_argument("--n_iter",       type=int,   default=300)
    p.add_argument("--batch_views",  type=int,   default=24)
    p.add_argument("--lr",           type=float, default=5e-3)
    p.add_argument("--dpid_lam",     type=float, default=1.0)
    p.add_argument("--dpid_support", type=int,   default=4)
    p.add_argument("--color_weight", type=float, default=0.1)
    p.add_argument(
        "--device",
        type=str,
        default=None,
        help="cuda / cpu / cuda:0；默认自动检测",
    )
    p.add_argument("--seed",      type=int, default=0)
    p.add_argument(
        "--bit_depth", type=int, default=8, choices=[8, 16],
        help="输出 PNG 位深",
    )
    p.add_argument(
        "--no_eval",
        action="store_true",
        help="跳过渲染层评估（FLIP 较慢，可选关掉）",
    )
    p.add_argument(
        "--quiet",
        action="store_true",
        help="不打印逐步 loss",
    )
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[load] albedo = {args.albedo}  (sRGB={not args.no_srgb_albedo})")
    A_hr = load_albedo(args.albedo, srgb=not args.no_srgb_albedo)
    print(f"[load] normal = {args.normal}")
    N_hr = load_normal(args.normal)
    print(f"[load] rough  = {args.rough}")
    R_hr = load_scalar(args.rough)
    print(f"[load] metal  = {args.metal}")
    M_hr = load_scalar(args.metal)
    print(f"[load] ao     = {args.ao}")
    AO_hr = load_scalar(args.ao)

    H, W = A_hr.shape[:2]
    print(f"[run]  HR=({H}x{W})  ->  LR=({H // 2}x{W // 2})")

    t0 = time.time()
    A_lr, N_lr, R_lr, M_lr, AO_lr = compress_pbr_textures(
        A_hr, N_hr, R_hr, M_hr, AO_hr,
        n_iter=args.n_iter,
        batch_views=args.batch_views,
        dpid_lam=args.dpid_lam,
        dpid_support=args.dpid_support,
        color_weight=args.color_weight,
        lr=args.lr,
        device=args.device,
        seed=args.seed,
        verbose=not args.quiet,
    )
    print(f"[done] 耗时 {time.time() - t0:.1f}s")

    save_albedo(
        out_dir / "albedo.png",
        A_lr,
        srgb=not args.no_srgb_albedo,
        bit_depth=args.bit_depth,
    )
    save_normal(out_dir / "normal.png",    N_lr, bit_depth=args.bit_depth)
    save_scalar(out_dir / "roughness.png", R_lr, bit_depth=args.bit_depth)
    save_scalar(out_dir / "metallic.png",  M_lr, bit_depth=args.bit_depth)
    save_scalar(out_dir / "ao.png",        AO_lr, bit_depth=args.bit_depth)
    print(f"[save] 写入 {out_dir}")

    if not args.no_eval:
        print("[eval] 计算渲染层指标 ...")
        m = evaluate_render_quality(
            A_hr, N_hr, R_hr, M_hr, AO_hr,
            A_lr, N_lr, R_lr, M_lr, AO_lr,
            device=args.device, seed=args.seed,
        )
        for k, v in m.items():
            print(f"  {k:32s} = {v}")
        with (out_dir / "metrics.json").open("w", encoding="utf-8") as f:
            json.dump(m, f, indent=2, default=str, ensure_ascii=False)
        print(f"[save] 指标写入 {out_dir / 'metrics.json'}")


if __name__ == "__main__":
    main()
