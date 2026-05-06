"""V3 端到端报告：native-4x + 材质路由（5 张全量）。

把方法从 chain-2x 切到 native-4x（``compress_pbr_textures(factor=4)``），
并启用 ``auto_route``——这是面向工业落地的最终形态。

对比 4 个方法：
- **A1 box-4x**       : 单次 4×4 box，下界基线
- **B1 lanczos-4x**   : 单次 Lanczos-3 4× 下采，业界默认
- **C2 native**       : ``factor=4, auto_route=False``，裸方法上限
- **C2_R native+routed** : ``factor=4, auto_route=True``，**最终落地形态**

核心 promise：
1. C2_R 在每张上 FLIP ≤ A1（不输 box）
2. C2_R 在 GEOMETRIC 纹理上恢复 C2 的全部增益
3. C2_R 在 CONSTANT/METAL_FLAT 纹理上时间砍到 box 同档（毫秒级）

输出：
    examples/bench_out/verify_v3.csv
    examples/bench_out/verify_v3.md
"""
from __future__ import annotations

import csv
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from pbr_compress.brdf import sample_directions
from pbr_compress.filters import box_downsample
from pbr_compress.io import load_albedo, load_normal, load_scalar
from pbr_compress.metrics import psnr
from pbr_compress.pipeline import classify_material, compress_pbr_textures
from pbr_compress.render import render_hr_footprint_avg, render_lr

ROOT = Path(__file__).resolve().parent.parent
TEX_ROOT = ROOT / "textures"
OUT_DIR = Path(__file__).resolve().parent / "bench_out"

DOWNSAMPLE_FACTOR = 4
EVAL_PAIRS = 32
EVAL_CHUNK = 4

OURS_KWARGS = dict(
    n_iter=300,
    batch_views=24,
    dpid_lam=1.0,
    seed=0,
    verbose=False,
)


# ---------------------------------------------------------------------------
# IO
# ---------------------------------------------------------------------------


def find_one(folder: Path, suffixes):
    for f in sorted(folder.glob("*.png")):
        if any(s.lower() in f.name.lower() for s in suffixes):
            return f
    return None


def load_set(folder: Path):
    A = load_albedo(find_one(folder, ["_Color"]), srgb=True)
    N = load_normal(find_one(folder, ["_NormalGL"]))
    R = load_scalar(find_one(folder, ["_Roughness"]))
    H, W = A.shape[:2]
    mp = find_one(folder, ["_Metalness", "_Metallic"])
    M = load_scalar(mp) if mp else np.zeros((H, W, 1), dtype=np.float32)
    aop = find_one(folder, ["_AmbientOcclusion", "_AO"])
    AO = load_scalar(aop) if aop else np.ones((H, W, 1), dtype=np.float32)
    return A, N, R, M, AO


# ---------------------------------------------------------------------------
# 评估
# ---------------------------------------------------------------------------


def evaluate_pair(ref_t: torch.Tensor, cmp_t: torch.Tensor):
    def reinhard(x): return x / (1.0 + x)
    ref = ref_t.cpu().numpy()
    cmp = cmp_t.cpu().numpy()
    ref_ldr = np.clip(reinhard(ref), 0.0, 1.0)
    cmp_ldr = np.clip(reinhard(cmp), 0.0, 1.0)
    out = {
        "l1": float(np.abs(ref - cmp).mean()),
        "psnr_ldr": psnr(ref_ldr, cmp_ldr, data_range=1.0),
    }
    try:
        import flip_evaluator
        flips = []
        B = ref_ldr.shape[2]
        for i in range(B):
            r = ref_ldr[:, :, i, :].astype(np.float32)
            c = cmp_ldr[:, :, i, :].astype(np.float32)
            _, mean_flip, _ = flip_evaluator.evaluate(r, c, "LDR")
            flips.append(float(mean_flip))
        out["flip"] = float(np.mean(flips))
    except Exception:
        out["flip"] = None
    return out


def render_lr_chunked(A_lr, N_lr, R_lr, M_lr, AO_lr, wi_all, wo_all, chunk):
    out = []
    with torch.no_grad():
        for s in range(0, wi_all.shape[0], chunk):
            cmp_c = render_lr(
                A_lr, N_lr, R_lr, M_lr, AO_lr,
                wi_all[s:s + chunk], wo_all[s:s + chunk],
            )
            out.append(cmp_c.cpu())
    return torch.cat(out, dim=2)


def render_ref_chunked(A_hr, N_hr, R_hr, M_hr, AO_hr, wi_all, wo_all, chunk, factor):
    out = []
    with torch.no_grad():
        for s in range(0, wi_all.shape[0], chunk):
            ref_c = render_hr_footprint_avg(
                A_hr, N_hr, R_hr, M_hr, AO_hr,
                wi_all[s:s + chunk], wo_all[s:s + chunk],
                factor=factor,
            )
            out.append(ref_c.cpu())
    return torch.cat(out, dim=2)


# ---------------------------------------------------------------------------
# 4 个方法
# ---------------------------------------------------------------------------


def method_box_4x(A, N, R, M, AO):
    A_lr = box_downsample(A, factor=4)
    N_avg = box_downsample(N, factor=4)
    length = np.linalg.norm(N_avg, axis=-1, keepdims=True)
    N_lr = N_avg / np.maximum(length, 1e-8)
    R_lr = box_downsample(R, factor=4)
    M_avg = box_downsample(M, factor=4)
    M_lr = (M_avg >= 0.5).astype(M.dtype)
    AO_lr = box_downsample(AO, factor=4)
    return A_lr, N_lr, R_lr, M_lr, AO_lr


def _lanczos_4x(x: np.ndarray) -> np.ndarray:
    H, W = x.shape[:2]
    new_w, new_h = W // 4, H // 4
    out = np.empty((new_h, new_w, x.shape[-1]), dtype=np.float32)
    for c in range(x.shape[-1]):
        ch = x[..., c].astype(np.float32)
        im = Image.fromarray(ch, mode="F")
        im_lr = im.resize((new_w, new_h), Image.LANCZOS)
        out[..., c] = np.asarray(im_lr, dtype=np.float32)
    return out


def method_lanczos_4x(A, N, R, M, AO):
    A_lr = np.clip(_lanczos_4x(A), 0.0, 1.0)
    N_avg = _lanczos_4x(N)
    length = np.linalg.norm(N_avg, axis=-1, keepdims=True)
    N_lr = N_avg / np.maximum(length, 1e-8)
    R_lr = np.clip(_lanczos_4x(R), 0.0, 1.0)
    M_lr = (np.clip(_lanczos_4x(M), 0.0, 1.0) >= 0.5).astype(M.dtype)
    AO_lr = np.clip(_lanczos_4x(AO), 0.0, 1.0)
    return A_lr, N_lr, R_lr, M_lr, AO_lr


def method_native_forced(A, N, R, M, AO):
    return compress_pbr_textures(
        A, N, R, M, AO, factor=4, auto_route=False, **OURS_KWARGS
    )


def method_native_routed(A, N, R, M, AO):
    return compress_pbr_textures(
        A, N, R, M, AO, factor=4, auto_route=True, **OURS_KWARGS
    )


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------


def evaluate_method(method_fn, A, N, R, M, AO, ref, wi, wo, device, chunk):
    t0 = time.time()
    lr_tuple = method_fn(A, N, R, M, AO)
    t_proc = time.time() - t0

    def t(x): return torch.from_numpy(np.ascontiguousarray(x)).float().to(device)
    A_lr_t, N_lr_t, R_lr_t, M_lr_t, AO_lr_t = map(t, lr_tuple)

    cmp = render_lr_chunked(A_lr_t, N_lr_t, R_lr_t, M_lr_t, AO_lr_t, wi, wo, chunk)
    metrics = evaluate_pair(ref, cmp)
    metrics["proc_s"] = round(t_proc, 2)
    return metrics


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tex_dirs = [d for d in sorted(TEX_ROOT.iterdir()) if d.is_dir()]
    print(
        f"=== verify_v3  factor={DOWNSAMPLE_FACTOR}  eval_pairs={EVAL_PAIRS}  "
        f"device={device} ===\n"
    )

    rows = []
    for folder in tex_dirs:
        name = folder.name
        A, N, R, M, AO = load_set(folder)
        H, W = A.shape[:2]
        cls, stats = classify_material(N, R, factor=DOWNSAMPLE_FACTOR)
        print(
            f"\n[{name}]  HR=({H}x{W}) -> LR=({H//DOWNSAMPLE_FACTOR}x"
            f"{W//DOWNSAMPLE_FACTOR})  cls={cls.value}  "
            f"nvar={stats['normal_var']:.2e}  rstd={stats['roughness_std']:.3f}"
        )

        # 共享 ref
        def t(x): return torch.from_numpy(np.ascontiguousarray(x)).float().to(device)
        A_t, N_t, R_t, M_t, AO_t = map(t, [A, N, R, M, AO])
        g = torch.Generator(device=device); g.manual_seed(0)
        wi, wo = sample_directions(EVAL_PAIRS, device, generator=g)
        ref = render_ref_chunked(
            A_t, N_t, R_t, M_t, AO_t, wi, wo, EVAL_CHUNK, DOWNSAMPLE_FACTOR
        )
        del A_t, N_t, R_t, M_t, AO_t
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        m_box = evaluate_method(method_box_4x, A, N, R, M, AO,
                                ref, wi, wo, device, EVAL_CHUNK)
        m_lcz = evaluate_method(method_lanczos_4x, A, N, R, M, AO,
                                ref, wi, wo, device, EVAL_CHUNK)
        m_nat = evaluate_method(method_native_forced, A, N, R, M, AO,
                                ref, wi, wo, device, EVAL_CHUNK)
        m_rou = evaluate_method(method_native_routed, A, N, R, M, AO,
                                ref, wi, wo, device, EVAL_CHUNK)

        for label, m in [
            ("A1 box-4x", m_box),
            ("B1 lanczos", m_lcz),
            ("C2 native", m_nat),
            ("C2_R routed", m_rou),
        ]:
            print(
                f"  {label:14s} FLIP={m['flip']:.5f}  PSNR={m['psnr_ldr']:.2f}  "
                f"({m['proc_s']:.1f}s)"
            )

        rows.append({
            "name": name, "cls": cls.value,
            "normal_var": stats["normal_var"],
            "roughness_std": stats["roughness_std"],
            "box_flip": m_box["flip"], "box_psnr": m_box["psnr_ldr"],
            "box_proc_s": m_box["proc_s"],
            "lcz_flip": m_lcz["flip"], "lcz_psnr": m_lcz["psnr_ldr"],
            "lcz_proc_s": m_lcz["proc_s"],
            "nat_flip": m_nat["flip"], "nat_psnr": m_nat["psnr_ldr"],
            "nat_proc_s": m_nat["proc_s"],
            "rou_flip": m_rou["flip"], "rou_psnr": m_rou["psnr_ldr"],
            "rou_proc_s": m_rou["proc_s"],
        })

    # ---------- CSV ----------
    csv_path = OUT_DIR / "verify_v3.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"\nCSV -> {csv_path}")

    # ---------- Markdown ----------
    md = [
        f"# V3 端到端：native-4x + 材质路由（5 张全量）\n",
        "| 方法 | 含义 |",
        "|---|---|",
        "| A1 box-4x | 单次 4×4 box，下界基线 |",
        "| B1 lanczos-4x | 单次 Lanczos-3 4×，业界默认 |",
        "| C2 native | `factor=4, auto_route=False`，裸方法 |",
        "| **C2_R native+routed** | `factor=4, auto_route=True`，**最终落地形态** |",
        "\n## 主表 FLIP / PSNR / 时间\n",
        "| 纹理 | 类别 | A1 FLIP | B1 FLIP | C2 FLIP | **C2_R FLIP** | "
        "A1 PSNR | B1 PSNR | C2 PSNR | **C2_R PSNR** | C2 时间 | **C2_R 时间** |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in rows:
        md.append(
            f"| {r['name'].replace('_2K-PNG', '')} | {r['cls']} | "
            f"{r['box_flip']:.5f} | {r['lcz_flip']:.5f} | "
            f"{r['nat_flip']:.5f} | **{r['rou_flip']:.5f}** | "
            f"{r['box_psnr']:.2f} | {r['lcz_psnr']:.2f} | "
            f"{r['nat_psnr']:.2f} | **{r['rou_psnr']:.2f}** | "
            f"{r['nat_proc_s']:.1f}s | **{r['rou_proc_s']:.1f}s** |"
        )

    md.append("\n## 「不输 box」核验\n")
    md.append("| 纹理 | 类别 | C2 - A1 (FLIP) | **C2_R - A1 (FLIP)** | "
              "C2_R 是否 ≤ A1 |")
    md.append("|---|---|---:|---:|:---:|")
    safe = 0
    for r in rows:
        d_nat = r["nat_flip"] - r["box_flip"]
        d_rou = r["rou_flip"] - r["box_flip"]
        ok = "✓" if d_rou <= 1e-5 else "✗"
        if d_rou <= 1e-5:
            safe += 1
        md.append(
            f"| {r['name'].replace('_2K-PNG', '')} | {r['cls']} | "
            f"{d_nat:+.5f} | **{d_rou:+.5f}** | {ok} |"
        )
    md.append(f"\n**C2_R 在 {safe}/{len(rows)} 张纹理上 ≤ A1。**\n")

    geo = [r for r in rows if r["cls"] == "GEOMETRIC"]
    if geo:
        md.append("## GEOMETRIC 纹理：C2_R 是否复现 C2\n")
        md.append("| 纹理 | C2 FLIP | C2_R FLIP | 差异 |")
        md.append("|---|---:|---:|---:|")
        for r in geo:
            md.append(
                f"| {r['name'].replace('_2K-PNG', '')} | "
                f"{r['nat_flip']:.5f} | {r['rou_flip']:.5f} | "
                f"{r['rou_flip'] - r['nat_flip']:+.5f} |"
            )

    md.append("\n## vs 业界默认 Lanczos\n")
    md.append("| 纹理 | 类别 | B1 lanczos | **C2_R** | 差异 |")
    md.append("|---|---|---:|---:|---:|")
    for r in rows:
        md.append(
            f"| {r['name'].replace('_2K-PNG', '')} | {r['cls']} | "
            f"{r['lcz_flip']:.5f} | **{r['rou_flip']:.5f}** | "
            f"{r['rou_flip'] - r['lcz_flip']:+.5f} |"
        )

    md_path = OUT_DIR / "verify_v3.md"
    md_path.write_text("\n".join(md), encoding="utf-8")
    print(f"MD  -> {md_path}")
    print(f"\n=== C2_R 安全性: {safe}/{len(rows)} 张 ≤ box ===")


if __name__ == "__main__":
    main()
