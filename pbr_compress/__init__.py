"""PBR 纹理 2× 离线压缩管线。

实现严格按照 doc.md 的规格分模块组织：

- :mod:`pbr_compress.filters` :   ``box_downsample_2x``、``dpid_downsample_2x``
- :mod:`pbr_compress.analytic`:   Normal / Roughness(LEAN) / Metallic / AO
- :mod:`pbr_compress.brdf`    :   GGX + Schlick + Smith BRDF 评估、方向采样
- :mod:`pbr_compress.render`  :   高分 footprint 平均渲染 / 低分直接渲染
- :mod:`pbr_compress.optimize`:   Albedo 可微优化主循环
- :mod:`pbr_compress.pipeline`:   ``compress_pbr_textures`` 顶层入口
- :mod:`pbr_compress.metrics` :   贴图层与渲染层质量指标
- :mod:`pbr_compress.io`      :   PNG/TIFF/EXR 读写、sRGB / 法线编解码
- :mod:`pbr_compress.cli`     :   命令行入口
"""
from .filters import box_downsample_2x, dpid_downsample_2x
from .analytic import (
    downsample_normal,
    downsample_roughness_lean,
    downsample_metallic,
    downsample_ao,
)
from .brdf import evaluate_brdf, sample_directions
from .render import render_hr_footprint_avg, render_lr
from .optimize import compute_loss, optimize_albedo
from .pipeline import compress_pbr_textures
from .metrics import (
    psnr,
    ssim_simple,
    normal_angle_error,
    evaluate_render_quality,
)

__all__ = [
    "box_downsample_2x",
    "dpid_downsample_2x",
    "downsample_normal",
    "downsample_roughness_lean",
    "downsample_metallic",
    "downsample_ao",
    "evaluate_brdf",
    "sample_directions",
    "render_hr_footprint_avg",
    "render_lr",
    "compute_loss",
    "optimize_albedo",
    "compress_pbr_textures",
    "psnr",
    "ssim_simple",
    "normal_angle_error",
    "evaluate_render_quality",
]

__version__ = "0.1.0"
