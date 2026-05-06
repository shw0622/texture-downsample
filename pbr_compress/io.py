"""图像 IO + 颜色空间 / 法线编解码工具。

对应 doc.md 第 8 节注意事项 1 / 2 的工程实现：

- LDR PBR 通道：磁盘 ``[0, 255]`` 整型 → ``[0, 1]`` linear float32
- HDR：通过 ``imageio`` + ``tifffile`` 读写浮点图（TIFF / EXR / HDR）
- Albedo：默认按 sRGB 解码到 linear；输出再编码回 sRGB
- Normal：磁盘 ``(n+1)/2 ∈ [0, 1]`` ↔ 内部 ``[-1, 1]`` 单位向量
"""
from __future__ import annotations

from pathlib import Path
from typing import Union

import imageio.v3 as iio
import numpy as np

PathLike = Union[str, Path]

__all__ = [
    "srgb_to_linear",
    "linear_to_srgb",
    "decode_normal",
    "encode_normal",
    "load_image",
    "save_image",
    "load_albedo",
    "save_albedo",
    "load_normal",
    "save_normal",
    "load_scalar",
    "save_scalar",
]


# ---------------------------------------------------------------------------
# sRGB <-> Linear（标准分段函数）
# ---------------------------------------------------------------------------


def srgb_to_linear(x: np.ndarray) -> np.ndarray:
    x = np.clip(x, 0.0, 1.0)
    a = 0.055
    threshold = 0.04045
    low = x / 12.92
    high = ((x + a) / (1.0 + a)) ** 2.4
    return np.where(x <= threshold, low, high).astype(x.dtype, copy=False)


def linear_to_srgb(x: np.ndarray) -> np.ndarray:
    x = np.clip(x, 0.0, 1.0)
    a = 0.055
    threshold = 0.0031308
    low = x * 12.92
    high = (1.0 + a) * (x ** (1.0 / 2.4)) - a
    out = np.where(x <= threshold, low, high)
    return np.clip(out, 0.0, 1.0).astype(x.dtype, copy=False)


# ---------------------------------------------------------------------------
# Normal 编解码
# ---------------------------------------------------------------------------


def decode_normal(rgb01: np.ndarray) -> np.ndarray:
    """``[0, 1]`` -> ``[-1, 1]`` 单位向量（含再归一化以容错）。"""
    n = rgb01 * 2.0 - 1.0
    length = np.linalg.norm(n, axis=-1, keepdims=True)
    return n / np.maximum(length, 1e-8)


def encode_normal(n: np.ndarray) -> np.ndarray:
    """``[-1, 1]`` -> ``[0, 1]``。"""
    return np.clip(n * 0.5 + 0.5, 0.0, 1.0)


# ---------------------------------------------------------------------------
# 通用 IO
# ---------------------------------------------------------------------------


def _to_float(img: np.ndarray) -> np.ndarray:
    if img.dtype == np.uint8:
        return img.astype(np.float32) / 255.0
    if img.dtype == np.uint16:
        return img.astype(np.float32) / 65535.0
    return img.astype(np.float32)


def _to_uint8(img: np.ndarray) -> np.ndarray:
    return (np.clip(img, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)


def _to_uint16(img: np.ndarray) -> np.ndarray:
    return (np.clip(img, 0.0, 1.0) * 65535.0 + 0.5).astype(np.uint16)


def load_image(path: PathLike) -> np.ndarray:
    """读图为 ``float32``，按 dtype 自动归一化到 ``[0, 1]``。

    返回 shape ``(H, W, C)``，其中 ``C`` 由原图决定（灰度图 → ``C=1``，
    带 alpha 通道的图自动丢弃 alpha）。
    """
    img = iio.imread(str(path))
    if img.ndim == 2:
        img = img[..., None]
    if img.ndim == 3 and img.shape[-1] == 4:
        img = img[..., :3]
    return _to_float(img)


def save_image(path: PathLike, img: np.ndarray, bit_depth: int = 8) -> None:
    """保存 ``[0, 1]`` float 图像。

    Parameters
    ----------
    bit_depth : int
        ``8`` / ``16`` 对应 LDR PNG；``>=32`` 或扩展名为 ``.exr/.hdr/.tif``
        时按浮点 HDR 写入。
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    suffix = path.suffix.lower()
    is_hdr = suffix in (".exr", ".hdr") or bit_depth >= 32

    # 单通道写灰度图
    out = img
    if out.ndim == 3 and out.shape[-1] == 1:
        out = out[..., 0]

    if is_hdr:
        iio.imwrite(str(path), out.astype(np.float32))
    elif bit_depth == 16:
        iio.imwrite(str(path), _to_uint16(out))
    else:
        iio.imwrite(str(path), _to_uint8(out))


# ---------------------------------------------------------------------------
# 高层语义 IO
# ---------------------------------------------------------------------------


def load_albedo(path: PathLike, srgb: bool = True) -> np.ndarray:
    """读 albedo，输出 ``(H, W, 3)`` linear ``[0, 1]``。

    Parameters
    ----------
    srgb : bool
        磁盘是否为 sRGB 编码（默认 True，需要解码到 linear）。
    """
    img = load_image(path)
    if img.shape[-1] == 1:
        img = np.repeat(img, 3, axis=-1)
    if srgb:
        img = srgb_to_linear(img)
    return img.astype(np.float32, copy=False)


def save_albedo(
    path: PathLike, A: np.ndarray, srgb: bool = True, bit_depth: int = 8
) -> None:
    out = linear_to_srgb(A) if srgb else A
    save_image(path, out, bit_depth=bit_depth)


def load_normal(path: PathLike) -> np.ndarray:
    """读法线，磁盘 ``[0, 1]`` 编码，输出 ``[-1, 1]`` 单位向量。"""
    img = load_image(path)
    if img.shape[-1] < 3:
        raise ValueError(f"normal map 必须是 RGB，实际 channels={img.shape[-1]}")
    return decode_normal(img[..., :3]).astype(np.float32, copy=False)


def save_normal(path: PathLike, N: np.ndarray, bit_depth: int = 8) -> None:
    save_image(path, encode_normal(N), bit_depth=bit_depth)


def load_scalar(path: PathLike) -> np.ndarray:
    """读单通道贴图（roughness/metallic/AO），输出 ``(H, W, 1)``。

    若磁盘是 RGB 则取 R 通道。
    """
    img = load_image(path)
    if img.shape[-1] >= 2:
        img = img[..., 0:1]
    return img.astype(np.float32, copy=False)


def save_scalar(path: PathLike, x: np.ndarray, bit_depth: int = 8) -> None:
    save_image(path, x, bit_depth=bit_depth)
