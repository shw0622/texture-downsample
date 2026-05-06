"""GGX + Schlick + Smith BRDF 评估及上半球余弦加权方向采样（PyTorch）。

对应 doc.md 第 4.1 与 4.2 节。所有张量末维为 channel/3D 方向，前面的维度
（H, W, B 等）由调用方任意广播。
"""
from __future__ import annotations

import math
from typing import Optional, Tuple

import torch
import torch.nn.functional as F

__all__ = ["evaluate_brdf", "sample_directions"]


def evaluate_brdf(
    A: torch.Tensor,
    N: torch.Tensor,
    R: torch.Tensor,
    M: torch.Tensor,
    AO: torch.Tensor,
    wi: torch.Tensor,
    wo: torch.Tensor,
) -> torch.Tensor:
    """逐 texel × 视-光对的 BRDF 出射 radiance（假设 ``Li = 1``）。

    返回的张量已乘 ``NdotL`` 与 ``AO``，可直接进入 loss / 平均。

    Parameters
    ----------
    A  : ``(..., 3)`` albedo，``[0, 1]``
    N  : ``(..., 3)`` 单位法线（切空间，``z`` 朝外）
    R  : ``(..., 1)`` roughness，``[0, 1]``
    M  : ``(..., 1)`` metallic，``{0, 1}``
    AO : ``(..., 1)`` ambient occlusion，``[0, 1]``
    wi, wo : ``(..., 3)`` 单位向量，``z > 0``

    Returns
    -------
    Tensor, ``(..., 3)``
    """
    H = F.normalize(wi + wo, dim=-1)
    NdotL = (N * wi).sum(-1, keepdim=True).clamp(1e-4, 1.0)
    NdotV = (N * wo).sum(-1, keepdim=True).clamp(1e-4, 1.0)
    NdotH = (N * H).sum(-1, keepdim=True).clamp(1e-4, 1.0)
    VdotH = (wo * H).sum(-1, keepdim=True).clamp(1e-4, 1.0)

    alpha = R * R
    alpha2 = alpha * alpha

    # GGX NDF
    denom = NdotH * NdotH * (alpha2 - 1.0) + 1.0
    D = alpha2 / (math.pi * denom * denom + 1e-8)

    # Smith G (Schlick-GGX 近似，用 R 计算 k)
    k = (R + 1.0) ** 2 / 8.0
    G1_V = NdotV / (NdotV * (1.0 - k) + k)
    G1_L = NdotL / (NdotL * (1.0 - k) + k)
    G = G1_V * G1_L

    # Fresnel (Schlick)：金属时 F0 为 albedo，非金属为 0.04
    F0 = 0.04 * (1.0 - M) + A * M
    Fr = F0 + (1.0 - F0) * (1.0 - VdotH).clamp(0.0, 1.0) ** 5

    f_s = D * G * Fr / (4.0 * NdotV * NdotL + 1e-8)
    f_d = A * (1.0 - M) / math.pi                    # 金属无 diffuse

    return (f_d + f_s) * NdotL * AO


def sample_directions(
    batch_size: int,
    device: torch.device | str,
    generator: Optional[torch.Generator] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """在上半球余弦加权采样 ``wi`` 与 ``wo``（切空间）。

    Parameters
    ----------
    batch_size : int
        采样的视-光对数。
    device : torch.device | str
        生成张量所在设备。
    generator : torch.Generator | None
        若提供则使用对应设备的 RNG（用于复现）。

    Returns
    -------
    wi, wo : 各 ``(B, 3)`` 单位向量
    """
    device = torch.device(device)

    def cosine_hemisphere(n: int) -> torch.Tensor:
        if generator is not None:
            u1 = torch.rand(n, 1, device=device, generator=generator)
            u2 = torch.rand(n, 1, device=device, generator=generator)
        else:
            u1 = torch.rand(n, 1, device=device)
            u2 = torch.rand(n, 1, device=device)
        r = torch.sqrt(u1)
        phi = 2.0 * math.pi * u2
        x = r * torch.cos(phi)
        y = r * torch.sin(phi)
        z = torch.sqrt((1.0 - u1).clamp(0.0, 1.0))
        return torch.cat([x, y, z], dim=-1)             # (n, 3)

    wi = cosine_hemisphere(batch_size)
    wo = cosine_hemisphere(batch_size)
    return wi, wo
