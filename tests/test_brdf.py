"""brdf.py 单元测试。"""
import math

import numpy as np
import torch

from pbr_compress.brdf import evaluate_brdf, sample_directions


def _basic_inputs(device="cpu"):
    A = torch.full((1, 1, 1, 3), 0.5, device=device)
    N = torch.zeros((1, 1, 1, 3), device=device)
    N[..., 2] = 1.0
    R = torch.full((1, 1, 1, 1), 0.3, device=device)
    M = torch.zeros((1, 1, 1, 1), device=device)
    AO = torch.ones((1, 1, 1, 1), device=device)
    wi = torch.zeros((1, 1, 1, 3), device=device); wi[..., 2] = 1.0
    wo = torch.zeros((1, 1, 1, 3), device=device); wo[..., 2] = 1.0
    return A, N, R, M, AO, wi, wo


def test_brdf_shape_and_finite():
    out = evaluate_brdf(*_basic_inputs())
    assert out.shape == (1, 1, 1, 3)
    assert torch.all(torch.isfinite(out))
    assert torch.all(out >= 0)


def test_brdf_zero_albedo_dielectric_still_has_specular():
    """非金属 + albedo=0：diffuse=0，但 F0=0.04 仍贡献 specular。"""
    A, N, R, M, AO, wi, wo = _basic_inputs()
    A = torch.zeros_like(A)
    out = evaluate_brdf(A, N, R, M, AO, wi, wo)
    assert torch.all(out > 0)


def test_brdf_metal_no_diffuse():
    """metallic=1 时 diffuse=0；只剩 specular 项，且 F0=albedo。"""
    A, N, R, M, AO, wi, wo = _basic_inputs()
    A_metal = torch.full_like(A, 1.0)
    M_metal = torch.ones_like(M)
    out_metal = evaluate_brdf(A_metal, N, R, M_metal, AO, wi, wo)

    A_zero = torch.zeros_like(A)
    out_metal_zero = evaluate_brdf(A_zero, N, R, M_metal, AO, wi, wo)
    # albedo=1 的金属 specular 强于 albedo=0 的金属
    assert (out_metal > out_metal_zero).all()


def test_brdf_ao_scales_output():
    A, N, R, M, AO, wi, wo = _basic_inputs()
    full = evaluate_brdf(A, N, R, M, AO, wi, wo)
    half = evaluate_brdf(A, N, R, M, AO * 0.5, wi, wo)
    assert torch.allclose(half, full * 0.5, atol=1e-6)


def test_brdf_grad_flows_to_albedo():
    A, N, R, M, AO, wi, wo = _basic_inputs()
    A = A.clone().detach().requires_grad_(True)
    out = evaluate_brdf(A, N, R, M, AO, wi, wo)
    out.sum().backward()
    assert A.grad is not None
    assert torch.all(torch.isfinite(A.grad))
    assert (A.grad.abs() > 0).any()


def test_sample_directions_in_hemisphere():
    wi, wo = sample_directions(64, "cpu")
    assert wi.shape == (64, 3) and wo.shape == (64, 3)
    assert torch.all(wi[..., 2] >= 0)
    assert torch.all(wo[..., 2] >= 0)
    norms_wi = torch.linalg.norm(wi, dim=-1)
    norms_wo = torch.linalg.norm(wo, dim=-1)
    assert torch.allclose(norms_wi, torch.ones_like(norms_wi), atol=1e-5)
    assert torch.allclose(norms_wo, torch.ones_like(norms_wo), atol=1e-5)


def test_sample_directions_seed_repeatable():
    g1 = torch.Generator(device="cpu"); g1.manual_seed(123)
    g2 = torch.Generator(device="cpu"); g2.manual_seed(123)
    wi1, wo1 = sample_directions(16, "cpu", generator=g1)
    wi2, wo2 = sample_directions(16, "cpu", generator=g2)
    assert torch.equal(wi1, wi2)
    assert torch.equal(wo1, wo2)


def test_cosine_distribution_average_z_close_to_2_3():
    """半球上余弦加权采样的 ``E[z] = 2/3``，大量样本应接近。"""
    g = torch.Generator(device="cpu"); g.manual_seed(0)
    wi, _ = sample_directions(20000, "cpu", generator=g)
    mean_z = float(wi[..., 2].mean())
    assert math.isclose(mean_z, 2.0 / 3.0, abs_tol=2e-2)
