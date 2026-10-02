# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""The pipeline-parallel prefill's own kernels (prefill/kernels) against torch, on random data:
the causal conv1d with its carry for any number of valid rows, the GDN output gate, and the state handoff
into the resident decode's layouts."""
import pytest
import torch
from loguru import logger

import ttnn
from models.experimental.qwen36_resident.prefill.pp_prefill import (
    A0,
    CONV_CH,
    EPS,
    GDN_COLS,
    TILE,
    QD,
    VD,
    Z0,
    PPPrefill,
)
from models.experimental.qwen36_resident.tests.resident_model import (
    DK,
    DV,
    HD,
    NKV,
    NV,
    Dims,
    ResidentModel,
    State,
    is_attn,
    random_weights,
)


def _bare(mesh, C):
    pp = PPPrefill.__new__(PPPrefill)
    pp.mesh, pp.n, pp.C = mesh, mesh.get_num_devices(), C
    return pp


def _dev(mesh, t, dtype, layout=ttnn.TILE_LAYOUT):
    return ttnn.from_torch(
        t,
        dtype=dtype,
        layout=layout,
        device=mesh,
        mesh_mapper=ttnn.ShardTensorToMesh(mesh, dim=0),
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )


def _cat(mesh, t):
    return ttnn.to_torch(t, mesh_composer=ttnn.ConcatMeshToTensor(mesh, dim=0)).float()


@pytest.mark.parametrize("C", [256, 512])
@pytest.mark.parametrize("mesh_device", [(1, 4)], indirect=True)
def test_conv(mesh_device, C):
    """silu(causal conv1d) of a chunk and the new carry (the 3 inputs ending at the last valid row), with
    a different number of valid rows on each chip (full, partial, inside the first tile, 1)"""
    n = mesh_device.get_num_devices()
    pp = _bare(mesh_device, C)
    g = torch.Generator().manual_seed(0)
    x = torch.randn(n, 1, C, GDN_COLS, generator=g).bfloat16().float()
    carry = torch.zeros(n, 1, TILE, CONV_CH)
    carry[:, 0, :3] = torch.randn(n, 3, CONV_CH, generator=g).bfloat16().float()
    taps = torch.randn(CONV_CH, 4, generator=g).bfloat16().float()
    tt = torch.zeros(n, 1, 4 * TILE, CONV_CH)
    for k in range(4):
        tt[:, 0, k * TILE] = taps[:, k]
    valid = [C, C - 5, 30, 1][:n]
    ctrl = torch.zeros(n, 1, 1, 8, dtype=torch.int32)
    for p in range(n):
        ctrl[p, 0, 0, :2] = torch.tensor([valid[p], 1])
    pp.carry = {0: _dev(mesh_device, carry, ttnn.bfloat16)}
    pp.w = [dict(taps=_dev(mesh_device, tt, ttnn.bfloat16))]
    pp.q_t = _dev(mesh_device, torch.zeros(n, C, QD), ttnn.bfloat16)
    pp.k_t = _dev(mesh_device, torch.zeros(n, C, QD), ttnn.bfloat16)
    pp.v_t = _dev(mesh_device, torch.zeros(n, C, VD), ttnn.bfloat16)
    pp.ctrl = _dev(mesh_device, ctrl, ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT)
    pp._conv_program()
    pp._conv(_dev(mesh_device, x, ttnn.bfloat16), 0)
    y = torch.cat([_cat(mesh_device, t) for t in (pp.q_t, pp.k_t, pp.v_t)], dim=-1)[:, None]
    new_carry = _cat(mesh_device, pp.carry[0])
    for p in range(n):
        win = torch.cat([carry[p, 0, :3], x[p, 0, :, :CONV_CH]])
        ref = torch.nn.functional.silu(sum(win[k : k + C] * taps[:, k] for k in range(4)))
        y_err = float((y[p, 0] - ref).abs().max())
        c_err = float((new_carry[p, 0, :3] - win[valid[p] : valid[p] + 3]).abs().max())
        logger.info(f"chip {p}, {valid[p]} valid rows: y max err {y_err:.4f}, carry max err {c_err:.4f}")
        assert y_err < 0.05 * float(ref.abs().max()) and c_err == 0


@pytest.mark.parametrize("C", [256, 1024])
@pytest.mark.parametrize("mesh_device", [(1, 4)], indirect=True)
def test_gates(mesh_device, C):
    """g = neg_a * softplus(a + dt), beta = sigmoid(b) from the projection's a | b columns, masked to the
    valid rows (zero on a chip whose chunk does not update)"""
    n = mesh_device.get_num_devices()
    pp = _bare(mesh_device, C)
    g = torch.Generator().manual_seed(0)
    p = torch.randn(n, 1, C, GDN_COLS, generator=g).bfloat16().float()
    dt, neg_a = torch.randn(n, NV, generator=g), -torch.rand(n, NV, generator=g) * 4
    row0 = (
        lambda v: torch.cat([v, torch.zeros(n, 2 * TILE - NV)], -1)[:, None, None].expand(n, 1, TILE, 2 * TILE).clone()
    )
    valid, update = [C, C - 5, 30, C][:n], [1, 1, 1, 0][:n]
    ctrl = torch.zeros(n, 1, 1, 8, dtype=torch.int32)
    for c in range(n):
        ctrl[c, 0, 0, :2] = torch.tensor([valid[c], update[c]])
    pp.w = [dict(dt=_dev(mesh_device, row0(dt), ttnn.float32), neg_a=_dev(mesh_device, row0(neg_a), ttnn.float32))]
    pp.ctrl = _dev(mesh_device, ctrl, ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT)
    pp.g_t = _dev(mesh_device, torch.zeros(n, C, NV), ttnn.float32)
    pp.beta_t = _dev(mesh_device, torch.zeros(n, C, NV), ttnn.float32)
    pp._gates(_dev(mesh_device, p, ttnn.bfloat16), 0)
    got_g, got_b = _cat(mesh_device, pp.g_t), _cat(mesh_device, pp.beta_t)
    a, b = p[:, 0, :, A0 : A0 + NV], p[:, 0, :, A0 + NV : A0 + 2 * NV]
    mask = torch.zeros(n, C, 1)
    for c in range(n):
        mask[c, : valid[c]] = float(update[c])
    ref_g = neg_a[:, None] * torch.nn.functional.softplus(a + dt[:, None]) * mask
    ref_b = torch.sigmoid(b) * mask
    errs = (float((got_g - ref_g).abs().max()), float((got_b - ref_b).abs().max()))
    logger.info(f"gates max err g {errs[0]:.4f} (max {float(ref_g.abs().max()):.2f}), beta {errs[1]:.4f}")
    assert errs[0] < 0.02 * float(ref_g.abs().max()) and errs[1] < 0.01


@pytest.mark.parametrize("C", [256, 1024])
@pytest.mark.parametrize("mesh_device", [(1, 4)], indirect=True)
def test_gate(mesh_device, C):
    """rmsnorm over each head of the head-major delta-rule output times silu(z), token-major"""
    n = mesh_device.get_num_devices()
    pp = _bare(mesh_device, C)
    g = torch.Generator().manual_seed(0)
    o = torch.randn(n, NV, C, DV, generator=g)
    p = torch.randn(n, 1, C, GDN_COLS, generator=g).bfloat16().float()
    pp.gate_y = _dev(mesh_device, torch.zeros(n, 1, C, VD), ttnn.bfloat16)
    got = _cat(mesh_device, pp._gate(_dev(mesh_device, o, ttnn.float32), _dev(mesh_device, p, ttnn.bfloat16)))
    on = o * torch.rsqrt(o.pow(2).mean(-1, keepdim=True) + EPS)
    ref = on.permute(0, 2, 1, 3).reshape(n, 1, C, VD) * torch.nn.functional.silu(p[..., Z0 : Z0 + VD])
    err = float((got - ref).abs().max())
    logger.info(f"gate max err {err:.4f} (max {float(ref.abs().max()):.2f})")
    assert err < 0.02 * float(ref.abs().max())


@pytest.mark.parametrize("T", [100, 1000])
@pytest.mark.parametrize("device_params", [{"fabric_config": ttnn.FabricConfig.FABRIC_1D}], indirect=True)
@pytest.mark.parametrize("mesh_device", [(1, 4)], indirect=True)
def test_handoff(mesh_device, T):
    """GDN states and KV caches of 16 layers (4 per chip) moved chip to chip into the decode's
    TP-sharded layouts: bit-exact against the host conversion"""
    n, layers, block = mesh_device.get_num_devices(), 16, 64
    d = Dims(n, mesh_device.dram_grid_size().x)
    n_attn = layers // 4
    w = random_weights(d, gdn_copies=layers - n_attn, attn_copies=n_attn, mlp_copies=1)
    model = ResidentModel(mesh_device, d, w, State(d, layers - n_attn, n_attn, 0, T + 64, zero=True), layers, 4)
    pp = _bare(mesh_device, 256)
    pp.layers, pp.Ls, pp.block = layers, layers // n, block
    pp.kinds = [is_attn(j, 4) for j in range(pp.Ls)]
    pp.blocks = (T + block - 1) // block
    g = torch.Generator().manual_seed(0)
    pp.S, pp.carry, pp.K, pp.V = {}, {}, {}, {}
    for j, attn in enumerate(pp.kinds):
        if attn:
            kv = lambda: torch.randn(n * (pp.blocks + 4), NKV, block, HD, generator=g)
            pp.K[j], pp.V[j] = _dev(mesh_device, kv(), ttnn.bfloat8_b), _dev(mesh_device, kv(), ttnn.bfloat8_b)
        else:
            pp.S[j] = _dev(mesh_device, torch.randn(n, NV, DK, DV, generator=g), ttnn.float32)
            pp.carry[j] = _dev(mesh_device, torch.randn(n, 1, TILE, CONV_CH, generator=g), ttnn.bfloat16)
    pp.handoff(model, T)
    st = pp.export_state(d, T, T + 64)
    ones = State(d, layers - n_attn, n_attn, T, T + 64, zero=True)
    for c in range(n_attn):
        for chip in range(n):
            ones.K[c][chip][:T] = 1.0
    hist_mask = model._hist_host(st) != 0
    kv_mask = model._kv_host(ones, "K") != 0
    errs = dict(
        state=float((_cat(mesh_device, model.state_t) - model._state_host(st)).abs().max()),
        hist=float((_cat(mesh_device, model.hist_t)[hist_mask] - model._hist_host(st)[hist_mask]).abs().max()),
        K=float((_cat(mesh_device, model.K_t)[kv_mask] - model._kv_host(st, "K")[kv_mask]).abs().max()),
        V=float((_cat(mesh_device, model.V_t)[kv_mask] - model._kv_host(st, "V")[kv_mask]).abs().max()),
    )
    logger.info(f"handoff max abs diff {errs}")
    assert all(v == 0 for v in errs.values()), errs
