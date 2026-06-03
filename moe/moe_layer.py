"""
MoE Transformer Block — replaces the dense FFN in a Qwen2DecoderLayer.

Supports two modes:
  - ``MoEBlock``: E routed experts, top-k gating, partitioned dim coverage
  - ``MoEBlockWithShared``: 1 always-active shared expert + E routed experts
"""

import torch
import torch.nn as nn

from .expert import Expert
from .router import TopKRouter, load_balance_loss, router_z_loss


# ═══════════════════════════════════════════════════════════════════════════
# Shared-expert MoE block
# ═══════════════════════════════════════════════════════════════════════════

class MoEBlockWithShared(nn.Module):
    """
    One shared expert (always active) + E routed experts (top-k).

    Output = shared(x) + sum_i(w_i × routed_i(x))

    The shared expert covers dense-dims [0, shared_is).
    Routed experts partition the remaining [shared_is, dense_is) with
    staggered overlapping slices so any top-2 pair covers the gap.
    """

    def __init__(
        self,
        hidden_size: int,
        shared_intermediate_size: int,
        routed_intermediate_size: int,
        num_routed_experts: int = 3,
        k: int = 2,
        capacity_factor: float = 2.0,
    ):
        super().__init__()
        self.shared = Expert(hidden_size, shared_intermediate_size)
        self.router = TopKRouter(hidden_size, num_routed_experts, k, capacity_factor)
        self.routed = nn.ModuleList([
            Expert(hidden_size, routed_intermediate_size)
            for _ in range(num_routed_experts)
        ])
        self.num_routed = num_routed_experts
        self.k = k
        self._aux = {
            "balance_loss": torch.tensor(0.0),
            "z_loss": torch.tensor(0.0),
            "dropped_frac": torch.tensor(0.0),
        }

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, S, D = x.shape
        x_flat = x.reshape(-1, D)

        # Shared expert — every token, no routing
        y = self.shared(x_flat)

        # Routed experts — top-k gating
        dm, cw, rp, df = self.router(x_flat)

        for e in range(self.num_routed):
            token_mask = dm[:, e, :].any(dim=-1)
            idx = token_mask.nonzero(as_tuple=True)[0]
            if idx.numel() == 0:
                continue
            expert_out = self.routed[e](x_flat[idx])
            w = cw[idx, e, :].sum(dim=-1, keepdim=True)
            y[idx] += w * expert_out

        y = y.view(B, S, D)

        self._aux = {
            "balance_loss": load_balance_loss(rp, dm).detach(),
            "z_loss": router_z_loss(self.router.gate(x_flat)).detach(),
            "dropped_frac": torch.tensor(df, device=x.device),
        }
        return y

    def get_aux_losses(self):
        return self._aux

    def init_from_dense(self, dense_ffn: nn.Module, noise_std: float = 0.0):
        d_is = dense_ffn.gate_proj.out_features
        s_is = self.shared.gate_proj.out_features
        r_is = self.routed[0].gate_proj.out_features

        routable_start = s_is         # first dim after shared
        routable_len = d_is - s_is    # dims to partition across routed experts

        with torch.no_grad():
            # Shared expert: dense [0, s_is)
            _copy_slice(self.shared, dense_ffn, 0, s_is, s_is, noise_std)

            # Routed experts: evenly-spaced staggered slices.
            # Stride = routable_len / num_routed so adjacent experts
            # (including wrap) overlap to cover the full range with
            # any top-2 pair.
            stride = routable_len // self.num_routed
            for i, expert in enumerate(self.routed):
                start = routable_start + i * stride
                _copy_wrapped_slice(expert, dense_ffn, start, start + r_is,
                                    routable_start, d_is, r_is, noise_std)


# ═══════════════════════════════════════════════════════════════════════════
# Routed-only MoE block (original, no shared expert)
# ═══════════════════════════════════════════════════════════════════════════

class MoEBlock(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        num_experts: int = 8,
        k: int = 2,
        capacity_factor: float = 2.0,
    ):
        super().__init__()
        self.router = TopKRouter(hidden_size, num_experts, k, capacity_factor)
        self.experts = nn.ModuleList([
            Expert(hidden_size, intermediate_size)
            for _ in range(num_experts)
        ])
        self.num_experts = num_experts
        self.k = k
        self._aux = {
            "balance_loss": torch.tensor(0.0),
            "z_loss": torch.tensor(0.0),
            "dropped_frac": torch.tensor(0.0),
        }

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, S, D = x.shape
        x_flat = x.reshape(-1, D)

        dm, cw, rp, df = self.router(x_flat)
        y = torch.zeros_like(x_flat)

        for e in range(self.num_experts):
            token_mask = dm[:, e, :].any(dim=-1)
            idx = token_mask.nonzero(as_tuple=True)[0]
            if idx.numel() == 0:
                continue
            expert_out = self.experts[e](x_flat[idx])
            w = cw[idx, e, :].sum(dim=-1, keepdim=True)
            y[idx] += w * expert_out

        y = y.view(B, S, D)

        self._aux = {
            "balance_loss": load_balance_loss(rp, dm).detach(),
            "z_loss": router_z_loss(self.router.gate(x_flat)).detach(),
            "dropped_frac": torch.tensor(df, device=x.device),
        }
        return y

    def get_aux_losses(self):
        return self._aux

    def init_from_dense(self, dense_ffn: nn.Module, noise_std: float = 0.0):
        e_is = self.experts[0].gate_proj.out_features
        d_is = dense_ffn.gate_proj.out_features

        slices = [
            (0,           min(e_is, d_is)),
            (e_is,        min(2 * e_is, d_is)),
            (0,           min(e_is, d_is)),
            (e_is,        min(2 * e_is, d_is)),
        ]

        with torch.no_grad():
            for expert, (start, end) in zip(self.experts, slices):
                _copy_slice(expert, dense_ffn, start, end, e_is, noise_std)


# ═══════════════════════════════════════════════════════════════════════════
# Weight-copy helpers
# ═══════════════════════════════════════════════════════════════════════════

def _copy_slice(expert, dense_ffn, start, end, e_is, noise_std):
    n = end - start
    if n > 0:
        expert.gate_proj.weight[:n].copy_(dense_ffn.gate_proj.weight[start:end])
        expert.up_proj.weight[:n].copy_(dense_ffn.up_proj.weight[start:end])
        expert.down_proj.weight[:, :n].copy_(dense_ffn.down_proj.weight[:, start:end])
    if n < e_is:
        expert.gate_proj.weight[n:].zero_()
        expert.up_proj.weight[n:].zero_()
        expert.down_proj.weight[:, n:].zero_()
    if noise_std > 0:
        for p in expert.parameters():
            p.add_(torch.randn_like(p) * noise_std)


def _copy_wrapped_slice(expert, dense_ffn, start, end, boundary, d_is, e_is, noise_std):
    """Copy [start, end) from dense FFN, wrapping at *boundary* back to *boundary*."""
    taken = 0

    # Segment from start to min(end, d_is)
    seg_end = min(end, d_is)
    if seg_end > start:
        n = seg_end - start
        expert.gate_proj.weight[taken:taken + n].copy_(
            dense_ffn.gate_proj.weight[start:seg_end])
        expert.up_proj.weight[taken:taken + n].copy_(
            dense_ffn.up_proj.weight[start:seg_end])
        expert.down_proj.weight[:, taken:taken + n].copy_(
            dense_ffn.down_proj.weight[:, start:seg_end])
        taken += n

    # Wrap-around segment from boundary onwards
    remaining = (end - start) - taken
    if remaining > 0:
        expert.gate_proj.weight[taken:taken + remaining].copy_(
            dense_ffn.gate_proj.weight[boundary:boundary + remaining])
        expert.up_proj.weight[taken:taken + remaining].copy_(
            dense_ffn.up_proj.weight[boundary:boundary + remaining])
        expert.down_proj.weight[:, taken:taken + remaining].copy_(
            dense_ffn.down_proj.weight[:, boundary:boundary + remaining])
        taken += remaining

    if taken < e_is:
        expert.gate_proj.weight[taken:].zero_()
        expert.up_proj.weight[taken:].zero_()
        expert.down_proj.weight[:, taken:].zero_()

    if noise_std > 0:
        for p in expert.parameters():
            p.add_(torch.randn_like(p) * noise_std)


# ═══════════════════════════════════════════════════════════════════════════
# Aux-loss collection
# ═══════════════════════════════════════════════════════════════════════════

def collect_aux_losses(model) -> dict[str, torch.Tensor]:
    balances, zs, drops = [], [], []
    for layer in model.model.layers:
        moe = layer.mlp
        if not hasattr(moe, "get_aux_losses"):
            continue
        aux = moe.get_aux_losses()
        balances.append(aux["balance_loss"])
        zs.append(aux["z_loss"])
        drops.append(aux["dropped_frac"])
    if not balances:
        return {
            "balance_loss": torch.tensor(0.0),
            "z_loss": torch.tensor(0.0),
            "dropped_frac": torch.tensor(0.0),
        }
    return {
        "balance_loss": torch.stack(balances).mean(),
        "z_loss": torch.stack(zs).mean(),
        "dropped_frac": torch.stack(drops).mean(),
    }


# ═══════════════════════════════════════════════════════════════════════════
# Conversion entry-point
# ═══════════════════════════════════════════════════════════════════════════

def convert_to_moe(model, moe_config: dict):
    hidden_size = moe_config["hidden_size"]
    moe_layers = moe_config["moe_layers"]
    capacity_factor = moe_config.get("capacity_factor", 2.0)
    k = moe_config["k"]
    noise_std = moe_config.get("noise_std", 0.0)
    use_shared = "shared_intermediate_size" in moe_config

    layers = model.model.layers
    for idx in moe_layers:
        dense_mlp = layers[idx].mlp
        device = dense_mlp.gate_proj.weight.device
        dtype = next(dense_mlp.parameters()).dtype

        if use_shared:
            block = MoEBlockWithShared(
                hidden_size=hidden_size,
                shared_intermediate_size=moe_config["shared_intermediate_size"],
                routed_intermediate_size=moe_config["routed_intermediate_size"],
                num_routed_experts=moe_config["num_experts"],
                k=k,
                capacity_factor=capacity_factor,
            )
        else:
            block = MoEBlock(
                hidden_size=hidden_size,
                intermediate_size=moe_config["intermediate_size"],
                num_experts=moe_config["num_experts"],
                k=k,
                capacity_factor=capacity_factor,
            )

        block.init_from_dense(dense_mlp, noise_std=noise_std)
        block = block.to(device=device, dtype=dtype)
        layers[idx].mlp = block
        del dense_mlp

    import gc
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    kind = "shared+MoE" if use_shared else "MoE"
    n = moe_config["num_experts"]
    print(f"Converted {len(moe_layers)} layers to {kind} "
          f"({n} routed experts, k={k}).")
    return model
