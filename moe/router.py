"""
Top-K router with load-balancing loss.

Implements:
  - Top-k gating with softmax over selected experts only
  - Capacity-based token dropping
  - Auxiliary balance loss and router Z-loss
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class TopKRouter(nn.Module):
    """
    Route each token to its top-k experts.

    Parameters
    ----------
    hidden_size : int
        Input dimension.
    num_experts : int
        Number of experts (E).
    k : int
        Number of experts activated per token (default 2).
    capacity_factor : float
        Multiplier on uniform token capacity per expert.
        Set to a large value (e.g. 1000) to effectively disable dropping.
    """

    def __init__(
        self,
        hidden_size: int,
        num_experts: int,
        k: int = 2,
        capacity_factor: float = 2.0,
    ):
        super().__init__()
        self.gate = nn.Linear(hidden_size, num_experts, bias=False)
        self.num_experts = num_experts
        self.k = k
        self.capacity_factor = capacity_factor

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns
        -------
        dispatch_mask : (N, E, C)  bool
        combine_weights : (N, E, C)  float
        router_probs : (N, E)  float  (for aux loss)
        dropped_frac : float
        """
        N, D = x.shape
        E = self.num_experts
        k = self.k

        logits = self.gate(x)                            # (N, E)
        router_probs = F.softmax(logits, dim=-1)         # (N, E)  — for aux loss

        topk_vals, topk_idx = torch.topk(logits, k, dim=-1)   # (N, k)
        # Temperature-scaled softmax: keeps weights ~(1.0, 1.0)
        # at init while logit variance still breaks top-k ties
        # differently per token (so all experts get used).
        topk_weights = F.softmax(topk_vals / 2.0, dim=-1)       # (N, k)

        # ── Capacity ──────────────────────────────────────────────
        capacity = max(1, int(self.capacity_factor * N * k / E))

        # ── Build dispatch mask (vectorised, E × k loop) ──────────
        dispatch_mask = torch.zeros(N, E, capacity, device=x.device, dtype=torch.bool)
        combine_weights = torch.zeros(N, E, capacity, device=x.device, dtype=x.dtype)

        expert_offsets = torch.zeros(E, dtype=torch.long, device=x.device)

        for ki in range(k):
            e_ids = topk_idx[:, ki]                                  # (N,)
            w = topk_weights[:, ki]                                  # (N,)
            for e in range(E):
                token_mask = (e_ids == e)                            # (N,)
                n_tok = token_mask.sum()
                if n_tok == 0:
                    continue
                start = expert_offsets[e]
                end = min(start + n_tok, capacity)
                n_assign = end - start
                if n_assign > 0:
                    idx = token_mask.nonzero(as_tuple=False)[:n_assign, 0]
                    slots = torch.arange(start, end, device=x.device)
                    dispatch_mask[idx, e, slots] = True
                    combine_weights[idx, e, slots] = w[idx]
                    expert_offsets[e] = end

        total_assignments = N * k
        kept = dispatch_mask.sum()
        dropped_frac = (total_assignments - kept) / max(total_assignments, 1)

        return dispatch_mask, combine_weights, router_probs, dropped_frac


def load_balance_loss(router_probs: torch.Tensor, dispatch_mask: torch.Tensor) -> torch.Tensor:
    """
    Compute auxiliary load-balancing loss.

    L_balance = E * sum_i (f_i * P_i)

    f_i = fraction of tokens routed to expert i (from dispatch mask)
    P_i = mean router probability for expert i   (from router probs)
    """
    E = router_probs.size(-1)

    # f_i: fraction of tokens assigned to each expert
    token_per_expert = dispatch_mask.sum(dim=(0, 2))     # (E,)
    f = token_per_expert / token_per_expert.sum().clamp(min=1)

    # P_i: mean router probability per expert
    P = router_probs.mean(dim=0)                         # (E,)

    return E * (f * P).sum()


def router_z_loss(logits: torch.Tensor) -> torch.Tensor:
    """
    Router Z-loss: penalises extreme router logit magnitudes.

    L_z = (log sum_i exp(r_i))^2
    """
    logsumexp = torch.logsumexp(logits, dim=-1)          # (N,)
    return (logsumexp ** 2).mean()
