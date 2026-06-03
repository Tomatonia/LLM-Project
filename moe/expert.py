"""
Expert FFN module — architecturally identical to Qwen2MLP.

Each expert is a standard SwiGLU feed-forward block with no bias,
matching Qwen2.5-1.5B's native FFN so weights can be directly copied
during sparse upcycling.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class Expert(nn.Module):
    """
    Single MoE expert: hidden → intermediate → hidden with SiLU gating.

    Forward:  down_proj( SiLU(gate_proj(x)) * up_proj(x) )
    """

    def __init__(self, hidden_size: int, intermediate_size: int):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (*, hidden_size)
        gate = F.silu(self.gate_proj(x))
        up = self.up_proj(x)
        return self.down_proj(gate * up)
