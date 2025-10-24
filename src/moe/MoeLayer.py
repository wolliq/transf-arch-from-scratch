import torch
from torch import nn
import torch.nn.functional as F


class SwiGLU(nn.Module):
    def __init__(self, hidden_dim, intermediate_dim):
        super().__init__()
        self.gate = nn.Linear(hidden_dim, intermediate_dim)
        self.up = nn.Linear(hidden_dim, intermediate_dim)
        self.down = nn.Linear(intermediate_dim, hidden_dim)
        self.act = nn.SiLU()

    def forward(self, x):
        x = self.act(self.gate(x)) * self.up(x)
        x = self.down(x)
        return x


class MoELayer(nn.Module):
    def __init__(self, hidden_dim, intermediate_dim, num_experts, top_k=2):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        # Create expert networks
        self.experts = nn.ModuleList([
            SwiGLU(hidden_dim, intermediate_dim) for _ in range(num_experts)
        ])
        self.router = nn.Linear(hidden_dim, num_experts)

    def forward(self, hidden_states):
        batch_size, seq_len, hidden_dim = hidden_states.shape

        # Reshape for expert processing, then compute routing probabilities
        hidden_states_reshaped = hidden_states.view(-1, hidden_dim)
        # shape of router_logits: (batch_size * seq_len, num_experts)
        router_logits = self.router(hidden_states_reshaped)

        # Select top-k experts, then softmax output probabilities will sum to 1
        # output shape: (batch_size * seq_len, k)
        top_k_logits, top_k_indices = torch.topk(router_logits, self.top_k, dim=-1)
        top_k_probs = F.softmax(top_k_logits, dim=-1)

        # Allocate output tensor
        output = torch.zeros(batch_size * seq_len, hidden_dim,
                             device=hidden_states.device,
                             dtype=hidden_states.dtype)

        # Process through selected experts
        unique_experts = torch.unique(top_k_indices)
        for i in unique_experts:
            expert_id = int(i)
            # token_mask (boolean tensor) = which token of the input should use this expert
            # token_mask shape: (batch_size * seq_len,)
            mask = (top_k_indices == expert_id)
            token_mask = mask.any(dim=1)
            assert token_mask.any(), f"Expecting some tokens using expert {expert_id}"

            # select tokens, apply the expert, then add to the output
            expert_input = hidden_states_reshaped[token_mask]
            expert_weight = top_k_probs[mask].unsqueeze(-1)       # shape: (N, 1)
            expert_output = self.experts[expert_id](expert_input) # shape: (N, hidden_dim)
            output[token_mask] += expert_output * expert_weight

        # Reshape back to original shape
        output = output.view(batch_size, seq_len, hidden_dim)
        return output