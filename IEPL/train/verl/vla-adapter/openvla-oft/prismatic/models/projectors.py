"""Implementation of additional projectors for additional inputs to the VLA models."""
import torch
import torch.nn as nn


class ProprioProjector(nn.Module):
    """
    Projects proprio state inputs into the LLM's embedding space.
    """
    def __init__(self, llm_dim: int, proprio_dim: int) -> None:
        super().__init__()
        self.llm_dim = llm_dim
        self.proprio_dim = proprio_dim

        self.fc1 = nn.Linear(self.proprio_dim, self.llm_dim, bias=True)
        self.fc2 = nn.Linear(self.llm_dim, self.llm_dim, bias=True)
        self.act_fn1 = nn.GELU()

    def forward(self, proprio: torch.Tensor = None) -> torch.Tensor:
        # proprio: (bsz, proprio_dim)
        # breakpoint()
        projected_features = self.fc1(proprio)
        projected_features = self.act_fn1(projected_features)
        projected_features = self.fc2(projected_features)
        return projected_features


class NoisyActionProjector(nn.Module):
    """
    [Diffusion] Projects noisy action inputs into the LLM's embedding space.

    Note that since each action is tokenized into 7 tokens in OpenVLA (rather
    than having 1 token per action), each noisy action token will have dimension 1
    instead of 7.
    """
    def __init__(self, llm_dim: int) -> None:
        super().__init__()
        self.llm_dim = llm_dim
        self.action_token_dim = 1

        self.fc1 = nn.Linear(self.action_token_dim, self.llm_dim, bias=True)
        self.fc2 = nn.Linear(self.llm_dim, self.llm_dim, bias=True)
        self.act_fn1 = nn.GELU()

    def forward(self, noisy_actions: torch.Tensor = None) -> torch.Tensor:
        # noisy_actions: (bsz, num_action_tokens=chunk_len*action_dim, 1)
        projected_features = self.fc1(noisy_actions)
        projected_features = self.act_fn1(projected_features)
        projected_features = self.fc2(projected_features)
        return projected_features


class ContextProjector(nn.Module):
    """
    Projects LLM hidden states (e.g. 4096-dim) to DiT's expected context dimension (896-dim).
    This is required when using LLaMA2-7B (hidden_dim=4096) with DiT models that expect 896-dim context.
    """
    def __init__(self, llm_dim: int, dit_context_dim: int = 896) -> None:
        super().__init__()
        self.llm_dim = llm_dim
        self.dit_context_dim = dit_context_dim

        self.fc1 = nn.Linear(self.llm_dim, self.dit_context_dim, bias=True)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # hidden_states: (B, L, T_ctx, llm_dim) or (B, T_ctx, llm_dim)
        # output: (B, L, T_ctx, dit_context_dim) or (B, T_ctx, dit_context_dim)
        return self.fc1(hidden_states)
