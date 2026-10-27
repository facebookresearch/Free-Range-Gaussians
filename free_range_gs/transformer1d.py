import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Any
from dataclasses import dataclass

# The timestep helpers below are a minimal compatibility implementation of the
# Apache-2.0-licensed Hugging Face Diffusers API used by the training snapshot.


def get_timestep_embedding(
    timesteps: torch.Tensor,
    embedding_dim: int,
    flip_sin_to_cos: bool = False,
    downscale_freq_shift: float = 1,
    scale: float = 1,
    max_period: int = 10000,
) -> torch.Tensor:
    """Diffusers-compatible sinusoidal timestep embedding."""
    if timesteps.ndim != 1:
        raise ValueError("timesteps must be one-dimensional")
    half_dim = embedding_dim // 2
    exponent = -math.log(max_period) * torch.arange(
        0, half_dim, dtype=torch.float32, device=timesteps.device
    )
    exponent = exponent / (half_dim - downscale_freq_shift)
    embedding = scale * timesteps[:, None].float() * torch.exp(exponent)[None, :]
    embedding = torch.cat([torch.sin(embedding), torch.cos(embedding)], dim=-1)
    if flip_sin_to_cos:
        embedding = torch.cat([embedding[:, half_dim:], embedding[:, :half_dim]], dim=-1)
    if embedding_dim % 2:
        embedding = F.pad(embedding, (0, 1, 0, 0))
    return embedding


class TimestepEmbedding(nn.Module):
    """Parameter-compatible subset of diffusers' TimestepEmbedding."""

    def __init__(self, in_channels: int, time_embed_dim: int):
        super().__init__()
        self.linear_1 = nn.Linear(in_channels, time_embed_dim, True)
        self.cond_proj = None
        self.act = nn.SiLU()
        self.linear_2 = nn.Linear(time_embed_dim, time_embed_dim, True)
        self.post_act = None

    def forward(self, sample, condition=None):
        if condition is not None:
            raise ValueError("conditional timestep projection is unsupported")
        return self.linear_2(self.act(self.linear_1(sample)))


class Timesteps(nn.Module):
    """Parameter-free subset of diffusers' Timesteps."""

    def __init__(self, num_channels, flip_sin_to_cos, downscale_freq_shift, scale=1):
        super().__init__()
        self.num_channels = num_channels
        self.flip_sin_to_cos = flip_sin_to_cos
        self.downscale_freq_shift = downscale_freq_shift
        self.scale = scale

    def forward(self, timesteps):
        return get_timestep_embedding(
            timesteps,
            self.num_channels,
            self.flip_sin_to_cos,
            self.downscale_freq_shift,
            self.scale,
        )


class RMSNorm(nn.Module):
    """
    Root Mean Square Layer Normalization.
    Used for QK normalization in attention for training stability.
    """
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype
        x = x.float()
        rms = torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + self.eps)
        x = x / rms * self.weight.float()
        return x.to(dtype)


class Attention(nn.Module):
    """
    Multi-head attention using torch.nn.functional.scaled_dot_product_attention.
    Supports both self-attention and cross-attention.
    Uses QK-Norm for training stability.
    """

    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        head_dim: int = 64,
        dropout: float = 0.0,
        bias: bool = False,
        cross_attention_dim: Optional[int] = None,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.inner_dim = num_heads * head_dim
        self.dropout = dropout

        # Query projection
        self.to_q = nn.Linear(dim, self.inner_dim, bias=bias)

        # Key/Value projections (from cross_attention_dim if provided)
        kv_dim = cross_attention_dim if cross_attention_dim is not None else dim
        self.to_k = nn.Linear(kv_dim, self.inner_dim, bias=bias)
        self.to_v = nn.Linear(kv_dim, self.inner_dim, bias=bias)

        # QK normalization for training stability
        self.q_norm = RMSNorm(head_dim)
        self.k_norm = RMSNorm(head_dim)

        # Output projection
        self.to_out = nn.Sequential(
            nn.Linear(self.inner_dim, dim, bias=True),
            nn.Dropout(dropout),
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        batch_size, seq_len, _ = hidden_states.shape

        # Query from hidden_states
        q = self.to_q(hidden_states)

        # Key/Value from encoder_hidden_states (cross-attention) or hidden_states (self-attention)
        kv_states = encoder_hidden_states if encoder_hidden_states is not None else hidden_states
        k = self.to_k(kv_states)
        v = self.to_v(kv_states)

        # Reshape for multi-head attention: [B, L, H*D] -> [B, H, L, D]
        q = q.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)

        # Apply QK normalization for training stability
        q = self.q_norm(q)
        k = self.k_norm(k)

        # Scaled dot-product attention (uses Flash Attention when available)
        out = F.scaled_dot_product_attention(
            q, k, v,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=False,
        )

        # Reshape back: [B, H, L, D] -> [B, L, H*D]
        out = out.transpose(1, 2).reshape(batch_size, seq_len, self.inner_dim)
        out = self.to_out(out)

        return out


class AdaLayerNorm(nn.Module):
    """
    Adaptive Layer Normalization with timestep conditioning.
    Produces scale, shift, and gate parameters from timestep embedding.
    """

    def __init__(self, embedding_dim: int, num_modulation_params: int = 6):
        super().__init__()
        self.silu = nn.SiLU()
        self.linear = nn.Linear(embedding_dim, num_modulation_params * embedding_dim, bias=True)
        self.num_modulation_params = num_modulation_params

        # Initialize to identity transform
        nn.init.zeros_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)

    def forward(self, timestep_emb: torch.Tensor) -> torch.Tensor:
        """
        Args:
            timestep_emb: [B, D] timestep embedding
        Returns:
            [B, num_params, D] modulation parameters
        """
        emb = self.linear(self.silu(timestep_emb))
        return emb.view(timestep_emb.shape[0], self.num_modulation_params, -1)


class DiTTransformerBlock(nn.Module):
    """
    DiT-style Transformer block with:
    - Timestep-conditioned AdaLN modulation (scale, shift, gate)
    - Self-attention using F.scaled_dot_product_attention
    - Optional cross-attention
    - Feed-forward with gated output
    """

    def __init__(
        self,
        dim: int,
        num_attention_heads: int,
        attention_head_dim: int,
        dropout: float = 0.0,
        cross_attention_dim: Optional[int] = None,
        activation_fn: str = "geglu",
        attention_bias: bool = False,
    ):
        super().__init__()
        self.has_cross_attention = cross_attention_dim is not None

        # Determine number of modulation params: 6 for self-attn + mlp, +3 if cross-attn
        num_modulation_params = 9 if self.has_cross_attention else 6

        # AdaLN: projects timestep embedding to modulation parameters
        self.adaLN = AdaLayerNorm(dim, num_modulation_params=num_modulation_params)

        # 1. Self-attention
        self.norm1 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.attn1 = Attention(
            dim=dim,
            num_heads=num_attention_heads,
            head_dim=attention_head_dim,
            dropout=dropout,
            bias=attention_bias,
            cross_attention_dim=None,  # Self-attention
        )

        # 2. Cross-attention (optional)
        if self.has_cross_attention:
            self.norm2 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
            self.attn2 = Attention(
                dim=dim,
                num_heads=num_attention_heads,
                head_dim=attention_head_dim,
                dropout=dropout,
                bias=attention_bias,
                cross_attention_dim=cross_attention_dim,
            )
        else:
            self.norm2 = None
            self.attn2 = None

        # 3. Feed-forward
        self.norm3 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        if activation_fn == "geglu":
            self.ff = GEGLU(dim, dim * 4, dim)
        else:
            self.ff = nn.Sequential(
                nn.Linear(dim, dim * 4),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(dim * 4, dim),
                nn.Dropout(dropout)
            )

    def forward(
        self,
        hidden_states: torch.Tensor,
        timestep_emb: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # Get modulation parameters from AdaLN: [B, num_params, D]
        modulation = self.adaLN(timestep_emb)

        if self.has_cross_attention:
            (shift_msa, scale_msa, gate_msa,
             shift_ca, scale_ca, gate_ca,
             shift_mlp, scale_mlp, gate_mlp) = modulation.chunk(9, dim=1)
        else:
            shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = modulation.chunk(6, dim=1)

        # 1. Self-attention with AdaLN modulation
        norm_hidden_states = self.norm1(hidden_states)
        norm_hidden_states = norm_hidden_states * (1 + scale_msa) + shift_msa
        attn_output = self.attn1(norm_hidden_states)
        hidden_states = hidden_states + gate_msa * attn_output

        # 2. Cross-attention with AdaLN modulation (if applicable)
        if self.has_cross_attention and encoder_hidden_states is not None:
            norm_hidden_states = self.norm2(hidden_states)
            norm_hidden_states = norm_hidden_states * (1 + scale_ca) + shift_ca
            attn_output = self.attn2(norm_hidden_states, encoder_hidden_states)
            hidden_states = hidden_states + gate_ca * attn_output

        # 3. Feed-forward with AdaLN modulation
        norm_hidden_states = self.norm3(hidden_states)
        norm_hidden_states = norm_hidden_states * (1 + scale_mlp) + shift_mlp
        ff_output = self.ff(norm_hidden_states)
        hidden_states = hidden_states + gate_mlp * ff_output

        return hidden_states


class GEGLU(nn.Module):
    """
    GEGLU activation function - Gated Linear Unit with GELU activation.
    From: https://arxiv.org/abs/2002.05202
    """

    def __init__(self, dim_in: int, dim_hidden: int, dim_out: int):
        super().__init__()
        self.proj = nn.Linear(dim_in, dim_hidden * 2)
        self.out_proj = nn.Linear(dim_hidden, dim_out)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x, gate = self.proj(x).chunk(2, dim=-1)
        return self.out_proj(x * F.gelu(gate))


class Transformer1D(nn.Module):
    """
    1D Transformer for point generation, aligned with DiT architecture.

    This transformer processes sequences of tokens with both self-attention and cross-attention.
    Each layer contains:
    1. Self-attention with AdaLN modulation (scale, shift, gate)
    2. Cross-attention with AdaLN modulation - if cross_attention_dim is provided
    3. Feed-forward network with AdaLN modulation

    Key DiT-style features:
    - Timestep conditioning via AdaLN (adaptive layer normalization)
    - Gated residual connections
    - Uses F.scaled_dot_product_attention for efficiency
    """

    def __init__(self,
        num_attention_heads: int = 8,
        attention_head_dim: int = 32,
        in_channels: int = 512,
        num_layers: int = 10,
        dropout: float = 0.0,
        cross_attention_dim: Optional[int] = None,
        activation_fn: str = "geglu",
        attention_bias: bool = False,
        use_mixed_precision: bool = False,
        **kwargs):

        super().__init__()

        self.num_attention_heads = num_attention_heads
        self.num_layers = num_layers
        self.attention_head_dim = attention_head_dim
        self.cross_attention_dim = cross_attention_dim
        self.use_mixed_precision = use_mixed_precision
        inner_dim = self.num_attention_heads * self.attention_head_dim
        self.inner_dim = inner_dim

        # Input normalization (LayerNorm like DiT - works directly on [B, L, C])
        self.norm_in = nn.LayerNorm(in_channels, eps=1e-6)

        # Input projection
        self.proj_in = nn.Linear(in_channels, inner_dim)

        # Timestep embedding (DiT-style)
        self.time_proj = Timesteps(num_channels=256, flip_sin_to_cos=True, downscale_freq_shift=0)
        self.time_embedding = TimestepEmbedding(in_channels=256, time_embed_dim=inner_dim)

        # Transformer blocks
        self.transformer_blocks = nn.ModuleList([
            DiTTransformerBlock(
                dim=inner_dim,
                num_attention_heads=num_attention_heads,
                attention_head_dim=attention_head_dim,
                dropout=dropout,
                cross_attention_dim=cross_attention_dim,
                activation_fn=activation_fn,
                attention_bias=attention_bias,
            )
            for _ in range(num_layers)
        ])

        # Output normalization and projection (DiT-style with final AdaLN)
        self.norm_out = nn.LayerNorm(inner_dim, elementwise_affine=False, eps=1e-6)
        self.adaLN_out = AdaLayerNorm(inner_dim, num_modulation_params=2)  # shift, scale
        self.proj_out = nn.Linear(inner_dim, in_channels)

        print(f"Transformer1D (DiT-style) initialized with {num_layers} layers")
        print(f"  - inner_dim: {inner_dim} ({num_attention_heads} heads x {attention_head_dim} head_dim)")
        if cross_attention_dim is not None:
            print(f"  - Cross-attention enabled with dim={cross_attention_dim}")
        else:
            print(f"  - Self-attention only (no cross-attention)")

    def forward(
        self,
        hidden_states: torch.Tensor,
        timestep: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        residual = hidden_states

        # 1. Input processing
        hidden_states = self.norm_in(hidden_states)
        hidden_states = self.proj_in(hidden_states)

        # 2. Compute timestep embedding
        timestep_proj = self.time_proj(timestep)
        timestep_emb = self.time_embedding(timestep_proj.to(hidden_states.dtype))  # [B, inner_dim]

        # 3. Apply transformer blocks
        for block in self.transformer_blocks:
            hidden_states = block(
                hidden_states,
                timestep_emb=timestep_emb,
                encoder_hidden_states=encoder_hidden_states,
            )

        # 4. Output processing (DiT-style final AdaLN)
        modulation = self.adaLN_out(timestep_emb)  # [B, 2, inner_dim]
        shift, scale = modulation.chunk(2, dim=1)
        hidden_states = self.norm_out(hidden_states)
        hidden_states = hidden_states * (1 + scale) + shift

        hidden_states = self.proj_out(hidden_states)

        # Add residual connection
        output = hidden_states + residual

        return output

    def get_attention_info(self) -> Dict[str, Any]:
        """Get information about the attention layers."""
        info = {
            "num_layers": self.num_layers,
            "num_attention_heads": self.num_attention_heads,
            "attention_head_dim": self.attention_head_dim,
            "total_attention_dim": self.num_attention_heads * self.attention_head_dim,
            "has_cross_attention": self.cross_attention_dim is not None,
        }

        if info["has_cross_attention"]:
            info["cross_attention_layers"] = self.num_layers
            info["cross_attention_dim"] = self.cross_attention_dim
        else:
            info["cross_attention_layers"] = 0

        return info
