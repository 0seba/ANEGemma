from typing import Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from ane_models.dia.dia.dia.layers import (
    DenseGeneral,
    MlpBlock,
    RotaryEmbedding,
    Attention,
    EncoderLayer,
)
from layers.ane_ops import (
    ane_linear,
    simple_attention,
    update_kv_cache,
    ANERMSNorm,
)
from layers.ane_dense_general import ANEDenseGeneral


class ANERotaryEmbedding(nn.Module):
    """
    ANE-friendly Rotary Position Embedding implementation.

    Precomputes sin and cos values for all positions in fp32 and stores them in fp16
    for ANE compatibility. During inference, it simply indexes the precomputed values.

    This follows ANE guideline #4: precalculating frequencies for every position
    instead of computing them on the fly which would require fp32 precision.
    """

    def __init__(
        self,
        rotary_embedding: RotaryEmbedding,
        max_seq_len: int = 4096,
    ):
        super().__init__()
        self.embedding_dims = rotary_embedding.embedding_dims

        # Precompute sin and cos values for all positions up to max_seq_len
        positions = torch.arange(max_seq_len, dtype=torch.float32).unsqueeze(-1) # (max_seq_len, 1)
        timescale = rotary_embedding.timescale # (1, 1, half_dim)

        # Calculate in fp32 for accuracy but store in fp16 for ANE compatibility (the fp16 will be done internally during coremltools conversion)
        sinusoid_inp = positions / timescale # (max_seq_len, half_dim)
        sin_values = torch.sin(sinusoid_inp) # .to(torch.float16)
        cos_values = torch.cos(sinusoid_inp) # .to(torch.float16)

        self.register_buffer("sin_values", sin_values, persistent=True)
        self.register_buffer("cos_values", cos_values, persistent=True)
        self.max_seq_len = max_seq_len

    def forward(self, position: torch.Tensor, permute_for_ane: bool = False):
        """
        Returns precomputed sin and cos values for the given positions.

        Args:
            position: Tensor of positions [batch_size, seq_len] or [seq_len]
            permute_for_ane: If True, permute output to [batch, 1, half_dim, seq_len] format
                             for direct use with apply_rotary_embedding

        Returns:
            Tuple of (sin, cos) values for the requested positions
                If permute_for_ane=False: [batch_size, seq_len, half_embedding_dim]
                If permute_for_ane=True:  [batch_size, half_embedding_dim, seq_len]
        """
        if position.max() >= self.max_seq_len:
            raise ValueError(f"Position index {position.max()} exceeds maximum precomputed length ({self.max_seq_len})")

        # Lookup precomputed values and reshape to expected output format
        sin = self.sin_values[position] # (B, L, D/2)
        cos = self.cos_values[position]

        # Optionally permute for ANE format for direct use with apply_rotary_embedding
        if permute_for_ane:
            sin = sin.transpose(-1, -2)  # [batch, 1, D/2, seq_len]
            cos = cos.transpose(-1, -2)  # [batch, 1, D/2, seq_len]

        return sin, cos


def apply_rotary_embedding(inputs: torch.Tensor, sin: torch.Tensor, cos: torch.Tensor) -> torch.Tensor:
    """
    Apply rotary position embeddings to input tensors using precomputed values.

    Args:
        inputs: Tensor in NCHW format [*batch_dims, hidden_dim, sequence_length]
        sin: Precomputed sin values in format [*batch_dims, half_dim, seq_len]
        cos: Precomputed cos values in format [*batch_dims, half_dim, seq_len]

    Returns:
        Tensor with rotary embeddings
    """
    input_dtype = inputs.dtype

    first_half, second_half = inputs.chunk(chunks=2, dim=-2)
    first_part = first_half * cos - second_half * sin
    second_part = second_half * cos + first_half * sin

    return torch.cat((first_part, second_part), dim=-2).to(input_dtype)

class ANEAttention(nn.Module):
    """
    ANE-friendly implementation of the Attention module using 2D convolution operations.

    This implementation minimizes reshapes and transposes for ANE compatibility by:
    1. Using ANEDenseGeneral for projections
    2. Using precomputed rotary embeddings
    3. Using simple_attention for efficient attention computation
    4. Maintaining NCHW format throughout the forward pass
    """

    def __init__(
        self,
        attn: Attention,
        # layer_idx: int = 0,
    ):
        super().__init__()
        # self.layer_idx = layer_idx
        self.num_query_heads = attn.num_query_heads
        self.num_kv_heads = attn.num_kv_heads
        self.head_dim = attn.head_dim
        self.is_cross_attn = attn.is_cross_attn

        # Convert projections to ANE-friendly versions
        self.q_proj = ANEDenseGeneral(attn.q_proj)
        self.k_proj = ANEDenseGeneral(attn.k_proj)
        self.v_proj = ANEDenseGeneral(attn.v_proj)
        self.o_proj = ANEDenseGeneral(attn.o_proj)

    def forward(
        self,
        Xq: Tensor,  # (B, D, 1, T) NCHW format for ANE
        Xkv: Tensor | None,  # (B, E, 1, S) NCHW format for ANE
        sin_q: Tensor,  # (B, 1, H/2, S) precomputed sin for query in ANE format
        cos_q: Tensor,  # (B, 1,  H/2, S) precomputed cos for query in ANE format
        sin_k: Optional[Tensor] = None,  # (B, 1, H/2, T) precomputed sin for key in ANE format
        cos_k: Optional[Tensor] = None,  # (B, 1, H/2, T) precomputed cos for key in ANE format
        attn_mask: Optional[Tensor] = None,  # (B, 1, T, S) or None
        cache: Optional[Tuple[Tensor, Tensor]] = None,  # KV Cache for ANE format
        kv_write_idx: Optional[Tensor] = None,
        kv_layer_write_idx: Optional[int] = None,
    ) -> Tensor:
        """
        Performs attention calculation with minimal reshaping operations.

        Args:
            Xq: Query input tensor in NCHW format
            Xkv: Key/value input tensor in NCHW format
            sin_q, cos_q: Precomputed sin/cos values for query positions in ANE format
            sin_k, cos_k: Precomputed sin/cos values for key positions in ANE format
            attn_mask: Attention mask tensor
            cache: KV cache tuple
            prefill: Whether this is a prefill operation
            kv_write_idx: Index to write to in KV cache
            kv_layer_write_idx: Layer index in KV cache

        Returns:
            Tuple of (output tensor, updated KV cache)
        """
        # if kv_layer_write_idx is None:
        #     kv_layer_write_idx = self.layer_idx

        original_dtype = Xq.dtype
        batch_size = Xq.shape[0]
        seqlen_q = Xq.shape[-1]

        # (B, D, 1, S) -> (B, N*H, 1, S) for query
        q_proj = self.q_proj(Xq)
        q_proj = q_proj.view(batch_size, self.num_query_heads, self.head_dim, seqlen_q) # (B, N, H, S)
        q_rotated = apply_rotary_embedding(q_proj, sin_q, cos_q)
        # q_proj = q_proj.permute(0, 2, 1, 3)

        if self.is_cross_attn and cache is not None:
            k_cache, v_cache = cache
            attention_output = simple_attention(
                q_rotated,
                k_cache[kv_layer_write_idx : kv_layer_write_idx + batch_size],
                v_cache[kv_layer_write_idx : kv_layer_write_idx + batch_size],
                attention_mask=attn_mask,
            )
        else:
            seqlen_k = Xkv.shape[-1]
            k_proj = self.k_proj(Xkv)  # (B, H*D, 1, S)
            k_proj = k_proj.view(batch_size, self.num_kv_heads, self.head_dim, seqlen_k) # (B, H, D, S)
            k_rotated = apply_rotary_embedding(k_proj, sin_k, cos_k)
            k_rotated = k_rotated.permute(0, 1, 3, 2) # (B, H, S, D)
            v_proj = self.v_proj(Xkv).view(batch_size, self.num_kv_heads, self.head_dim, seqlen_k) # (B, H, D, S)
            # v_proj = v_proj.permute(0, 1, 3, 2) # (B, H, S, D)
            # print("v_proj permute size:", v_proj.size())
            if cache is not None:
                update_kv_cache(k_rotated, v_proj, cache, kv_write_idx, kv_layer_write_idx)
                k_cache, v_cache = cache
                key = k_cache[kv_layer_write_idx : kv_layer_write_idx + batch_size]
                value = v_cache[kv_layer_write_idx : kv_layer_write_idx + batch_size]
            else:
                key = k_rotated
                value = v_proj

            attention_output = simple_attention(
                q_rotated, key, value,
                attention_mask=attn_mask,
                attn_logit_softcapping=None,
            )

        if isinstance(attention_output, list):
            # Concatenate per-head outputs if returned as list
            output = torch.cat(attention_output, dim=1)
        else:
            # Otherwise, prepare for output projection
            # output = attention_output.permute(0, 3, 1, 2)  # (B, H, N, T)
            output = attention_output

        output = self.o_proj(output)
        return output.to(original_dtype)


class ANEMlpBlock(nn.Module):
    """
    ANE-friendly implementation of the MLP block using 2D convolution operations.

    This wrapper transforms the MlpBlock to be compatible with Apple Neural Engine by:
    1. Extracting the gate and up weights at inference time from the fused weight matrix
    2. Using point-wise convolutions via ane_linear instead of matrix multiplication
    3. Using ANEDenseGeneral for the output projection

    Attributes:
        mlp_block (MlpBlock): The original MlpBlock being wrapped
        wo (ANEDenseGeneral): ANE-friendly wrapper for the output projection
        dtype (torch.dtype): The compute dtype from the original MlpBlock
    """
    def __init__(
        self,
        mlp_block: MlpBlock,
    ):
        super().__init__()
        self.mlp_block = mlp_block
        self.wo = ANEDenseGeneral(mlp_block.wo)
        self.dtype = mlp_block.dtype

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass implementation for ANE-friendly MLP block.

        Args:
            x: Input tensor in NCHW format [batch_size, embed_dim, 1, seq_len]

        Returns:
            Tensor: Output tensor in NCHW format [batch_size, embed_dim, 1, seq_len]
        """
        # Extract gate and up weights at inference time
        # Original weight shape: [embed_dim, 2, intermediate_dim]
        wi_weight = self.mlp_block.wi_fused.weight
        gate_weight = wi_weight[:, 0, :]  # First component (dim=1, idx=0)
        up_weight = wi_weight[:, 1, :]    # Second component (dim=1, idx=1)
        gate = ane_linear(x, gate_weight.T, w_num_unsqueezes=2)
        up = ane_linear(x, up_weight.T, w_num_unsqueezes=2)
        gate_activated = F.silu(gate)
        hidden = torch.mul(gate_activated, up).to(self.dtype)
        return self.wo(hidden)

class ANEEncoderLayer(nn.Module):
    def __init__(self, layer: EncoderLayer):
        super().__init__()
        self.layer = layer
        self.pre_sa_norm = ANERMSNorm(
            self.layer.pre_sa_norm,
            dim=-1,
            w_num_unsqueezes=2,
        )
        self.self_attention = ANEAttention(
            self.layer.self_attention,
        )
        self.post_sa_norm = ANERMSNorm(
            self.layer.post_sa_norm,
            dim=-1,
            w_num_unsqueezes=2,
        )
        self.mlp = ANEMlpBlock(
            self.layer.mlp,
        )

    def forward(self,
        x: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
        sin_q: Optional[torch.Tensor] = None,
        cos_q: Optional[torch.Tensor] = None,
    ):
        residual = x
        x = self.pre_sa_norm(x)
        x = self.self_attention(x, x, attn_mask=attn_mask, sin_q=sin_q, cos_q=cos_q, sin_k=sin_q, cos_k=cos_q)
        x = residual = x + residual
        x = self.post_sa_norm(x)
        x = self.mlp(x)
        return residual + x
