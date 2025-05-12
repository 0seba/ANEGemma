from typing import Optional, Tuple
import torch
import torch.nn as nn
from torch import Tensor

def ane_rms_norm(
    x: Tensor,
    dim: int,
    w: Optional[Tensor] = None,
    w_num_unsqueezes: Optional[int] = None,
    eps: float = 1e-6,
    add_unit_offset: bool = False,
    scaling_factor: float = 1.0,
):
    maxval = (x.abs().max(dim=dim, keepdim=True).values / scaling_factor).clamp(
        min=2**-24
    )  # divide by factor to use more of the float16 range
    xscaled = x / maxval
    sq_sum = xscaled.square().sum(dim=dim, keepdim=True)
    # not using eps here, coreml rsqrt applies a 1e-12 default eps,
    # not sure how it interprets that value in fp16, because of the maxval scaling
    # this is also no exactly equivalent to the pytorch fp32 implementation
    rsqrt = torch.rsqrt(sq_sum)
    dimroot = x.size(dim) ** (1 / 2)
    x = rsqrt * dimroot * xscaled

    if w is not None:
        if add_unit_offset:
            w = w + 1
        if w_num_unsqueezes is not None:
            w = w.view(*w.size() + (1,) * w_num_unsqueezes)
        x = x * w

    return x

def ane_linear(x: Tensor, w: Tensor, bias: Optional[Tensor] = None, w_num_unsqueezes: int = 2):
    # Assume x has shape (B, C, 1, L)
    # w has shape (D, C)
    w = w.reshape(*(w.size() + (1,) * w_num_unsqueezes))
    return torch.nn.functional.conv2d(x, w, bias=bias)

class ANELinear(nn.Module):
    def __init__(self, layer: nn.Linear):
        super().__init__()
        self.layer = layer

    def forward(self, x):
        return ane_linear(x, self.layer.weight, bias=self.layer.bias)

def update_kv_cache(
    key: Tensor,
    value: Tensor,
    kv_cache: Tuple[Tensor, Tensor],
    kv_write_idx: Tensor, # current implementation supports only one contiguous write
    kv_layer_write_idx: int,

):
    # seqlen will be a time runtime constant, we have to create
    # a function for each seqlen or use EnumeratedShapes
    b, numheads, seqlen, headdim = key.size()

    k_cache, v_cache = kv_cache
    if seqlen == k_cache.size(2):
        start = 0
        end = seqlen
    else:
        start = kv_write_idx
        end = kv_write_idx + seqlen

    if k_cache.size(0) == 1:
        kv_layer_write_idx = 0

    k_cache[
        kv_layer_write_idx : kv_layer_write_idx + b,
        : numheads,
        start:end,
        : headdim,
    ] = key
    v_cache[
        kv_layer_write_idx : kv_layer_write_idx + b,
        : numheads,
        : headdim,
        start:end,
    ] = value


def simple_attention(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    attention_mask: Optional[Tensor] = None,
    attn_logit_softcapping: Optional[float] = None,
):
    """
        query: (batch, heads, head_dim, sequence_length)
        key: (batch, heads, sequence, head_dim)
        values: (batch, heads, sequence, head_dim)
        kv_cache: ((batch, heads, cache_length, head_dim), (batch, heads, head_dim, cache_length))
        kv_write_index: (1,)
        kv_layer_write_index: int
        num_q_heads: int
        num_kv_heads: int
        attention_mask: (batch, 1, cache_length, sequence_length)
    """
    num_q_heads = query.size(1)
    num_kv_heads = key.size(1)
    q = query.chunk(num_q_heads, 1)
    k = key.chunk(num_kv_heads, 1)
    v = value.chunk(num_kv_heads, 1)

    assert (num_q_heads % num_kv_heads) == 0
    num_queries_per_kv = num_q_heads // num_kv_heads
    per_head_attention = []
    for i in range(num_q_heads):
        groupi = i // num_queries_per_kv
        qi = q[i]
        # perform KQ intead of QK to read contigously from K which is bigger (?)
        scores = torch.matmul(
            k[groupi], qi
        )  # (1, 1, CACHE_LEN, SEQLEN)
        if attn_logit_softcapping is not None:
            scores = scores / attn_logit_softcapping
            scores = torch.tanh(scores)
            scores = scores * attn_logit_softcapping
        if attention_mask is not None:
            scores = scores + attention_mask
        scores = torch.nn.functional.softmax(scores.float(), dim=-2) # .to(scores.dtype)
        # output = torch.matmul(scores.permute(0, 1, 3, 2), v[groupi])
        # per_head_attention.append(output.permute(0, 3, 1, 2))
        output = torch.matmul(v[groupi].float(), scores).to(qi.dtype)
        per_head_attention.append(output.permute(0, 2, 1, 3))

    # output = torch.cat(per_head_attention, dim=1)
    # return before concat, this way we could experiment later to perform
    # split output projection and the reduce sum, maybe faster due to
    # cache (?)
    return per_head_attention
