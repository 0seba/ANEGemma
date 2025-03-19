# https://github.com/google/gemma_pytorch/blob/main/gemma/model.py

# Copyright 2024 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Inference-only Gemma model implementation."""

import torch
from torch import nn
import torch.nn.functional as F
from typing import Any, List, Optional, Sequence, Tuple, Union, Mapping

import config as gemma_config
from original_pytorch_implementation import (
    Embedding,
    GemmaMLP,
    GemmaModel,
    GemmaForCausalLM,
    Linear,
    RMSNorm,
    GemmaAttention,
    Gemma2DecoderLayer,
)


class ANERMSNorm(RMSNorm):
    def __init__(self, pytorch_layer: RMSNorm):
        nn.Module.__init__(self)
        self.quant = False
        self.eps = pytorch_layer.eps
        self.add_unit_offset = pytorch_layer.add_unit_offset
        self.weight = pytorch_layer.weight  # .view(*pytorch_layer.weight.size(), 1, 1)

    def _norm(self, x: torch.Tensor, dim: int):
        maxval = (x.abs().max(dim=dim, keepdim=True).values / 1).clamp(
            min=2**-24
        )  # divide by factor to use more of the float16 range
        xscaled = x / maxval
        # TODO find and way to use reduce_sum_square MIL op
        sq_sum = xscaled.square().sum(dim=dim, keepdim=True)
        rsqrt = torch.rsqrt(sq_sum)
        dimroot = (x.size(dim) ** (1 / 2))
        return rsqrt * dimroot * xscaled

    def forward(self, x: torch.Tensor, num_expands=2, dim=-3) -> torch.Tensor:
        # Llama does x.to(float16) * w whilst Gemma2 is (x * w).to(float16)
        # See https://github.com/huggingface/transformers/pull/29402
        # output = self._norm(x.float(), dim=dim)
        # weight = self.weight.float()
        output = self._norm(x, dim=dim)
        weight = self.weight
        weight = weight.view(*weight.size() + (1,) * num_expands)
        if self.add_unit_offset:
            output = output * (1 + weight)
        else:
            output = output * weight
        return output  # .type_as(x)


class ANELinear(nn.Module):
    def __init__(self, pytorch_layer: Linear):
        super().__init__()
        self.weight = pytorch_layer.weight  # .view(*pytorch_layer.weight.size(), 1, 1)
        # TODO: support quantized
        # if pytorch_layer.quant:
        #     self.weight_scaler = torch.unsqueeze(pytorch_layer.weight_scaler, (-1, -2))

    def forward(self, x):
        return F.conv2d(x, self.weight.unsqueeze(-1).unsqueeze(-1))


class ANEGemmaMLP(GemmaMLP):
    def __init__(self, pytorch_layer: GemmaMLP):
        nn.Module.__init__(self)
        self.gate_proj = ANELinear(pytorch_layer.gate_proj)
        self.up_proj = ANELinear(pytorch_layer.up_proj)
        self.down_proj = ANELinear(pytorch_layer.down_proj)


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1, x2 = x.chunk(2, dim=-1)
    neg_x2 = -x2
    return torch.concat((neg_x2, x1), dim=-1)


def apply_rotary_pos_emb(
    x: torch.Tensor, cos_emb: torch.Tensor, sin_emb: torch.Tensor
) -> torch.Tensor:
    lhs = x * cos_emb
    xrot = rotate_half(x)
    rhs = xrot * sin_emb
    return lhs + rhs


class ANEGemmaAttention(nn.Module):
    def __init__(self, pytorch_layer: GemmaAttention, layer_index: int):
        super().__init__()
        self.layer_index = layer_index
        self.num_heads = pytorch_layer.num_heads
        self.num_kv_heads = pytorch_layer.num_kv_heads
        self.num_queries_per_kv = pytorch_layer.num_queries_per_kv
        self.hidden_size = pytorch_layer.hidden_size
        self.head_dim = pytorch_layer.head_dim
        self.q_size = pytorch_layer.q_size
        self.kv_size = pytorch_layer.kv_size
        self.scaling = pytorch_layer.scaling
        self.attn_logit_softcapping = pytorch_layer.attn_logit_softcapping
        self.attn_type = pytorch_layer.attn_type
        self.sliding_window_size = pytorch_layer.sliding_window_size

        self.query_norm = (
            ANERMSNorm(pytorch_layer.query_norm)
            if pytorch_layer.query_norm is not None
            else None
        )
        self.key_norm = (
            ANERMSNorm(pytorch_layer.key_norm)
            if pytorch_layer.key_norm is not None
            else None
        )

        self.qkv_proj = ANELinear(pytorch_layer.qkv_proj)
        self.o_proj = ANELinear(pytorch_layer.o_proj)

    def forward(
        self,
        hidden_states: torch.Tensor,
        freqs_cis: Tuple[torch.Tensor, torch.Tensor],
        kv_write_indices: torch.Tensor,
        kv_cache: Tuple[torch.Tensor, torch.Tensor],
        mask: torch.Tensor,
        local_mask: torch.Tensor = None,
    ) -> torch.Tensor:
        batch_size = hidden_states.size(0)
        seqlen = hidden_states.size(-1)
        qkv = self.qkv_proj(hidden_states)
        qkv = qkv.view(
            batch_size, self.num_heads + self.num_kv_heads * 2, self.head_dim, -1
        )
        qkv = qkv.permute(0, 1, 3, 2)
        xq, xk, xv = qkv.split(
            [self.num_heads, self.num_kv_heads, self.num_kv_heads],
            dim=1,
        )  # (batch_size, num_heads, seq_len, head_dim)

        if self.key_norm:
            xk = self.key_norm(xk, num_expands=0, dim=-1)
        xk = apply_rotary_pos_emb(
            xk, freqs_cis[self.attn_type][0], freqs_cis[self.attn_type][1]
        )

        if self.query_norm:
            xq = self.query_norm(xq, num_expands=0, dim=-1)
        xq = apply_rotary_pos_emb(
            xq, freqs_cis[self.attn_type][0], freqs_cis[self.attn_type][1]
        )
        xq = xq * self.scaling
        xq = xq.chunk(self.num_heads, 1)

        k_cache, v_cache = kv_cache
        if k_cache.size(0) > 1:
            k_cache[
                self.layer_index : self.layer_index + 1,
                : self.num_kv_heads,
                kv_write_indices : kv_write_indices + 1,
                : self.head_dim,
            ] = xk
            v_cache[
                self.layer_index : self.layer_index + 1,
                : self.num_kv_heads,
                kv_write_indices : kv_write_indices + 1,
                : self.head_dim,
            ] = xv
            xk = k_cache[self.layer_index : self.layer_index + 1]
            xv = v_cache[self.layer_index : self.layer_index + 1]
        else:
            k_cache[
                :1,
                : self.num_kv_heads,
                kv_write_indices : kv_write_indices + 1,
                : self.head_dim,
            ] = xk
            v_cache[
                :1,
                : self.num_kv_heads,
                kv_write_indices : kv_write_indices + 1,
                : self.head_dim,
            ] = xv
            xk = k_cache
            xv = v_cache

        xk = xk.split(self.num_kv_heads, 1)
        xv = xv.split(self.num_kv_heads, 1)

        # we are performing our computations transposed
        # coreml graph pass should convert this mask tranpose in every attention to a single one
        # and reuse
        if (
            self.attn_type == gemma_config.AttentionType.LOCAL_SLIDING
            and self.sliding_window_size is not None
            and local_mask is not None
        ):
            mask = local_mask
        mask = mask.permute(0, 1, 3, 2)
        per_head_attention = []
        for i in range(self.num_heads):
            groupi = i // self.num_queries_per_kv
            xqi = xq[i]  # (1, numheads, seqlen, headdims)
            # CoreML should optimize this matmul with transpose
            # https://github.com/apple/coremltools/blob/d3f5493115c1dddcdf63f45e84ca9375a8c3e639/coremltools/converters/mil/mil/passes/defs/optimize_linear.py#L308
            scores = torch.matmul(
                xk[groupi], xqi.permute(0, 1, 3, 2)
            )  # (1, 1, CACHE_LEN, SEQLEN)
            if self.attn_logit_softcapping is not None:
                scores = scores / self.attn_logit_softcapping
                scores = torch.tanh(scores)
                scores = scores * self.attn_logit_softcapping
            scores = scores + mask
            scores = F.softmax(scores, dim=-2)
            output = torch.matmul(scores.permute(0, 1, 3, 2), xv[groupi])
            per_head_attention.append(output.permute(0, 3, 1, 2))

        output = torch.cat(per_head_attention, dim=1)
        output = self.o_proj(output)
        return output


# Name mantains pytorch Gemma structure
class ANEGemma2DecoderLayer(Gemma2DecoderLayer):
    def __init__(
        self,
        pytorch_layer: Gemma2DecoderLayer,
        config: gemma_config.GemmaConfig,
        layer_index: int,
    ):
        nn.Module.__init__(self)
        # super().__init__()
        self.config = config
        self.attn_type = pytorch_layer.attn_type
        self.self_attn = ANEGemmaAttention(pytorch_layer.self_attn, layer_index)
        self.mlp = ANEGemmaMLP(pytorch_layer.mlp)
        self.input_layernorm = ANERMSNorm(pytorch_layer.input_layernorm)
        self.post_attention_layernorm = ANERMSNorm(
            pytorch_layer.post_attention_layernorm
        )
        self.pre_feedforward_layernorm = (
            ANERMSNorm(pytorch_layer.pre_feedforward_layernorm)
            if config.use_pre_ffw_norm
            else None
        )
        self.post_feedforward_layernorm = (
            ANERMSNorm(pytorch_layer.post_feedforward_layernorm)
            if config.use_post_ffw_norm
            else None
        )


class ANEGemmaModel(nn.Module):
    def __init__(
        self,
        layers: List[Gemma2DecoderLayer],
        config: gemma_config.GemmaConfig,
        state_implementation: str,
    ):
        super().__init__()
        self.config: gemma_config.GemmaConfig = config
        self.layers = nn.ModuleList()
        for i, layer in enumerate(layers):
            self.layers.append(ANEGemma2DecoderLayer(layer, config, i))
        # self.norm = ANERMSNorm(pytorch_model.norm)
        self.state_implementation = state_implementation

    def forward(
        self,
        hidden_states,
        rope_embs,
        kv_write_indices,
        mask,
        local_mask,
        layer_from=0,
        layer_to=None,
        kv_cache=None,
    ):
        if kv_cache is not None:
            pass
        elif self.state_implementation == "single":
            kv_cache = (self.k_cache, self.v_cache)
        elif self.state_implementation == "per_layer":
            kv_cache = [
                (getattr(self, f"k_cache_{i}"), getattr(self, f"v_cache_{i}"))
                for i in range(len(self.layers))
            ]
        else:
            raise ValueError(
                f"Unknown state implementation: {self.state_implementation}"
            )
        if layer_to is None:
            layer_to = len(self.layers)
        for i in range(layer_from, layer_to):
            layer: ANEGemma2DecoderLayer = self.layers[i]
            kv_cache_i = (
                kv_cache[i] if self.state_implementation == "per_layer" else kv_cache
            )
            hidden_states = layer(
                hidden_states,
                rope_embs,
                kv_write_indices,
                kv_cache_i,
                mask,
                local_mask,
            )
        return hidden_states


class ANEGemmaForCausalLM(nn.Module):
    def __init__(self, pytorch_model: GemmaForCausalLM, state_implementation: str):
        super().__init__()
        self.config: gemma_config.GemmaConfig = pytorch_model.config
        self.tokenizer = pytorch_model.tokenizer
        self.model = ANEGemmaModel(
            pytorch_model.model.layers,
            config=self.config,
            state_implementation=state_implementation,
        )
        self.embedder: Embedding = pytorch_model.embedder
        self.norm = ANERMSNorm(pytorch_model.model.norm)

    def forward(
        self,
        hidden_states,
        kv_write_indices,
        rope_embs,
        mask=None,
        local_mask=None,
        layer_from=0,
        layer_to=None,
        apply_final_norm=True,
        kv_cache=None,
    ):
        normalizer = torch.tensor(
            self.config.hidden_size**0.5, dtype=hidden_states.dtype
        )
        hidden_states = hidden_states * normalizer
        hidden_states = self.model(
            hidden_states,
            rope_embs,
            kv_write_indices,
            mask,
            local_mask,
            layer_from,
            layer_to,
            kv_cache,
        )
        if apply_final_norm:
            hidden_states = self.norm(hidden_states)
        return hidden_states


def precompute_freqs_cis(
    dim: int,
    end: int,
    theta: float = 10000.0,
    rope_scaling_factor: int = 1,
) -> torch.Tensor:
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    freqs = freqs / rope_scaling_factor
    t = torch.arange(end, device=freqs.device, dtype=torch.float32)
    freqs = torch.outer(t, freqs).float()
    return freqs


def compute_rope_embedding(freqs):
    emb = torch.cat((freqs, freqs), dim=-1)
    cos_emb = torch.cos(emb)
    sin_emb = torch.sin(emb)
    return cos_emb.half(), sin_emb.half()


class Wrapper(torch.nn.Module):
    def __init__(
        self,
        model: ANEGemmaForCausalLM,
        layer_from=0,
        layer_to=None,
        state_implementation="single",
        apply_final_norm=True,
        predict=True,
        prediction_head_chunk_size=16_384,
    ):
        super().__init__()
        self.model = model
        self.config = self.model.config
        self.layer_from = layer_from
        self.layer_to = layer_to
        self.state_implementation = state_implementation
        self.apply_final_norm = apply_final_norm
        self.predict = predict
        self.prediction_head_chunk_size = prediction_head_chunk_size
        self.num_layers = (layer_to or model.config.num_hidden_layers) - layer_from
        self.local_cos_emb, self.local_sin_emb = compute_rope_embedding(
            precompute_freqs_cis(
                self.config.head_dim,
                self.config.max_position_embeddings,
                self.config.rope_wave_length[gemma_config.AttentionType.LOCAL_SLIDING],
            )
        )
        self.global_cos_emb, self.global_sin_emb = compute_rope_embedding(
            precompute_freqs_cis(
                self.config.head_dim,
                self.config.max_position_embeddings,
                self.config.rope_wave_length[gemma_config.AttentionType.GLOBAL],
            )
        )
        self.init_cache(state_implementation)

    def forward(self, hidden_states, kv_write_indices, position, *args):
        if self.state_implementation == "single":
            kv_cache = (self.k_cache, self.v_cache)
        else:
            kv_cache = []
            for i in range(self.num_layers):
                kv_cache.append(
                    (getattr(self, f"k_cache_{i}"), getattr(self, f"v_cache_{i}"))
                )
        if not torch.is_tensor(position):
            position = torch.asarray(position, dtype=torch.int32).view(1, 1)

        rope_embs = {}
        rope_embs[gemma_config.AttentionType.LOCAL_SLIDING] = (
            self.local_cos_emb[position].unsqueeze(1),
            self.local_sin_emb[position].unsqueeze(1),
        )
        rope_embs[gemma_config.AttentionType.GLOBAL] = (
            self.global_cos_emb[position].unsqueeze(1),
            self.global_sin_emb[position].unsqueeze(1),
        )
        hidden_states = self.model(
            hidden_states,
            kv_write_indices,
            rope_embs,
            *args,
            layer_from=self.layer_from,
            layer_to=self.layer_to,
            apply_final_norm=self.apply_final_norm,
            kv_cache=kv_cache,
        )
        if self.predict:
            head_chunks = self.model.embedder.weight.split(
                self.prediction_head_chunk_size, dim=0
            )
            logits, topks, values, lses = [], [], [], []
            for head_chunk in head_chunks:
                _logits = F.conv2d(
                    hidden_states, head_chunk.unsqueeze(-1).unsqueeze(-1)
                )
                if self.model.config.final_logit_softcapping is not None:
                    _logits = _logits / self.model.config.final_logit_softcapping
                    _logits = torch.tanh(_logits)
                    _logits = _logits * self.model.config.final_logit_softcapping
                logits.append(_logits)
                lses.append(torch.logsumexp(_logits, dim=1, keepdim=True))

                # Would be better to do this here? Less data movement
                # But topk is not supported on M1
                # topk = torch.topk(_logits, k=1, dim=1, sorted=True)
                # topks.append(topk.indices)
                # values.append(topk.values)

            lse = torch.logsumexp(torch.cat(lses, dim=1), dim=1, keepdim=True)
            for i in range(len(logits)):
                logits[i] = torch.exp(logits[i] - lse)

            # for _logits in logits:
            #     # topk = torch.topk(_logits, k=1, dim=1, sorted=True)
            #     # topks.append(topk.indices)
            #     # values.append(topk.values)
            #     topks.append(torch.argmax(_logits, dim=1, keepdim=True))
            #     values.append(torch.max(_logits, dim=1, keepdim=True))
            # values = torch.cat(values, dim=1)
            # topks = [torch.asarray(tp, dtype=torch.int32) for tp in topks]
            # topks = torch.cat(topks, dim=1) # Cannot concat uint16 on ANE (tested on M1)

            # topk = torch.topk(values, k=1, dim=1, sorted=True)
            # topk, topkvalue = topk.indices, topk.values
            # topkvalue -= lse
            # topk = topk * self.prediction_head_chunk_size + torch.gather(
            #     topks, dim=1, index=topk
            # )  # this last gather is gonna get pushed to cpu

            logits = torch.cat(logits, dim=1)
            # return (
            #     topks,
            #     values,
            #     lse,
            #     # topks,
            #     # values,
            #     logits,
            # )
            return logits.argmax(1), logits, lse

        return hidden_states

    def init_cache(self, state_implementation):
        if state_implementation == "single":
            self.register_buffer(
                "k_cache",
                torch.zeros(
                    (
                        self.num_layers,
                        self.config.num_key_value_heads,
                        self.config.sliding_window_size,
                        self.config.head_dim,
                    ),
                    dtype=torch.float16,
                ),
            )
            self.register_buffer(
                "v_cache",
                torch.zeros(
                    (
                        self.num_layers,
                        self.config.num_key_value_heads,
                        self.config.sliding_window_size,
                        self.config.head_dim,
                    ),
                    dtype=torch.float16,
                ),
            )
        elif state_implementation == "per_layer":
            for i in range(self.num_layers):
                self.register_buffer(
                    f"k_cache_{i}",
                    torch.zeros(
                        (
                            1,
                            self.config.num_key_value_heads,
                            self.config.sliding_window_size,
                            self.config.head_dim,
                        ),
                        dtype=torch.float16,
                    ),
                )
                self.register_buffer(
                    f"v_cache_{i}",
                    torch.zeros(
                        (
                            1,
                            self.config.num_key_value_heads,
                            self.config.sliding_window_size,
                            self.config.head_dim,
                        ),
                        dtype=torch.float16,
                    ),
                )
