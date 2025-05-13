import torch
import torch.nn as nn
from unittest import TestCase

from ane_models.dia.dia.dia.layers import (
    MlpBlock,
    RotaryEmbedding,
    Attention,
    EncoderLayer,
    EncoderInferenceState,
)
from ane_models.dia.dia.dia.config import DiaConfig, ModelConfig, EncoderConfig, DecoderConfig, DataConfig
from ane_models.dia.dia.dia.state import KVCache, create_attn_mask
from ane_models.dia.ane_dia import (
    ANEMlpBlock,
    ANERotaryEmbedding,
    apply_rotary_embedding,
    ANEAttention,
    update_kv_cache,
    ANEEncoderLayer,
    ANERotaryEmbedding,
)


def test_ane_mlp_block():
    """
    Test that ANEMlpBlock produces the same outputs as MlpBlock
    using standard normal distribution for weights and inputs.
    """
    # Setup small test dimensions
    batch_size, seq_len, embed_dim, intermediate_dim = 2, 4, 8, 16
    compute_dtype = torch.float32

    # Create MlpBlock with random weights
    mlp_block = MlpBlock(embed_dim=embed_dim, intermediate_dim=intermediate_dim,
                        compute_dtype=compute_dtype)

    with torch.no_grad():
        mlp_block.wi_fused.weight = nn.Parameter(
            torch.randn((embed_dim, 2, intermediate_dim), dtype=compute_dtype))
        mlp_block.wo.weight = nn.Parameter(
            torch.randn((intermediate_dim, embed_dim), dtype=compute_dtype))

    # Create ANEMlpBlock and test data
    ane_mlp_block = ANEMlpBlock(mlp_block)
    x_std = torch.randn((batch_size, seq_len, embed_dim), dtype=compute_dtype)
    x_ane = x_std.transpose(1, 2).unsqueeze(2)

    # Run forward passes
    y_std = mlp_block(x_std)
    y_ane = ane_mlp_block(x_ane).squeeze(2).transpose(1, 2)

    # Verify outputs match
    test_case = TestCase()
    if torch.isnan(y_std).any() or torch.isnan(y_ane).any():
        raise ValueError("Test failed: NaN values detected in outputs")

    test_case.assertTrue(
        torch.allclose(y_std, y_ane, rtol=1e-5, atol=1e-5),
        "ANEMlpBlock output doesn't match MlpBlock output"
    )

    print("ANEMlpBlock test passed!")


def test_ane_rotary_embedding():
    """
    Test that ANERotaryEmbedding produces the same outputs as RotaryEmbedding
    when applied to the same inputs, using the ANE-friendly NCHW format.
    """
    # Setup test dimensions
    batch_size, num_heads, seq_len, embedding_dims = 8, 4, 32, 16
    max_seq_len = 2048
    compute_dtype = torch.float16

    # Create original RotaryEmbedding and ANE version
    original_rope = RotaryEmbedding(embedding_dims=embedding_dims, dtype=compute_dtype)
    ane_rope = ANERotaryEmbedding(original_rope, max_seq_len=max_seq_len)

    # Test data - standard format and NCHW format for ANE
    inputs = torch.randn((batch_size, seq_len, num_heads, embedding_dims), dtype=compute_dtype)
    inputs_ane = inputs.permute(0, 2, 3, 1)  # [batch, num_heads, embedding_dims, seq_len]

    # Process with ANE implementation
    batch_positions = torch.randint(0, max_seq_len, (batch_size, seq_len))
    original_output = original_rope(inputs, batch_positions)

    # Get sin/cos values already permuted for ANE
    sin, cos = ane_rope(batch_positions, permute_for_ane=True)
    sin, cos = sin.unsqueeze(1), cos.unsqueeze(1) # expand head dimension
    ane_output = apply_rotary_embedding(inputs_ane, sin, cos)
    ane_output = ane_output.permute(0, 3, 1, 2)  # [batch, seq_len, num_heads, embedding_dims]

    # Verify outputs match
    test_case = TestCase()
    test_case.assertTrue(
        torch.allclose(original_output, ane_output, rtol=1e-6, atol=1e-6),
        "ANERotaryEmbedding output doesn't match RotaryEmbedding output"
    )
    print("ANERotaryEmbedding test passed!")


def test_ane_self_attention():
    """
    Test that ANEAttention produces the same outputs as Attention
    for self-attention operations using the ANE-friendly NCHW format.

    This test verifies that:
    1. The ANE implementation correctly handles self-attention queries and keys
    2. The rotary embeddings are properly applied to queries and keys
    3. The attention mask works correctly
    4. The output projection produces results matching the original implementation
    """
    # Setup small test dimensions
    batch_size, seq_len = 2, 8
    embed_dim, head_dim = 12, 14
    num_query_heads, num_kv_heads = 6, 3
    offset = 4
    compute_dtype = torch.float16

    # Create simple config for testing
    config = DiaConfig(
        model=ModelConfig(
            encoder=EncoderConfig(
                n_layer=4,
                n_embd=embed_dim,
                n_hidden=embed_dim * 4,
                n_head=num_query_heads,
                head_dim=head_dim,
            ),
            decoder=DecoderConfig(
                n_layer=4,
                n_embd=embed_dim,
                n_hidden=embed_dim * 4,
                gqa_query_heads=num_query_heads,
                kv_heads=num_kv_heads,
                gqa_head_dim=head_dim,
                cross_query_heads=num_query_heads,
                cross_head_dim=head_dim,
            ),
            rope_min_timescale=1,
            rope_max_timescale=10000,
        ),
        data=DataConfig(
            text_length=128,
            audio_length=128,
        ),
    )

    # Create original Attention and ANE version
    attn = Attention(
        config=config,
        q_embed_dim=embed_dim,
        kv_embed_dim=embed_dim,
        num_query_heads=num_query_heads,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        compute_dtype=compute_dtype,
        is_cross_attn=False,
    )

    # Initialize with random weights
    with torch.no_grad():
        for layer in [attn.q_proj, attn.k_proj, attn.v_proj, attn.o_proj]:
            layer.weight.data.normal_(0, 0.1)

    ane_attn = ANEAttention(attn)

    # Create original rotary embedding and ANE version
    ane_rope = ANERotaryEmbedding(attn.rotary_emb, max_seq_len=1024)

    # Create test inputs
    x_std = torch.randn((batch_size, seq_len, embed_dim), dtype=compute_dtype) * 0.1
    x_ane = x_std.transpose(1, 2).unsqueeze(2)  # [batch, embed_dim, 1, seq_len]

    # Create positions and attention mask
    positions = torch.arange(seq_len, dtype=torch.long).unsqueeze(0).repeat(batch_size, 1) + offset
    attn_mask = create_attn_mask(
        torch.ones((batch_size, seq_len), dtype=torch.bool),
        torch.ones((batch_size, seq_len + offset), dtype=torch.bool),
        x_std.device,
        is_causal=True,
    )
    # attn_mask = None

    k = torch.randn((batch_size, num_kv_heads, seq_len + offset, head_dim), dtype=compute_dtype)
    v = torch.randn((batch_size, num_kv_heads, seq_len + offset, head_dim), dtype=compute_dtype)
    k_ane = k.clone()
    v_ane = v.clone().transpose(-1, -2)
    kv_cache = KVCache.from_kv(k, v)
    kv_cache.current_idx = torch.tensor(offset)

    # Forward pass with standard Attention
    with torch.no_grad():
        y_std = attn(
            Xq=x_std,
            Xkv=x_std,
            q_positions=positions,
            kv_positions=positions,
            attn_mask=attn_mask,
            cache=kv_cache,
            prefill=False,
            # current_idx=torch.arange(8, 16),
            current_idx=torch.arange(offset, offset + seq_len),
        )

    # Precompute sin/cos for ANE
    sin_q, cos_q = ane_rope(positions, permute_for_ane=True)
    sin_q, cos_q = sin_q.unsqueeze(1), cos_q.unsqueeze(1)
    sin_k, cos_k = sin_q, cos_q  # Same positions for self-attention
    attn_mask = torch.where(attn_mask.transpose(-1, -2), 0.0, -torch.inf)
    with torch.no_grad():
        y_ane = ane_attn(
            Xq=x_ane,
            Xkv=x_ane,
            sin_q=sin_q,
            cos_q=cos_q,
            sin_k=sin_k,
            cos_k=cos_k,
            attn_mask=attn_mask,
            cache=(k_ane, v_ane),
            kv_write_idx=torch.tensor(offset, device=x_std.device),
            kv_layer_write_idx=0,
        )
        y_ane = y_ane.squeeze(2).transpose(1, 2)  # Convert back to [batch, seq_len, embed_dim]
    # Verify outputs match
    test_case = TestCase()
    if torch.isnan(y_std).any() or torch.isnan(y_ane).any():
        raise ValueError("Test failed: NaN values detected in outputs")

    # Use slightly higher tolerance for attention operations
    test_case.assertTrue(
        torch.allclose(y_std, y_ane, rtol=1e-4, atol=1e-3),
        "ANEAttention output doesn't match Attention output for self-attention"
    )
    print("ANEAttention self-attention test passed!")


def test_ane_cross_attention():
    """
    Test that ANEAttention produces the same outputs as Attention
    for cross-attention operations using the ANE-friendly NCHW format.

    This test verifies that:
    1. The ANE implementation correctly handles cross-attention between query and key/value from different sources
    2. The rotary embeddings are properly applied to queries and keys with different positions
    3. Cross-attention mask is applied correctly
    4. Precomputed KV cache is used correctly for cross-attention
    """
    # Setup small test dimensions
    batch_size, q_seq_len, kv_seq_len = 2, 4, 10
    q_embed_dim, kv_embed_dim, head_dim = 12, 16, 14
    num_query_heads, num_kv_heads = 6, 3
    compute_dtype = torch.float16

    # Create simple config for testing
    config = DiaConfig(
        model=ModelConfig(
            encoder=EncoderConfig(
                n_layer=4,
                n_embd=kv_embed_dim,
                n_hidden=kv_embed_dim * 4,
                n_head=num_kv_heads,
                head_dim=head_dim,
            ),
            decoder=DecoderConfig(
                n_layer=4,
                n_embd=q_embed_dim,
                n_hidden=q_embed_dim * 4,
                gqa_query_heads=num_query_heads,
                kv_heads=num_kv_heads,
                gqa_head_dim=head_dim,
                cross_query_heads=num_query_heads,
                cross_head_dim=head_dim,
            ),
            rope_min_timescale=1,
            rope_max_timescale=10000,
        ),
        data=DataConfig(
            text_length=128,
            audio_length=128,
        ),
    )

    # Create original cross-attention and ANE version
    attn = Attention(
        config=config,
        q_embed_dim=q_embed_dim,
        kv_embed_dim=kv_embed_dim,  # Different embedding dimension for KV
        num_query_heads=num_query_heads,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        compute_dtype=compute_dtype,
        is_cross_attn=True,  # Important: this is cross-attention
    )

    # Initialize with random weights
    with torch.no_grad():
        for layer in [attn.q_proj, attn.k_proj, attn.v_proj, attn.o_proj]:
            layer.weight.data.normal_(0, 0.1)

    ane_attn = ANEAttention(attn)

    # Create original rotary embedding and ANE version
    ane_rope = ANERotaryEmbedding(attn.rotary_emb, max_seq_len=1024)

    # Create test inputs (query and key/value from different sources)
    x_q_std = torch.randn((batch_size, q_seq_len, q_embed_dim), dtype=compute_dtype) * 0.1

    # Convert to ANE format for query (no need for KV input in cross-attention since we use cache)
    x_q_ane = x_q_std.transpose(1, 2).unsqueeze(2)  # [batch, q_embed_dim, 1, q_seq_len]

    # Create positions with different values for query and key/value
    q_positions = torch.arange(q_seq_len, dtype=torch.long).unsqueeze(0).repeat(batch_size, 1)
    kv_positions = torch.arange(kv_seq_len, dtype=torch.long).unsqueeze(0).repeat(batch_size, 1)

    # Make KV cache with extra batch dimension for multiple layers
    k_cache = torch.randn((batch_size, num_kv_heads, kv_seq_len, head_dim), dtype=compute_dtype)
    v_cache = torch.randn((batch_size, num_kv_heads, kv_seq_len, head_dim), dtype=compute_dtype)

    # Create clones for ANE format
    k_cache_ane = k_cache.clone()
    v_cache_ane = v_cache.clone().transpose(-1, -2)  # Transpose for ANE format [B, H, D, S]

    # Create standard KV cache for the Attention module
    kv_cache = KVCache.from_kv(k_cache, v_cache)

    # Forward pass with standard Attention
    with torch.no_grad():
        y_std = attn(
            Xq=x_q_std,
            Xkv=None,
            q_positions=q_positions,
            cache=kv_cache,
            prefill=False,
            is_causal=False,
        )

    # Precompute sin/cos for ANE with different positions for query and key
    sin_q, cos_q = ane_rope(q_positions, permute_for_ane=True)
    sin_q, cos_q = sin_q.unsqueeze(1), cos_q.unsqueeze(1)

    # Forward pass with ANE Attention
    with torch.no_grad():
        y_ane = ane_attn(
            Xq=x_q_ane,
            Xkv=None,  # Not needed for cross-attention
            sin_q=sin_q,
            cos_q=cos_q,
            cache=(k_cache_ane, v_cache_ane),  # Pre-computed KV cache
            kv_layer_write_idx=0,  # Important for multi-layer indexing
        )
        y_ane = y_ane.squeeze(2).transpose(1, 2)  # Convert back to [batch, seq_len, embed_dim]

    # Verify outputs match
    test_case = TestCase()
    if torch.isnan(y_std).any() or torch.isnan(y_ane).any():
        raise ValueError("Test failed: NaN values detected in outputs")

    # Use slightly higher tolerance for attention operations with float16
    test_case.assertTrue(
        torch.allclose(y_std, y_ane, rtol=1e-4, atol=5e-4),
        "ANEAttention output doesn't match Attention output for cross-attention"
    )
    print("ANEAttention cross-attention test passed!")


def test_ane_encoder_layer():
    """
    Test that ANEEncoderLayer produces outputs close to EncoderLayer
    using the same weights and random input, including rotary embeddings and mask.
    """
    # Setup test dimensions and config
    batch_size, seq_len, embed_dim, n_hidden, n_head, head_dim = 2, 6, 8, 16, 4, 2
    compute_dtype = torch.float32

    config = DiaConfig(
        model=ModelConfig(
            encoder=EncoderConfig(
                n_layer=1,
                n_embd=embed_dim,
                n_hidden=n_hidden,
                n_head=n_head,
                head_dim=head_dim,
            ),
            decoder=DecoderConfig(
                n_layer=1,
                n_embd=embed_dim,
                n_hidden=n_hidden,
                gqa_query_heads=n_head,
                kv_heads=n_head,
                gqa_head_dim=head_dim,
                cross_query_heads=n_head,
                cross_head_dim=head_dim,
            ),
            rope_min_timescale=1,
            rope_max_timescale=10000,
        ),
        data=DataConfig(
            text_length=seq_len,
            audio_length=seq_len,
        ),
    )

    # Create EncoderLayer and ANEEncoderLayer
    encoder_layer = EncoderLayer(config, compute_dtype)
    ane_encoder_layer = ANEEncoderLayer(encoder_layer)

    # Create random input and dummy state
    x = torch.randn((batch_size, seq_len, embed_dim), dtype=compute_dtype)
    positions = torch.arange(seq_len, dtype=torch.long).unsqueeze(0).repeat(batch_size, 1)
    attn_mask = torch.ones((batch_size, 1, seq_len, seq_len), dtype=torch.bool)

    # Prepare rotary embeddings using the encoder's rotary_emb
    rotary_emb = encoder_layer.self_attention.rotary_emb
    ane_rope = ANERotaryEmbedding(rotary_emb, max_seq_len=seq_len+1)
    sin_q, cos_q = ane_rope(positions, permute_for_ane=True)
    # Add head dimension for ANEAttention (B, 1, D/2, T)
    sin_q, cos_q = sin_q.unsqueeze(1), cos_q.unsqueeze(1)

    # Prepare mask for ANE (NCHW): [B, 1, T, T]
    mask = attn_mask
    # mask = mask.to(x.dtype)
    # Convert to additive mask: 0 for keep, -inf for mask out
    mask = torch.where(mask, torch.tensor(0.0, dtype=x.dtype), torch.tensor(-float('inf'), dtype=x.dtype))

    # Forward pass through original EncoderLayer
    state = EncoderInferenceState(
        positions=positions,
        attn_mask=attn_mask,
        max_seq_len=seq_len,
        device=x.device,
        padding_mask=attn_mask,
    )
    with torch.no_grad():
        y_std = encoder_layer(x, state)

    # Prepare input for ANEEncoderLayer (channels first)
    x_ane = x.transpose(1, 2).unsqueeze(2)  # [B, D, 1, T]

    # Forward pass through ANEEncoderLayer
    with torch.no_grad():
        y_ane = ane_encoder_layer(x_ane, attn_mask=mask, sin_q=sin_q, cos_q=cos_q)
        y_ane = y_ane.squeeze(2).transpose(1, 2)  # [B, T, D]

    # Compare outputs
    test_case = TestCase()
    test_case.assertTrue(
        torch.allclose(y_std, y_ane, rtol=1e-4, atol=1e-4),
        "ANEEncoderLayer output doesn't match EncoderLayer output"
    )
    print("ANEEncoderLayer test passed!")


def test_all():
    test_ane_mlp_block()
    test_ane_rotary_embedding()
    test_ane_self_attention()
    test_ane_cross_attention()
    test_ane_encoder_layer()


if __name__ == "__main__":
    with torch.no_grad():
        test_all()
