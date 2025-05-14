import torch
import torch.nn as nn
from unittest import TestCase

from ane_models.dia.dia.dia.layers import (
    MlpBlock,
    RotaryEmbedding,
    Attention,
    EncoderLayer,
    EncoderInferenceState,
    Encoder,
    DecoderLayer,
    DecoderInferenceState,
)
from ane_models.dia.dia.dia.config import (
    DiaConfig,
    ModelConfig,
    EncoderConfig,
    DecoderConfig,
    DataConfig,
)
from ane_models.dia.dia.dia.state import KVCache, create_attn_mask
from ane_models.dia.ane_dia import (
    ANEMlpBlock,
    ANERotaryEmbedding,
    apply_rotary_embedding,
    ANEAttention,
    update_kv_cache,
    ANEEncoderLayer,
    ANERotaryEmbedding,
    ANEEncoder,
    ANEDecoderLayer,
)


def test_ane_mlp_block():
    """
    Test that ANEMlpBlock produces the same outputs as MlpBlock
    using standard normal distribution for weights and inputs.
    """
    torch.random.manual_seed(42)
    batch_size, seq_len, embed_dim, intermediate_dim = 2, 4, 8, 16
    compute_dtype = torch.float32

    mlp_block = MlpBlock(
        embed_dim=embed_dim,
        intermediate_dim=intermediate_dim,
        compute_dtype=compute_dtype,
    )

    with torch.no_grad():
        mlp_block.wi_fused.weight = nn.Parameter(
            torch.randn((embed_dim, 2, intermediate_dim), dtype=compute_dtype)
        )
        mlp_block.wo.weight = nn.Parameter(
            torch.randn((intermediate_dim, embed_dim), dtype=compute_dtype)
        )

    ane_mlp_block = ANEMlpBlock(mlp_block)
    x_std = torch.randn((batch_size, seq_len, embed_dim), dtype=compute_dtype)
    x_ane = x_std.transpose(1, 2).unsqueeze(2)

    y_std = mlp_block(x_std)
    y_ane = ane_mlp_block(x_ane).squeeze(2).transpose(1, 2)

    test_case = TestCase()
    if torch.isnan(y_std).any() or torch.isnan(y_ane).any():
        raise ValueError("Test failed: NaN values detected in outputs")

    test_case.assertTrue(
        torch.allclose(y_std, y_ane, rtol=1e-5, atol=1e-5),
        "ANEMlpBlock output doesn't match MlpBlock output",
    )

    print("ANEMlpBlock test passed!")


def test_ane_rotary_embedding():
    """
    Test that ANERotaryEmbedding produces the same outputs as RotaryEmbedding
    when applied to the same inputs, using the ANE-friendly NCHW format.
    """
    torch.random.manual_seed(42)
    batch_size, num_heads, seq_len, embedding_dims = 8, 4, 32, 16
    max_seq_len = 2048
    compute_dtype = torch.float16

    original_rope = RotaryEmbedding(embedding_dims=embedding_dims, dtype=compute_dtype)
    ane_rope = ANERotaryEmbedding(original_rope, max_seq_len=max_seq_len)

    inputs = torch.randn(
        (batch_size, seq_len, num_heads, embedding_dims), dtype=compute_dtype
    )
    inputs_ane = inputs.permute(
        0, 2, 3, 1
    )  # [batch, num_heads, embedding_dims, seq_len]

    batch_positions = torch.randint(0, max_seq_len, (batch_size, seq_len))
    original_output = original_rope(inputs, batch_positions)

    sin, cos = ane_rope(batch_positions, permute_for_ane=True)
    sin, cos = sin.unsqueeze(1), cos.unsqueeze(1)  # expand head dimension
    ane_output = apply_rotary_embedding(inputs_ane, sin, cos)
    ane_output = ane_output.permute(
        0, 3, 1, 2
    )  # [batch, seq_len, num_heads, embedding_dims]

    test_case = TestCase()
    test_case.assertTrue(
        torch.allclose(original_output, ane_output, rtol=1e-6, atol=1e-6),
        "ANERotaryEmbedding output doesn't match RotaryEmbedding output",
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
    torch.random.manual_seed(42)
    batch_size, seq_len = 2, 8
    embed_dim, head_dim = 12, 14
    num_query_heads, num_kv_heads = 6, 3
    offset = 4
    compute_dtype = torch.float16

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

    with torch.no_grad():
        for layer in [attn.q_proj, attn.k_proj, attn.v_proj, attn.o_proj]:
            layer.weight.data.normal_(0, 0.2)

    ane_attn = ANEAttention(attn)
    ane_rope = ANERotaryEmbedding(attn.rotary_emb, max_seq_len=1024)

    x_std = torch.randn((batch_size, seq_len, embed_dim), dtype=compute_dtype) * 0.2
    x_ane = x_std.transpose(1, 2).unsqueeze(2)  # [batch, embed_dim, 1, seq_len]

    positions = (
        torch.arange(seq_len, dtype=torch.long).unsqueeze(0).repeat(batch_size, 1)
        + offset
    )
    attn_mask = create_attn_mask(
        torch.ones((batch_size, seq_len), dtype=torch.bool),
        torch.ones((batch_size, seq_len + offset), dtype=torch.bool),
        x_std.device,
        is_causal=True,
    )

    k = torch.randn(
        (batch_size, num_kv_heads, seq_len + offset, head_dim), dtype=compute_dtype
    )
    v = torch.randn(
        (batch_size, num_kv_heads, seq_len + offset, head_dim), dtype=compute_dtype
    )
    k_ane = k.clone()
    v_ane = v.clone().transpose(-1, -2)
    kv_cache = KVCache.from_kv(k, v)
    kv_cache.current_idx = torch.tensor(offset)

    with torch.no_grad():
        y_std = attn(
            Xq=x_std,
            Xkv=x_std,
            q_positions=positions,
            kv_positions=positions,
            attn_mask=attn_mask,
            cache=kv_cache,
            prefill=False,
            current_idx=torch.arange(offset, offset + seq_len),
        )

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
        y_ane = y_ane.squeeze(2).transpose(
            1, 2
        )  # Convert back to [batch, seq_len, embed_dim]
    test_case = TestCase()
    if torch.isnan(y_std).any() or torch.isnan(y_ane).any():
        raise ValueError("Test failed: NaN values detected in outputs")

    test_case.assertTrue(
        torch.allclose(y_std, y_ane, rtol=1e-4, atol=1e-3),
        "ANEAttention output doesn't match Attention output for self-attention",
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
    torch.random.manual_seed(42)
    batch_size, q_seq_len, kv_seq_len = 2, 4, 10
    q_embed_dim, kv_embed_dim, head_dim = 12, 16, 14
    num_query_heads, num_kv_heads = 6, 3
    compute_dtype = torch.float16

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

    with torch.no_grad():
        for layer in [attn.q_proj, attn.k_proj, attn.v_proj, attn.o_proj]:
            layer.weight.data.normal_(0, 0.2)

    ane_attn = ANEAttention(attn)
    ane_rope = ANERotaryEmbedding(attn.rotary_emb, max_seq_len=1024)

    x_q_std = (
        torch.randn((batch_size, q_seq_len, q_embed_dim), dtype=compute_dtype) * 0.2
    )
    x_q_ane = x_q_std.transpose(1, 2).unsqueeze(2)  # [batch, q_embed_dim, 1, q_seq_len]
    q_positions = (
        torch.arange(q_seq_len, dtype=torch.long).unsqueeze(0).repeat(batch_size, 1)
    )

    k_cache = torch.randn(
        (batch_size, num_kv_heads, kv_seq_len, head_dim), dtype=compute_dtype
    )
    v_cache = torch.randn(
        (batch_size, num_kv_heads, kv_seq_len, head_dim), dtype=compute_dtype
    )

    k_cache_ane = k_cache.clone()
    v_cache_ane = v_cache.clone().transpose(-1, -2)
    kv_cache = KVCache.from_kv(k_cache, v_cache)

    with torch.no_grad():
        y_std = attn(
            Xq=x_q_std,
            Xkv=None,
            q_positions=q_positions,
            cache=kv_cache,
            prefill=False,
            is_causal=False,
        )

    sin_q, cos_q = ane_rope(q_positions, permute_for_ane=True)
    sin_q, cos_q = sin_q.unsqueeze(1), cos_q.unsqueeze(1)

    with torch.no_grad():
        y_ane = ane_attn(
            Xq=x_q_ane,
            Xkv=None,  # Not needed for cross-attention
            sin_q=sin_q,
            cos_q=cos_q,
            cache=(k_cache_ane, v_cache_ane),  # Pre-computed KV cache
            kv_layer_write_idx=0,  # Important for multi-layer indexing
        )
        y_ane = y_ane.squeeze(2).transpose(
            1, 2
        )  # Convert back to [batch, seq_len, embed_dim]

    test_case = TestCase()
    if torch.isnan(y_std).any() or torch.isnan(y_ane).any():
        raise ValueError("Test failed: NaN values detected in outputs")

    test_case.assertTrue(
        torch.allclose(y_std, y_ane, rtol=1e-4, atol=2e-3),
        "ANEAttention output doesn't match Attention output for cross-attention",
    )
    print("ANEAttention cross-attention test passed!")


def test_ane_encoder_layer():
    """
    Test that ANEEncoderLayer produces outputs close to EncoderLayer
    using the same weights and random input, including rotary embeddings and mask.
    """
    torch.random.manual_seed(41)  # seedmaxxing
    batch_size, seq_len, embed_dim, n_hidden, n_head, head_dim = 2, 6, 8, 16, 4, 2
    pad = 4
    compute_dtype = torch.float16

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
        data=DataConfig(text_length=seq_len, audio_length=seq_len, text_pad_value=0),
    )

    # Create standard and ANE encoder layers
    encoder_layer = EncoderLayer(config, compute_dtype)
    ane_encoder_layer = ANEEncoderLayer(encoder_layer)

    # Initialize weights explicitly
    with torch.no_grad():
        for name, param in encoder_layer.self_attention.named_parameters():
            if "weight" in name:
                param.data.normal_(0, 0.2)
        for name, param in encoder_layer.mlp.named_parameters():
            if "weight" in name:
                param.data.normal_(0, 0.2)
        for name, param in encoder_layer.named_parameters():
            if "weight" in name and "norm" in name:
                param.data.fill_(1.0)

    # Create test input data
    x = torch.randn((batch_size, seq_len + pad, embed_dim), dtype=compute_dtype) * 0.2
    positions = (
        torch.arange(seq_len + pad, dtype=torch.long).unsqueeze(0).repeat(batch_size, 1)
    )
    padding_mask = torch.zeros((batch_size, seq_len + pad), dtype=torch.bool)
    padding_mask[:, :seq_len] = 1
    attn_mask = create_attn_mask(padding_mask, padding_mask, x.device, is_causal=False)

    # Prepare ANE input format
    x_ane = x.transpose(1, 2).unsqueeze(2)  # [B, D, 1, T]
    ane_mask = attn_mask.transpose(-1, -2)
    ane_mask = torch.where(
        ane_mask,
        torch.tensor(0.0, dtype=x.dtype),
        torch.tensor(-float("inf"), dtype=x.dtype),
    )

    # Prepare rotary embeddings
    rotary_emb = encoder_layer.self_attention.rotary_emb
    ane_rope = ANERotaryEmbedding(rotary_emb, max_seq_len=seq_len + 1 + pad)
    sin_q, cos_q = ane_rope(positions, permute_for_ane=True)
    sin_q, cos_q = sin_q.unsqueeze(1), cos_q.unsqueeze(1)

    # Test 1: Test pre_sa_norm
    with torch.no_grad():
        # Standard
        y_std = encoder_layer.pre_sa_norm(x)
        # ANE
        y_ane = ane_encoder_layer.pre_sa_norm(x_ane)
        y_ane = y_ane.squeeze(2).transpose(1, 2)  # Convert back to [B, T, D]

        assert torch.allclose(
            y_std, y_ane, rtol=1e-4, atol=2e-3
        ), "pre_sa_norm output mismatch"

    # Test 2: Test self_attention
    with torch.no_grad():
        # Standard
        x_norm = encoder_layer.pre_sa_norm(x).to(compute_dtype)
        y_std = encoder_layer.self_attention(
            Xq=x_norm,
            Xkv=x_norm,
            q_positions=positions,
            kv_positions=positions,
            attn_mask=attn_mask,
        )
        # ANE
        x_norm_ane = ane_encoder_layer.pre_sa_norm(x_ane)
        y_ane = ane_encoder_layer.self_attention(
            Xq=x_norm_ane,
            Xkv=x_norm_ane,
            sin_q=sin_q,
            cos_q=cos_q,
            sin_k=sin_q,
            cos_k=cos_q,
            attn_mask=ane_mask,
        )
        y_ane = y_ane.squeeze(2).transpose(1, 2)  # Convert back to [B, T, D]

        assert torch.allclose(
            y_std, y_ane, rtol=1e-4, atol=1e-3
        ), "self_attention output mismatch"

    # Test 3: Test post_sa_norm
    with torch.no_grad():
        # Standard
        y_std = encoder_layer.post_sa_norm(x)
        # ANE
        y_ane = ane_encoder_layer.post_sa_norm(x_ane)
        y_ane = y_ane.squeeze(2).transpose(1, 2)  # Convert back to [B, T, D]
        assert torch.allclose(
            y_std, y_ane, rtol=1e-4, atol=2e-3
        ), "post_sa_norm output mismatch"

    # Test 4: Test MLP
    with torch.no_grad():
        # Standard
        y_std = encoder_layer.mlp(x)
        # ANE
        y_ane = ane_encoder_layer.mlp(x_ane)
        y_ane = y_ane.squeeze(2).transpose(1, 2)  # Convert back to [B, T, D]
        assert torch.allclose(y_std, y_ane, rtol=1e-4, atol=1e-3), "MLP output mismatch"

    # Test 5: Test full encoder layer
    with torch.no_grad():
        # Standard
        state = EncoderInferenceState(
            max_seq_len=seq_len + 1 + pad,
            device=x.device,
            positions=positions,
            padding_mask=padding_mask,
            attn_mask=attn_mask,
        )
        y_std = encoder_layer(x, state)

        # ANE
        y_ane = ane_encoder_layer(x_ane, attn_mask=ane_mask, sin_q=sin_q, cos_q=cos_q)
        y_ane = y_ane.squeeze(2).transpose(1, 2)  # Convert back to [B, T, D]
        assert torch.allclose(
            y_std, y_ane, rtol=1e-4, atol=1e-3
        ), "EncoderLayer output mismatch"

    print("ANEEncoderLayer test passed!")
    return True


def test_ane_encoder():
    """
    Test that ANEEncoder produces the same outputs as Encoder
    using the same weights and random input.
    """
    torch.random.manual_seed(42)
    # Setup test dimensions and config
    batch_size, seq_len, embed_dim, n_hidden, n_head, head_dim = 2, 6, 8, 16, 4, 2
    pad = 4
    vocab_size = 100
    compute_dtype = torch.float16

    config = DiaConfig(
        model=ModelConfig(
            src_vocab_size=vocab_size,
            encoder=EncoderConfig(
                n_layer=2,  # Test with 2 layers
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
        data=DataConfig(text_length=seq_len, audio_length=seq_len, text_pad_value=0),
    )

    # Create Encoder and ANEEncoder
    encoder = Encoder(config, compute_dtype)

    # Initialize weights explicitly
    with torch.no_grad():
        # Initialize token embeddings
        if hasattr(encoder, "token_embedding"):
            encoder.token_embedding.weight.data.normal_(0, 0.02)

        # Initialize positional embeddings if they exist
        if hasattr(encoder, "pos_embedding"):
            encoder.pos_embedding.weight.data.normal_(0, 0.02)

        # Initialize layer weights
        for layer in encoder.layers:
            # Initialize self-attention weights
            for name, param in layer.self_attention.named_parameters():
                if "weight" in name:
                    param.data.normal_(0, 0.02)
            # Initialize MLP weights
            for name, param in layer.mlp.named_parameters():
                if "weight" in name:
                    param.data.normal_(0, 0.02)
            # Initialize layer norm weights
            for name, param in layer.named_parameters():
                if "weight" in name and "norm" in name:
                    param.data.fill_(1.0)

    ane_encoder = ANEEncoder(encoder, max_seq_len=seq_len + 1 + pad)

    # Create random input
    x_ids = torch.randint(1, vocab_size, (batch_size, seq_len + pad), dtype=torch.long)
    x_ids[:, seq_len:] = 0  # Set padding tokens to 0

    # Create padding mask (1 for real tokens, 0 for padding)
    padding_mask = torch.cat(
        (
            torch.ones((batch_size, seq_len), dtype=torch.bool),
            torch.zeros((batch_size, pad), dtype=torch.bool),
        ),
        dim=1,
    ).to(x_ids.device)

    # Create attention mask using the same function as in EncoderInferenceState
    attn_mask = create_attn_mask(
        padding_mask, padding_mask, x_ids.device, is_causal=False
    )

    # Create positions
    positions = torch.arange(seq_len + pad, dtype=torch.long, device=x_ids.device)
    positions = positions.unsqueeze(0).repeat(batch_size, 1)

    # Create state
    state = EncoderInferenceState(
        max_seq_len=seq_len + pad,
        device=x_ids.device,
        positions=positions,
        padding_mask=padding_mask,
        attn_mask=attn_mask,
    )

    # Forward pass through original Encoder
    with torch.no_grad():
        y_std = encoder(x_ids, state)

    # Forward pass through ANEEncoder
    with torch.no_grad():
        y_ane = ane_encoder(x_ids, positions=positions, padding_mask=padding_mask)
        y_ane = y_ane.squeeze(2).transpose(1, 2)  # [B, T, D]

    # Compare outputs
    test_case = TestCase()
    test_case.assertTrue(
        torch.allclose(y_std, y_ane, rtol=1e-4, atol=1e-3),
        "ANEEncoder output doesn't match Encoder output",
    )
    print("ANEEncoder test passed!")


def test_ane_decoder_layer_with_caches():
    """
    Test ANEDecoderLayer with pre-populated KV caches for both self-attention and cross-attention.
    """
    torch.random.manual_seed(42)

    # Test configuration
    batch_size = 2
    seq_len = 1  # Single decoding step
    cross_kv_cache_len = 3  # 3 real tokens + 3 padding
    cross_kv_padding_len = 3
    self_kv_cache_len = 8  # 4 real tokens + 4 empty slots
    offset = 4
    embed_dim = 32
    num_heads = 2
    gqa_query_heads = 4
    head_dim = 16
    compute_dtype = torch.float16

    # Create a test configuration
    config = DiaConfig(
        model=ModelConfig(
            encoder=EncoderConfig(
                n_layer=2,
                n_embd=embed_dim,
                n_hidden=embed_dim * 4,
                n_head=num_heads,
                head_dim=head_dim,
            ),
            decoder=DecoderConfig(
                n_layer=2,
                n_embd=embed_dim,
                n_hidden=embed_dim * 4,
                gqa_query_heads=gqa_query_heads,
                kv_heads=num_heads,
                gqa_head_dim=head_dim,
                cross_query_heads=num_heads,
                cross_head_dim=head_dim,
            ),
            rope_min_timescale=1,
            rope_max_timescale=10000,
        ),
        data=DataConfig(
            text_length=cross_kv_cache_len,
            audio_length=self_kv_cache_len,
        ),
    )

    # Create standard and ANE decoder layers
    decoder_layer = DecoderLayer(config, compute_dtype)
    ane_decoder_layer = ANEDecoderLayer(decoder_layer)

    # Initialize weights explicitly for both layers
    with torch.no_grad():
        # Initialize self-attention weights
        for proj in [
            decoder_layer.self_attention.q_proj,
            decoder_layer.self_attention.k_proj,
            decoder_layer.self_attention.v_proj,
            decoder_layer.self_attention.o_proj,
        ]:
            nn.init.normal_(proj.weight, mean=0.0, std=0.2)
            if hasattr(proj, "bias") and proj.bias is not None:
                nn.init.zeros_(proj.bias)

        # Initialize cross-attention weights
        for proj in [
            decoder_layer.cross_attention.q_proj,
            decoder_layer.cross_attention.k_proj,
            decoder_layer.cross_attention.v_proj,
            decoder_layer.cross_attention.o_proj,
        ]:
            nn.init.normal_(proj.weight, mean=0.0, std=0.2)
            if hasattr(proj, "bias") and proj.bias is not None:
                nn.init.zeros_(proj.bias)

        # Initialize MLP weights
        nn.init.normal_(decoder_layer.mlp.wi_fused.weight, mean=0.0, std=0.2)
        nn.init.normal_(decoder_layer.mlp.wo.weight, mean=0.0, std=0.2)

    # Create test input data (single decoding step)
    x = torch.randn((batch_size, seq_len, embed_dim), dtype=compute_dtype) * 0.2
    x_ane = x.transpose(1, 2).unsqueeze(2)  # [B, D, 1, 1]
    positions = torch.arange(offset, offset + seq_len, dtype=torch.int32)

    self_attn_cache = KVCache.from_kv(
        torch.randn(
            (batch_size, num_heads, self_kv_cache_len, head_dim), dtype=compute_dtype
        ),
        torch.randn(
            (batch_size, num_heads, self_kv_cache_len, head_dim), dtype=compute_dtype
        ),
    )
    self_attn_cache.current_idx = torch.tensor(offset)
    ane_self_attn_cache = (
        self_attn_cache.k.detach().clone(),
        self_attn_cache.v.detach().clone().transpose(-1, -2),
    )

    cross_attn_cache = KVCache.from_kv(
        torch.randn(
            (batch_size, num_heads, cross_kv_cache_len, head_dim), dtype=compute_dtype
        )
        * 0.2,
        torch.randn(
            (batch_size, num_heads, cross_kv_cache_len, head_dim), dtype=compute_dtype
        )
        * 0.2,
    )
    cross_attn_cache.current_idx = torch.tensor(3)  # 3 real tokens
    ane_cross_attn_cache_k = torch.cat(
        (
            cross_attn_cache.k.detach().clone(),
            torch.zeros(
                (batch_size, num_heads, cross_kv_padding_len, head_dim),
                dtype=compute_dtype,
            ),
        ),
        dim=2,
    )
    ane_cross_attn_cache_v = torch.cat(
        (
            cross_attn_cache.v.detach().clone(),
            torch.zeros(
                (batch_size, num_heads, cross_kv_padding_len, head_dim),
                dtype=compute_dtype,
            ),
        ),
        dim=2,
    )
    ane_cross_attn_cache = (
        ane_cross_attn_cache_k,
        ane_cross_attn_cache_v.transpose(-1, -2),
    )

    std_output = decoder_layer(
        x,
        state=DecoderInferenceState(
            device=x.device,
            dtype=compute_dtype,
            enc_out=None,
            enc_positions=None,
            dec_positions=positions,
            self_attn_cache=self_attn_cache,
            cross_attn_cache=cross_attn_cache,
            casual_attn_mask=torch.tril(
                torch.ones(
                    (self_kv_cache_len, self_kv_cache_len),
                    dtype=torch.bool,
                    device=x.device,
                )
            ),
        ),
        self_attn_cache=self_attn_cache,
        cross_attn_cache=cross_attn_cache,
        current_idx=positions,
    )
    print(std_output)

    self_attn_mask = torch.tril(
        torch.ones(
            (batch_size, 1, seq_len, self_kv_cache_len),
            dtype=torch.bool,
            device=x.device,
        )
    )
    cross_attn_mask = torch.cat(
        (
            torch.ones(
                (batch_size, 1, seq_len, cross_kv_cache_len),
                dtype=torch.bool,
                device=x.device,
            ),
            torch.zeros(
                (batch_size, 1, seq_len, cross_kv_padding_len),
                dtype=torch.bool,
                device=x.device,
            ),
        ),
        dim=3,
    )
    self_attn_mask = torch.where(
        self_attn_mask,
        torch.tensor(0.0, dtype=compute_dtype),
        torch.tensor(-float("inf"), dtype=compute_dtype),
    ).transpose(-1, -2)
    cross_attn_mask = torch.where(
        cross_attn_mask,
        torch.tensor(0.0, dtype=compute_dtype),
        torch.tensor(-float("inf"), dtype=compute_dtype),
    ).transpose(-1, -2)

    ane_rotary_emb = ANERotaryEmbedding(
        decoder_layer.self_attention.rotary_emb, max_seq_len=self_kv_cache_len
    )
    sin_q, cos_q = ane_rotary_emb(positions.unsqueeze(0), permute_for_ane=True)
    sin_q, cos_q = sin_q.unsqueeze(1), cos_q.unsqueeze(1)  # Add head dimension

    ane_output = (
        ane_decoder_layer(
            x_ane,
            sin_q=sin_q,
            cos_q=cos_q,
            kv_write_index=torch.tensor(offset),
            kv_layer_write_idx=0,
            self_attn_mask=self_attn_mask,
            cross_attn_mask=cross_attn_mask,
            enc_out=None,
            self_attn_cache=ane_self_attn_cache,
            cross_attn_cache=ane_cross_attn_cache,
        )
        .squeeze(2)
        .transpose(1, 2)
    )
    print(ane_output)

    assert torch.allclose(std_output, ane_output, rtol=1e-4, atol=1e-3)
    print("ANE Decoder Layer with Caches test passed!")


def test_ane_decoder_self_attention():
    torch.random.manual_seed(42)

    # Test configuration
    batch_size = 2
    seq_len = 3  # Single decoding step
    self_kv_cache_len = 8  # 4 real tokens + 4 empty slots
    offset = 4
    embed_dim = 32
    num_heads = 2
    gqa_query_heads = 4
    head_dim = 16
    compute_dtype = torch.float16

    # Create a test configuration
    config = DiaConfig(
        model=ModelConfig(
            encoder=EncoderConfig(
                n_layer=2,
                n_embd=embed_dim,
                n_hidden=embed_dim * 4,
                n_head=num_heads,
                head_dim=head_dim,
            ),
            decoder=DecoderConfig(
                n_layer=2,
                n_embd=embed_dim,
                n_hidden=embed_dim * 4,
                gqa_query_heads=gqa_query_heads,
                kv_heads=num_heads,
                gqa_head_dim=head_dim,
                cross_query_heads=num_heads,
                cross_head_dim=head_dim,
            ),
            rope_min_timescale=1,
            rope_max_timescale=10000,
        ),
        data=DataConfig(
            text_length=self_kv_cache_len,
            audio_length=self_kv_cache_len,
        ),
    )

    # Create standard and ANE decoder layers
    decoder_layer = DecoderLayer(config, compute_dtype)
    ane_decoder_layer = ANEDecoderLayer(decoder_layer)

    # Initialize weights explicitly for both layers
    with torch.no_grad():
        # Initialize self-attention weights
        for proj in [
            decoder_layer.self_attention.q_proj,
            decoder_layer.self_attention.k_proj,
            decoder_layer.self_attention.v_proj,
            decoder_layer.self_attention.o_proj,
        ]:
            nn.init.normal_(proj.weight, mean=0.0, std=0.2)
            if hasattr(proj, "bias") and proj.bias is not None:
                nn.init.zeros_(proj.bias)

    x = torch.randn((batch_size, seq_len, embed_dim), dtype=compute_dtype) * 0.2
    x_ane = x.transpose(1, 2).unsqueeze(2)
    positions = torch.arange(seq_len, device=x.device) + offset

    self_attn_cache = KVCache.from_kv(
        torch.randn(
            (batch_size, num_heads, self_kv_cache_len, head_dim), dtype=compute_dtype
        ),
        torch.randn(
            (batch_size, num_heads, self_kv_cache_len, head_dim), dtype=compute_dtype
        ),
    )
    self_attn_cache.current_idx = torch.tensor(offset)
    ane_self_attn_cache = (
        self_attn_cache.k.detach().clone(),
        self_attn_cache.v.detach().clone().transpose(-1, -2),
    )

    std_output = decoder_layer.self_attention(
        Xq=x,
        Xkv=x,
        q_positions=positions,
        kv_positions=positions,
        attn_mask=None,
        cache=self_attn_cache,
        prefill=False,
        is_causal=True,
        current_idx=positions,
    )

    ane_rotary_emb = ANERotaryEmbedding(
        decoder_layer.self_attention.rotary_emb, max_seq_len=self_kv_cache_len
    )
    sin_q, cos_q = ane_rotary_emb(positions.unsqueeze(0), permute_for_ane=True)
    sin_q, cos_q = sin_q.unsqueeze(1), cos_q.unsqueeze(1)  # Add head dimension
    # make causal mask in which source is of length seq_len and target is the length of the cache
    attn_mask = (
        torch.tril(torch.ones(seq_len, self_kv_cache_len, dtype=torch.bool))
        .unsqueeze(0)
        .unsqueeze(0)
    )
    attn_mask = attn_mask.transpose(-1, -2)
    attn_mask = torch.where(
        attn_mask,
        torch.tensor(0.0, dtype=compute_dtype),
        torch.tensor(-float("inf"), dtype=compute_dtype),
    )
    ane_output = ane_decoder_layer.self_attention(
        Xq=x_ane,
        Xkv=x_ane,
        sin_q=sin_q,
        cos_q=cos_q,
        sin_k=sin_q,
        cos_k=cos_q,
        attn_mask=attn_mask,
        cache=ane_self_attn_cache,
        kv_write_idx=torch.tensor(offset),
        kv_layer_write_idx=0,
    )
    ane_output = ane_output.squeeze(2).transpose(1, 2)
    assert torch.allclose(std_output, ane_output, rtol=1e-4, atol=1e-3)
    print("ANEDecoderLayer self-attention test passed!")
    return True


def test_ane_decoder_cross_attention():
    torch.random.manual_seed(42)

    # Test configuration
    batch_size = 2
    seq_len = 3  # Single decoding step
    cross_kv_cache_len = 4  # 8 real tokens
    cross_kv_padding_len = 4
    embed_dim = 32
    num_heads = 2
    gqa_query_heads = 4
    head_dim = 16
    compute_dtype = torch.float16
    offset = 4

    # Create a test configuration
    config = DiaConfig(
        model=ModelConfig(
            encoder=EncoderConfig(
                n_layer=2,
                n_embd=embed_dim,
                n_hidden=embed_dim * 4,
                n_head=num_heads,
                head_dim=head_dim,
            ),
            decoder=DecoderConfig(
                n_layer=2,
                n_embd=embed_dim,
                n_hidden=embed_dim * 4,
                gqa_query_heads=gqa_query_heads,
                kv_heads=num_heads,
                gqa_head_dim=head_dim,
                cross_query_heads=num_heads,
                cross_head_dim=head_dim,
            ),
            rope_min_timescale=1,
            rope_max_timescale=10000,
        ),
        data=DataConfig(
            text_length=10,  # Arbitrary, not used directly
            audio_length=cross_kv_cache_len,
        ),
    )

    # Create standard and ANE decoder layers
    decoder_layer = DecoderLayer(config, compute_dtype)
    ane_decoder_layer = ANEDecoderLayer(decoder_layer)

    # Initialize weights explicitly for both layers
    with torch.no_grad():
        # Initialize cross-attention weights
        for proj in [
            decoder_layer.cross_attention.q_proj,
            decoder_layer.cross_attention.k_proj,
            decoder_layer.cross_attention.v_proj,
            decoder_layer.cross_attention.o_proj,
        ]:
            nn.init.normal_(proj.weight, mean=0.0, std=0.2)
            if hasattr(proj, "bias") and proj.bias is not None:
                nn.init.zeros_(proj.bias)

    # Input tensors
    x = torch.randn((batch_size, seq_len, embed_dim), dtype=compute_dtype) * 0.2
    x_ane = x.transpose(1, 2).unsqueeze(2)  # Convert to NCHW format

    # Encoder output (cross-attention key/value source)
    cross_attn_kv_cache = KVCache.from_kv(
        torch.randn(
            (batch_size, num_heads, cross_kv_cache_len, head_dim), dtype=compute_dtype
        )
        * 0.2,
        torch.randn(
            (batch_size, num_heads, cross_kv_cache_len, head_dim), dtype=compute_dtype
        )
        * 0.2,
    )

    cross_attn_kv_cache_ane_k = torch.cat(
        (
            cross_attn_kv_cache.k,
            torch.randn(
                (batch_size, num_heads, cross_kv_padding_len, head_dim),
                dtype=compute_dtype,
            )
            * 0.2,
        ),
        dim=2,
    )
    cross_attn_kv_cache_ane_v = torch.cat(
        (
            cross_attn_kv_cache.v,
            torch.randn(
                (batch_size, num_heads, cross_kv_padding_len, head_dim),
                dtype=compute_dtype,
            )
            * 0.2,
        ),
        dim=2,
    )
    cross_attn_kv_cache_ane = (
        cross_attn_kv_cache_ane_k,
        cross_attn_kv_cache_ane_v.transpose(-1, -2),
    )

    # Create attention masks
    cross_attn_mask = torch.ones(
        (batch_size, 1, seq_len, cross_kv_cache_len), dtype=torch.bool
    )
    cross_attn_mask_ane = torch.cat(
        (
            cross_attn_mask,
            torch.zeros(
                (batch_size, 1, seq_len, cross_kv_padding_len), dtype=torch.bool
            ),
        ),
        dim=3,
    )
    cross_attn_mask_ane = torch.where(
        cross_attn_mask_ane,
        torch.tensor(0.0, dtype=compute_dtype),
        torch.tensor(-float("inf"), dtype=compute_dtype),
    )
    cross_attn_mask_ane = cross_attn_mask_ane.transpose(-1, -2)

    # Create rotary embeddings
    positions = torch.arange(seq_len, device=x.device) + offset
    rotary_emb = ANERotaryEmbedding(
        decoder_layer.cross_attention.rotary_emb, max_seq_len=4096
    )
    sin_q, cos_q = rotary_emb(positions, permute_for_ane=True)
    sin_q_ane, cos_q_ane = sin_q.unsqueeze(0).unsqueeze(0), cos_q.unsqueeze(
        0
    ).unsqueeze(
        0
    )  # [1, 1, D/2, T]

    # Standard cross-attention
    std_output = decoder_layer.cross_attention(
        Xq=x,
        Xkv=None,
        q_positions=positions,
        kv_positions=None,
        cache=cross_attn_kv_cache,
        current_idx=None,
    )

    # ANE cross-attention
    ane_output = ane_decoder_layer.cross_attention(
        Xq=x_ane,
        Xkv=None,
        sin_q=sin_q_ane,
        cos_q=cos_q_ane,
        # sin_k=sin_q_ane,  # Using same sin/cos for simplicity
        # cos_k=cos_q_ane,
        attn_mask=cross_attn_mask_ane,
        cache=cross_attn_kv_cache_ane,
        kv_layer_write_idx=0,
    )
    ane_output = ane_output.squeeze(2).transpose(1, 2)  # Convert back to [B, T, D]

    # Compare outputs
    assert torch.allclose(std_output, ane_output, rtol=1e-4, atol=1e-3)
    print("ANEDecoderLayer cross-attention test passed!")
    return True


# Add to test_all()
def test_all():
    test_ane_mlp_block()
    test_ane_rotary_embedding()
    test_ane_self_attention()
    test_ane_cross_attention()
    test_ane_encoder_layer()
    test_ane_encoder()
    test_ane_decoder_self_attention()
    test_ane_decoder_cross_attention()
    # test_ane_decoder_layer_with_caches()


if __name__ == "__main__":
    with torch.no_grad():
        test_all()
