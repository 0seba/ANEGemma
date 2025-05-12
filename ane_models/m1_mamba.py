import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from third_party.M1.mamba.hybrid_mamba_layer import Mamba as _PyMamba


def apply_linear_as_conv2d(layer, x):
    return F.conv2d(x, layer.weight.unsqueeze(-1).unsqueeze(-1))


def repeat_kv(x, n_rep):
    if n_rep == 1:
        return x
    batch, num_key_value_heads, _, slen, head_dim = x.shape
    x = x.expand(
        batch,
        num_key_value_heads,
        n_rep,
        slen,
        head_dim,
    )
    x = x.reshape(
        batch,
        num_key_value_heads * n_rep,
        slen,
        head_dim,
    )
    return x

def pscan(A: torch.Tensor, X: torch.Tensor):
    # https://github.com/alxndrTL/mamba.py/blob/2cc168c78b2494557f45072c4b8daa9d6f5608c7/mambapy/pscan.py#L37
    # A: (B, Ngroup, Dstate, L)
    # X: (B, Ngroup, Dinner, Dstate, L)
    B, Ngroup, D, Dstate, L = X.size()
    num_steps = int(math.log2(L))

    # for my purposes L is a power of 2
    # ANE supports maximum 5 dimensions, I'm going to squeeze batch dimension
    levels = []
    Aa = A.flatten(0, 1)
    Xa = X.flatten(0, 1)
    for i in range(num_steps-2):
        T = Xa.size(-1)
        Aa = Aa.view(B * Ngroup, 1, Dstate, T // 2, 2)
        Xa = Xa.view(B * Ngroup, D, Dstate, T // 2, 2)
        Aa0, Aa1 = torch.chunk(Aa, 2, -1)
        Xa0, Xa1 = torch.chunk(Xa, 2, -1)
        
        Xa = Xa1 + Aa1.mul(Xa0)
        Aa = Aa1 * Aa0
        
        # Xa[..., 1].add_(Aa[..., 1].mul(Xa[..., 0]))
        # Aa[..., 1].mul_(Aa[..., 0])

        # Aa = Aa[..., 1]
        # Xa = Xa[..., 1]
    # we have only 4, 2 or 1 nodes left
    if Xa.size(-1) == 4:
        Xa0, Xa1, Xa2, Xa3 = torch.chunk(Xa, 4, -1)
        Aa0, Aa1, Aa2, Aa3 = torch.chunk(Aa, 4, -1)

        Xa1 = Xa1 + Aa1 * Xa0
        Aa1 = Aa1 * Aa0
        Xa2 = Xa2 + Aa2 * Xa1
        Xa3 = Xa3 * Aa3 * Xa2
        # Xa[..., 1].add_(Aa[..., 1].mul(Xa[..., 0]))
        # Aa[..., 1].mul_(Aa[..., 0])
        # Xa[..., 3].add_(Aa[..., 3].mul(Xa[..., 2] + Aa[..., 2].mul(Xa[..., 1])))
        # return Xa.view(B, Ngroup, D, Dstate, L)
    elif Xa.size(-1) == 2:
        Xa0, Xa1 = torch.chunk(Xa, 2, -1)
        Aa0, Aa1 = torch.chunk(Aa, 2, -1)
        Xa = Xa1 + Aa1.mul(Xa0)
        # Xa[..., 1].add_(Aa[..., 1].mul(Xa[..., 0]))



    return Xa.view(B, Ngroup, D, Dstate, L)

def selective_scan_fn(x, dt, A, B, C, D, z, delta_bias, delta_softplus, return_last_state):
    print("x", x.size())
    print("dt", dt.size())
    print("A", A.size())
    print("B", B.size())
    print("C", C.size())
    print("D", D.size())
    print("z", z.size())
    print("delta_bias", delta_bias.size())

    # x: (B, Dinner, 1, L)
    # dt: (B, Dinner, 1, L)
    # A: (Dinner, Dstate, 1)
    # B: (B, Ngroup, 1, Dstate, L)
    # C: (B, Ngroup, Dstate, 1, L)
    # D: (Dinner, 1, 1)
    # z: (B, Dinner, 1, L)
    # delta_bias: (Dinner, 1, 1)
    

    if delta_bias is not None:
        dt += delta_bias
    if delta_softplus:
        dt = F.softplus(dt)

    deltaA = torch.exp(dt * A) # (B, Ngroup, Dstate, L)
    deltaB = B * dt.unsqueeze(1) # (B, Ngroup, Dinner, Dstate, L)
    BX = x.unsqueeze(1) * deltaB # (B, Ngroup, Dinner, Dstate, L)
    hs = pscan(deltaA, BX) # (Ngroup, Dinner, Dstate, L)
    print("hs", hs.size())
    C = C.squeeze(0).transpose(-1, -2) # (Ngroup, Dstate, L, 1)
    hs = hs.transpose(-1, -2) # (Ngroup, Dinner, L, Dstate)
    y = torch.einsum("gild,gdlh->gilh", hs, C) # (Ngroup, Dinner, L, 1)
    print("y", y.size())
    y = y.squeeze(-1).unsqueeze(0)
    Dx = (D * x).unsqueeze(1) # (B, 1, Dinner, 1, L)
    y += Dx
    
    return y, None
    

class ANEMamba(_PyMamba):
    """
    CoreML-adapted wrapper for hybrid_mamba_layer.Mamba
    """

    def __init__(self, pytorch_layer: _PyMamba):
        nn.Module.__init__(self)
        # Store the original Mamba layer
        self.layer = pytorch_layer
        self.A = -torch.exp(self.layer.A_log.float()).unsqueeze(-1)
        # self.D = self.layer.D.unsqueeze(-1).unsqueeze(-1)

    def repeat_kv(self, x, flat_at_end=True):
        # Essentially a repeat interleave,
        # tranposes tend to be expensive on ANE,
        # eventually should try to make a custom torch op
        # that uses MIL's concatenate interleave
        # https://apple.github.io/coremltools/source/coremltools.converters.mil.mil.ops.defs.html#coremltools.converters.mil.mil.ops.defs.iOS15.tensor_operation.concat
        x = rearrange(
            x,
            "b (n_group dstate) () l -> b n_group () l dstate",
            dstate=self.layer.d_state,
        )
        x = repeat_kv(x, self.layer.repeat_group)
        if flat_at_end:
            x = rearrange(
                x,
                "b n_group l dstate -> b (n_group dstate) () l",
            )
        else:
            x = rearrange(
                x,
                "b n_group l dstate -> b n_group () dstate l",
            )
        return x

    def forward(
        self,
        hidden_states: torch.Tensor,
        conv_state: torch.Tensor,
    ):
        """
        - hidden_states: (B, Dm, 1, L)
        - conv_state: (num_blocks, D, 1, d_conv - 1) if repeat_kv_before_conv else (num_blocks, Dxb, 1, d_conv - 1)

        I am going to use channels first since I'll work with convolutions
        Dm: model dimension
        D: mamba inner dimension
        Ddt: dt dimension
        Dxb: another dimension
        D_state
        n_group

        x'(t) = W_x @ x(t) #  (B, Dxb, 1, L)  # M1-Mamba applies a linear to the input
        B(t) = W_B @ x(t) # (B, D_state * n_group, 1, L) # Like attention GQA
        C(t) = W_C @ x(t) # (B, D, 1, L)
        dt(t) = W_dt @ x(t) # (B, Ddt, 1, L)

        We have three cases for forward.
        (1) Sequence length 1 for generation
        (2) Short sequence length for speculative decoding, 8 very common
        (3) Prompt processing with long sequence length

        (1) and (3) required a single state, (2) requires a batch of states
        Seems that speculative decoding requires it's tricks, gonna leave
        that for later
        """
        zxbcdt = apply_linear_as_conv2d(self.layer.in_proj, hidden_states)
        z, x, B, C, dt = torch.split(
            zxbcdt,
            [
                self.layer.d_inner,
                self.layer.d_xb,
                self.layer.d_xb,
                self.layer.d_inner,
                self.layer.dt_rank,
            ],
            dim=1,
        )
        dt = apply_linear_as_conv2d(self.layer.dt_proj, dt)

        if self.layer.repeat_kv_before_conv:
            x = self.repeat_kv(x)

        x = torch.concatenate((conv_state, x), dim=-1)
        x = F.conv2d(
            x,
            self.layer.conv1d.weight.unsqueeze(-2),
            self.layer.conv1d.bias,
            groups=self.layer.conv1d.groups,
        )
        x = self.layer.act(x) # (B, D, 1, L)

        if not self.layer.repeat_kv_before_conv:
            x = self.repeat_kv(x)

        B = self.repeat_kv(B, flat_at_end=False)
        C = rearrange(C, "b (n_group dstate) () l -> b n_group dstate () l", dstate=self.layer.d_state)
        y, last_state = selective_scan_fn(
            x,
            dt,
            self.A,
            B,
            C,
            self.layer.D.unsqueeze(-1).unsqueeze(-1),
            z=z,
            delta_bias=self.layer.dt_proj.bias.unsqueeze(-1).unsqueeze(-1),
            delta_softplus=True,
            return_last_state=True,
        )
        print(y.size())
        out = apply_linear_as_conv2d(self.layer.out_proj, y)

        return x