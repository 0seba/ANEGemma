This document contains some recommendations to apply to Pytorch models in order to make them compatible to run on the Apple Neural Engine.

1. ANE works better with channels first and 2D convolutions layout, modify linear layers to use this configuration.
```python
import torch.nn as nn

use_bias = True
linear = nn.Linear(in_features, out_features, bias=use_bias)

class ANELinear(nn.Module):
    def __init__(self, linear_module: nn.Linear):
        super(ANELinear, self).__init__()
        self.linear = linear_module

    def forward(self, x):
        w = self.linear.weight.unsqueeze(-1).unsqueeze(-1)
        if self.linear.bias is not None:
            return torch.nn.functional.conv2d(x, w, self.linear.bias)
        return torch.nn.functional.conv2d(x, w)
```

2. There is no evidence that ANE supports any kind of `fp32` precision, this is problematic for models with very big activations, and specially with operations such as normalization from the squared sum. A fix to this issue is to use max-scaled normalization.

```python
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
        )  # Possibly try to divide by factor to use more of the float16 range
        xscaled = x / maxval
        # TODO find and way to use reduce_sum_square MIL op
        sq_sum = xscaled.square().sum(dim=dim, keepdim=True)
        rsqrt = torch.rsqrt(sq_sum)
        dimroot = x.size(dim) ** (1 / 2)
        return rsqrt * dimroot * xscaled

    def forward(self, x: torch.Tensor, num_expands=2, dim=-3) -> torch.Tensor:
        output = self._norm(x, dim=dim)
        weight = self.weight
        weight = weight.view(*weight.size() + (1,) * num_expands)
        if self.add_unit_offset:
            output = output * (1 + weight)
        else:
            output = output * weight
        return output  # .type_as(x)
```

3. It is very possible that this may not be enough in some cases, I haven't tried it yet but I think one solution could be to use rotations as described in https://stephenpanaro.com/blog/modernbert-on-apple-neural-engine.

4. Recent RoPE implementations calculate the frequencies on the fly, which requires `fp32` support, this means that for ANE we have 2 options:
  1. Precalculate the for the frequencies of every possition beforehand and store them inside of the model in `fp16`. In case this option is taken, perform the indexing of the frequencies corresponding position once at the beginning of the model and propagate it to all layers in the forward call instead of one time for each attention layer, gathering operations are usually very slow in ANE, specially for early architectures like M1's.
  2. Add the frequencies as an additional input to the model, this also have to be in `fp16`.
Since RoPE transformation inside of the ANE is also performed in `fp16` there is precision loss compared to GPU/CPUs that perform the transformation in `fp32` for both these methods.

** Other recommendations **
- Try to use as few transposes as possible
