from typing import Tuple
from coremltools.converters.mil.frontend.torch.torch_op_registry import (
    _TORCH_OPS_REGISTRY,
    register_torch_op,
)
from coremltools.converters.mil.frontend.torch.utils import TorchFrontend
from coremltools.converters.mil.frontend.torch.ops import (
    _get_inputs,
    _get_kwinputs,
    is_current_opset_version_compatible_with,
    _utils,
    mb,
    target,
)
from coremltools.converters.mil.mil.var import ListVar, Var

del _TORCH_OPS_REGISTRY["topk"]


# Modified topk to use uint16 as return_dtype and run on ANE
@register_torch_op
def topk(context, node):
    def _parse_positional_args(context, node) -> Tuple[Var]:
        inputs = _get_inputs(context, node, expected=(2, 3, 4, 5, 6, 7))
        nargs = len(inputs)

        x = inputs[0]
        k = inputs[1]

        dim = inputs[2] if nargs > 2 else -1
        largest = inputs[3] if nargs > 3 else True
        sorted = inputs[4] if nargs > 4 else True

        # When node.kind == topk.values, there can be 2 more args
        # `Tensor(a!) values` and `Tensor(b!) indices`, which are for in-place mutation,
        # so we ignore them since Core ML is functional
        return x, k, dim, largest, sorted

    def _parse_keyword_args(context, node, dim, largest, sorted) -> Tuple[Var]:
        dim = _get_kwinputs(context, node, "dim", default=[dim])[0]
        largest = _get_kwinputs(context, node, "largest", default=[largest])[0]
        sorted = _get_kwinputs(context, node, "sorted", default=[sorted])[0]
        return dim, largest, sorted

    def _translate_torch_args(dim, largest, sorted) -> Tuple[Var]:
        if isinstance(dim, Var):
            dim = dim.val

        if isinstance(largest, Var):
            largest = largest.val

        if isinstance(sorted, Var):
            sorted = sorted.val
        if not sorted and not is_current_opset_version_compatible_with(target.iOS16):
            raise Exception(
                "For opset <= iOS16, only sorted=True supported for the topk"
            )

        return dim, not largest, sorted

    x, k, dim, largest, sorted = _parse_positional_args(context, node)
    dim, largest, sorted = _parse_keyword_args(context, node, dim, largest, sorted)
    axis, ascending, sort = _translate_torch_args(dim, largest, sorted)

    kwargs = {"name": node.name, "x": x, "k": k, "axis": axis, "ascending": ascending}
    if is_current_opset_version_compatible_with(target.iOS16):
        kwargs["sort"] = sort
    # if axis is not None:
    #     kwargs["axis"] = axis
    # if ascending is not None and ascending:
    #     kwargs["ascending"] = ascending
    # if sort is not None and not sort:
    #     kwargs["sort"] = sort
    kwargs["output_indices_dtype"] = "uint16"
    for d in x.shape:
        if isinstance(d, int) and d > 16_384:
            kwargs["output_indices_dtype"] = "int32"
            break

    if kwargs["k"].val is None:
        res = _utils.dynamic_topk(
            x=kwargs["x"],
            k=kwargs["k"],
            axis=kwargs["axis"],
            ascending=kwargs["ascending"],
        )
    else:
        res = mb.topk(**kwargs)  # SET OUTPUT DTYPE HERE FOR ANE
    if context.frontend == TorchFrontend.TORCHSCRIPT:
        values_name = node.outputs[0]
        indices_name = node.outputs[1]
        context.add(res[0], torch_name=values_name)
        context.add(res[1], torch_name=indices_name)
    else:
        context.add(res, torch_name=node.name)


del _TORCH_OPS_REGISTRY["max"]
del _TORCH_OPS_REGISTRY["min"]
del _TORCH_OPS_REGISTRY["max.dim"]
del _TORCH_OPS_REGISTRY["min.dim"]


def _add_max_min(context, node, reduce_op, reduce_arg_op, alias_op):
    def _get_output_dtype(x):
        for d in x.shape:
            if isinstance(d, int) and d > 16_384:
                return "int32"
        return "uint16"

    if context.frontend == TorchFrontend.TORCHSCRIPT:
        # mimic functionality from https://pytorch.org/docs/stable/generated/torch.min.html
        # mimic functionality from https://pytorch.org/docs/stable/generated/torch.max.html

        inputs = _get_inputs(context, node, expected=[1, 2, 3])
        if len(inputs) == 1:
            value = reduce_op(x=inputs[0], axes=None, name=node.name)
            context.add(value)
        elif len(inputs) == 2:
            value = alias_op(x=inputs[0], y=inputs[1], name=node.name)
            context.add(value)
        elif len(inputs) == 3:
            _input = inputs[0]
            dim = inputs[1].val
            keepdim = inputs[2].val

            values = reduce_op(x=_input, axes=[dim], keep_dims=keepdim)
            indices = reduce_arg_op(
                x=_input,
                axis=dim,
                keep_dims=keepdim,
                output_dtype=_get_output_dtype(_input),
            )
            assert len(node.outputs) == 2
            values_name = node.outputs[0]
            indices_name = node.outputs[1]
            context.add(values, torch_name=values_name)
            context.add(indices, torch_name=indices_name)

    else:
        # clearly distinguish each variant of max

        def _parse_positional_args(context, node) -> Tuple[Var]:
            inputs = _get_inputs(context, node, min_expected=1)
            nargs = len(inputs)

            x = inputs[0]
            dim = None if nargs < 2 else inputs[1].val
            keepdim = False if nargs < 3 else inputs[2].val

            return x, dim, keepdim

        x, dim, keepdim = _parse_positional_args(context, node)

        func_suffix = node.kind.split(".")
        if len(func_suffix) == 1:
            value = reduce_op(x=x, axes=None, name=node.name)
            context.add(value)
        elif func_suffix[-1] == "dim":
            values = reduce_op(x=x, axes=[dim], keep_dims=keepdim)
            indices = reduce_arg_op(
                x=x, axis=dim, keep_dims=keepdim, output_dtype=_get_output_dtype(x)
            )
            context.add((values, indices), torch_name=node.name)


@register_torch_op(torch_alias=["max.dim"])
def max(context, node):
    _add_max_min(context, node, mb.reduce_max, mb.reduce_argmax, mb.maximum)


@register_torch_op(torch_alias=["min.dim"])
def min(context, node):
    _add_max_min(context, node, mb.reduce_min, mb.reduce_argmin, mb.minimum)
