import os
import math
import argparse
import contextlib

import torch

import numpy as np
import coremltools as ct

# just importing patches topk and argmin output_dtype to use uint16
import conversion_ops

from model import ANEGemmaForCausalLM, Wrapper
from config import get_model_config
from original_pytorch_implementation import GemmaForCausalLM, GemmaModel


def load_pytorch_model(weights_dir, device="cpu"):
    # Choose variant and machine type
    VARIANT = "1b"
    MACHINE_TYPE = "cpu"
    OUTPUT_LEN = 200
    METHOD = "it"

    tokenizer_path = os.path.join(weights_dir, "tokenizer.model")
    ckpt_path = os.path.join(weights_dir, f"model.ckpt")

    # Set up model config.
    model_config = get_model_config(VARIANT)
    model_config.dtype = "float16"
    model_config.tokenizer = tokenizer_path

    @contextlib.contextmanager
    def _set_default_tensor_type(dtype: torch.dtype):
        """Sets the default torch dtype to the given dtype."""
        torch.set_default_dtype(dtype)
        yield
        torch.set_default_dtype(torch.float)

    # Instantiate the model and load the weights.
    device = torch.device(MACHINE_TYPE)
    with _set_default_tensor_type(model_config.get_dtype()):
        model = GemmaForCausalLM(model_config)
        model.load_weights(ckpt_path)
        model = model.to(device).eval()

    return model


def convert_model(
    ane_model,
    seqlen,
    save_filename,
    global_kv_cache_length,
    layer_from=0,
    layer_to=None,
    predict=False,
    use_topk=False,
    apply_final_norm=True,
    skip_model_load=False,
    sample=False,
    prediction_head_chunk_size=16_384,
):

    num_layers = ane_model.config.num_hidden_layers
    wmodel = Wrapper(
        ane_model,
        layer_from=layer_from,
        layer_to=layer_to,
        use_topk=use_topk,
        predict=predict,
        apply_final_norm=apply_final_norm,
        global_kv_cache_length=global_kv_cache_length,
        state_implementation="single",  # I do not recomend one state per layer
        prediction_head_chunk_size=prediction_head_chunk_size,
    ).eval().float()

    states = []
    if wmodel.k_cache_local.size(0) > 0:
        states += [
            ct.StateType(
                wrapped_type=ct.TensorType(shape=wmodel.k_cache_local.size()),
                name="k_cache_local",
            ),
            ct.StateType(
                wrapped_type=ct.TensorType(shape=wmodel.v_cache_local.size()),
                name="v_cache_local",
            ),
        ]

    if wmodel.k_cache_global.size(0) > 0:
        states += [
            ct.StateType(
                wrapped_type=ct.TensorType(shape=wmodel.k_cache_global.size()),
                name="k_cache_global",
            ),
            ct.StateType(
                wrapped_type=ct.TensorType(shape=wmodel.v_cache_global.size()),
                name="v_cache_global",
            ),
        ]

        # Set the input_shape to use EnumeratedShapes.
    input_shape = ct.EnumeratedShapes(
        shapes=[(1, 1152, 1, s) for s in seqlen],
        default=[1, 1152, 1, seqlen[0]],
    )
    symbol = input_shape.symbolic_shape[3]

    inputs = [
        ct.TensorType(
            # name="input_hidden_states", shape=(1, 1152, 1, seqlen), dtype=np.float16
            name="input_hidden_states",
            shape=input_shape,
            dtype=np.float16,
        ),
    ]
    # example_inputs = [torch.randn(1, 1152, 1, seqlen, dtype=torch.float16)]
    # trace for seqlen 1 but use and materialize enumerated shapes is faster
    example_inputs = {
        "input_hidden_states": torch.randn(1, 1152, 1, 1, dtype=torch.float16)
    }
    position_shape = ct.EnumeratedShapes(
        shapes=[(s,) for s in seqlen],
        default=[seqlen[0]],
    )
    position_shape.symbolic_shape[0] = symbol
    mask_shape = ct.EnumeratedShapes(
        shapes=[(1, 1, s, global_kv_cache_length) for s in seqlen],
        default=[1, 1, seqlen[0], global_kv_cache_length],
    )
    mask_shape.symbolic_shape[2] = symbol
    local_mask_shape = ct.EnumeratedShapes(
        shapes=[(1, 1, s, wmodel.config.sliding_window_size) for s in seqlen],
        default=[1, 1, seqlen[0], wmodel.config.sliding_window_size],
    )
    local_mask_shape.symbolic_shape[2] = symbol
    if len(states) > 1:

        inputs += [
            ct.TensorType(name="global_write_indices", shape=(1,), dtype=np.int32),
            ct.TensorType(name="local_write_indices", shape=(1,), dtype=np.int32),
            ct.TensorType(name="position", shape=position_shape, dtype=np.int32),
            # ct.TensorType(name="position", shape=(seqlen,), dtype=np.int32),
            ct.TensorType(
                name="mask",
                # shape=(1, 1, seqlen, global_kv_cache_length),
                shape=mask_shape,
                dtype=np.float16,
            ),
            ct.TensorType(
                name="local_mask",
                # shape=(1, 1, seqlen, wmodel.config.sliding_window_size),
                shape=local_mask_shape,
                dtype=np.float16,
            ),
        ]
        example_inputs.update(
            {
                "global_write_indices": torch.tensor(
                    [1], dtype=torch.int32
                ),  # global_write_indices
                "local_write_indices": torch.tensor(
                    [1], dtype=torch.int32
                ),  # local_write_indices
                "position": torch.tensor([1], dtype=torch.int32),  # position_ids
                "mask": torch.zeros(
                    (1, 1, 1, global_kv_cache_length), dtype=torch.float16
                ),  # global mask
                "local_mask": torch.zeros(
                    (1, 1, 1, wmodel.config.sliding_window_size),
                    dtype=torch.float16,
                ),  # local mask
            }
        )
        # example_inputs += [
        #     torch.tensor([1], dtype=torch.int32),  # global_write_indices
        #     torch.tensor([1], dtype=torch.int32),  # local_write_indices
        #     torch.tensor([seqlen], dtype=torch.int32),  # position_ids
        #     torch.zeros(
        #         (1, 1, seqlen, global_kv_cache_length), dtype=torch.float16
        #     ),  # global mask
        #     torch.zeros(
        #         (1, 1, seqlen, wmodel.config.sliding_window_size), dtype=torch.float16
        #     ),  # local mask
        # ]

    if predict:
        outputs = [
            ct.TensorType(name="argmax"),
            ct.TensorType(name="max_value"),
            ct.TensorType(name="logits"),
            ct.TensorType(name="lse"),
        ]
        if sample:
            example_inputs.update(
                {
                    "min_p": torch.tensor([1], dtype=torch.float16),  # min_p
                    "min_p_rng": torch.tensor(
                        [1], dtype=torch.float32
                    ),  # min_p_rng for sampling
                }
            )
            # example_inputs += [
            #     torch.tensor([1], dtype=torch.float16),  # min_p
            #     torch.tensor([1], dtype=torch.float32),  # min_p_rng for sampling
            # ]
            min_p_shape = ct.EnumeratedShapes(
                shapes=[(s,) for s in seqlen],
                default=[seqlen[0]],
            )
            min_p_shape.symbolic_shape[0] = symbol
            inputs += [
                ct.TensorType(name="min_p", shape=min_p_shape, dtype=np.float16),
                ct.TensorType(name="min_p_rng", shape=min_p_shape, dtype=np.float32),
            ]
            outputs += [
                ct.TensorType(name="min_p_sample_index"),
                ct.TensorType(name="min_p_sample_value"),
                ct.TensorType(name="min_p_sum"),
            ]
    else:
        outputs = [ct.TensorType(name="output_hidden_states")]

    with torch.no_grad():
        # traced_model = torch.jit.trace(wmodel, example_inputs)
        traced_model = torch.jit.trace(wmodel, example_kwarg_inputs=example_inputs)
    # print(traced_model.graph)
    # print(traced_model.inlined_graph)
    # print(traced_model.code)

    num_logit_chunks = int(
        math.ceil(model.config.vocab_size / wmodel.prediction_head_chunk_size)
    )

    pipeline = ct.PassPipeline.DEFAULT
    pipeline.remove_passes({"common::add_int16_cast"})
    # pipeline.remove_passes({"common::add_fp16_cast"})
    # pipeline.insert_pass(0, "common::materialize_symbolic_shape_program")
    materialize_options = {"function_name_to_materialization_map": {}}
    for sq in seqlen:
        options = {
            "input_hidden_states": (1, 1152, 1, sq),
        }
        if len(states) > 1:
            options["position"] = (sq,)
            options["mask"] = (1, 1, sq, global_kv_cache_length)
            options["local_mask"] = (1, 1, sq, wmodel.config.sliding_window_size)
        if sample:
            options["min_p"] = (sq,)
            options["min_p_rng"] = (sq,)
        materialize_options["function_name_to_materialization_map"][
            f"{args.function_prefix}_{sq}"
        ] = options
    # pipeline.set_options("common::materialize_symbolic_shape_program", materialize_options)

    mlmodel: ct.models.MLModel = ct.convert(
        traced_model,
        convert_to="milinternal",
        inputs=inputs,
        outputs=outputs,
        states=states,
        minimum_deployment_target=ct.target.iOS18,
        compute_units=ct.ComputeUnit.CPU_AND_NE,
        # compute_precision=ct.precision.FLOAT16,
        skip_model_load=True,
        pass_pipeline=pipeline,
        compute_precision=ct.precision.FLOAT16,
    )
    print(mlmodel)
    mlmodel: ct.models.MLModel = ct.convert(
        mlmodel,
        # convert_to="milinternal",
        inputs=inputs,
        outputs=outputs,
        # states=states,
        minimum_deployment_target=ct.target.iOS18,
        compute_units=ct.ComputeUnit.CPU_AND_NE,
        # compute_precision=ct.precision.FLOAT16,
        skip_model_load=True,
        pass_pipeline=pipeline,
        compute_precision=ct.precision.FLOAT16,
    )


    print("Materializing and saving")
    # for sq, vals in materialize_options["function_name_to_materialization_map"].items():
    #     ct.utils.materialize_dynamic_shape_mlmodel(
    #         mlmodel,
    #         {"main": vals},
    #         f"{sq}_argmax_CHUNK_{prediction_head_chunk_size}_topk_{use_topk}",
    #     )
    ct.utils.materialize_dynamic_shape_mlmodel(
        mlmodel,
        materialize_options["function_name_to_materialization_map"],
        save_filename,
    )

    # mlmodel.save(save_filename)
    # ct.models.utils.save_multifunction(mlmodel, save_filename)
    # if not skip_model_load:
    #     print("Loading")
    #     mlmodel = ct.models.MLModel(save_filename.rstrip(".mlpackage") + ".mlpackage")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights_dir", type=str, required=True)
    parser.add_argument("--save_filename", type=str, required=True)
    parser.add_argument("--input_seqlen", nargs="+", type=int, required=True)
    parser.add_argument("--global_kv_cache_length", type=int, default=512)
    parser.add_argument("--layer_from", type=int, default=0)
    parser.add_argument("--layer_to", type=int)
    parser.add_argument("--predict", action="store_true", default=False)
    parser.add_argument("--apply_final_norm", action="store_true", default=False)
    parser.add_argument("--use_topk", action="store_true", default=False)
    parser.add_argument("--skip_model_load", action="store_true", default=False)
    parser.add_argument("--sample", action="store_true", default=False)
    parser.add_argument("--prediction_head_chunk_size", type=int, default=16_384)
    parser.add_argument("--function_prefix", type=str, required=True)

    parser.add_argument("--from-multifunction", type=str)
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    model = load_pytorch_model(args.weights_dir).half()
    ane_model = ANEGemmaForCausalLM(model, state_implementation="single")

    # if args.sample:
    #     assert args.input_seqlen[0] == 1 and len(args.input_seqlen) == 1, "Sample mode only supports seqlen=1"

    convert_model(
        ane_model,
        args.input_seqlen,
        args.save_filename,
        args.global_kv_cache_length,
        args.layer_from,
        args.layer_to,
        args.predict,
        args.use_topk,
        args.apply_final_norm,
        args.skip_model_load,
        args.sample,
        args.prediction_head_chunk_size,
    )
