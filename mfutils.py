import os
import pprint
import argparse
import cmd
import coremltools as ct
from coremltools.models.utils import (
    load_spec,
    MultiFunctionDescriptor,
    save_multifunction,
)
from coremltools.converters.mil.converter import mil_convert as _mil_convert
from coremltools.converters.mil.frontend.milproto import load as _milproto_to_pymil
from utils import print_model_description


def info(args):
    spec = load_spec(args.model_path)
    print_model_description(spec.description)


def merge(args):
    paths = args.model_paths
    mf_description = MultiFunctionDescriptor()

    def _add_function(_path, source_function_name):
        filename = os.path.basename(_path).split(".")[0]
        target_function_name = input(
            f"Enter new name for function '{source_function_name}' of model '{_path}', empty to mantain, 'o' to omit or 'f' to name it as the model file ({filename}):\n"
        )
        target_function_name = target_function_name.strip()
        if target_function_name.lower() == "o":
            return
        elif target_function_name.lower() == "f":
            target_function_name = filename
        else:
            target_function_name = (
                target_function_name
                if len(target_function_name) > 0
                else source_function_name
            )
        mf_description.add_function(_path, source_function_name, target_function_name)

    for path in paths:
        spec = load_spec(path)
        if len(spec.description.functions) == 0:
            _add_function(path, "main")
        else:
            for f in spec.description.functions:
                _add_function(path, f.name)

    print("Function schema:")
    pprint.pprint(mf_description._functions(), indent=4)

    # Ask for default function name and check if it exists, with while retry
    default_function_name = input(
        "Enter default function name (required) to save:\n"
    ).strip()
    while default_function_name not in mf_description._functions():
        print("Function name not found in the model. Please try again.")
        default_function_name = input(
            "Enter default function name (required) to save:\n"
        ).strip()
    mf_description.default_function_name = default_function_name
    print(f"Saving to {args.save_path}... this may take a while")
    save_multifunction(mf_description, args.save_path)


def materialize_dynamic_shape(args):
    desc = load_spec(args.model_path).description
    functions = desc.functions if len(desc.functions) > 0 else [desc]
    materialize_options = {}
    for func in functions:
        func_name = func.name if hasattr(func, "name") else "main"
        materialize_options[func_name] = {}
        for seqlen in args.lengths:
            func_mapping = {}
            materialize_options[func_name][
                func_name + f"_length_{seqlen}"
            ] = func_mapping
            for inp in func.input:
                inp_type = inp.type
                if inp_type.HasField("multiArrayType"):
                    if (
                        inp_type.multiArrayType.HasField("enumeratedShapes")
                        and inp_type.multiArrayType.enumeratedShapes.shapes
                    ):
                        # TODO TEMP hardcoded function inp_type names
                        shape = [*inp_type.multiArrayType.shape]
                        if inp.name == "input_hidden_states":
                            shape[3] = seqlen
                            func_mapping[inp.name] = tuple(shape)
                        elif inp.name in ["position", "min_p", "min_p_rng"]:
                            shape[0] = seqlen
                            func_mapping[inp.name] = tuple(shape)
                        elif inp.name in ["mask", "local_mask"]:
                            shape[2] = seqlen
                            func_mapping[inp.name] = tuple(shape)
                        else:
                            func_mapping[inp.name] = tuple(shape)
                    else:
                        func_mapping[inp.name] = tuple(inp_type.multiArrayType.shape)
    print("Materializing the following functions")
    pprint.pprint(materialize_options, indent=4)
    input("Press enter to continue")

    # Copied an adapted from CoreMLTools materialize_dynamic_shape_mlmodel method
    dynamic_shape_mlmodel = ct.models.MLModel(args.model_path, skip_model_load=True)
    if dynamic_shape_mlmodel._mil_program is not None:
        dynamic_shape_prog = dynamic_shape_mlmodel._mil_program
    else:
        dynamic_shape_prog = _milproto_to_pymil.load(
            dynamic_shape_mlmodel._spec,
            dynamic_shape_mlmodel._spec.specificationVersion,
            dynamic_shape_mlmodel.weights_dir,
        )
    print("Materializing, this may take a while...")
    for i, (k, v) in enumerate(materialize_options.items()):
        pass_pipeline = ct.PassPipeline.DEFAULT
        pass_pipeline.insert_pass(0, "common::materialize_symbolic_shape_program")
        pass_pipeline.set_options(
            "common::materialize_symbolic_shape_program",
            {
                "function_name_to_materialization_map": v,
                "source_function_name": k,
            },
        )
        ct.converters.mil.mil.passes.pass_pipeline.PassPipelineManager.apply_pipeline(
            dynamic_shape_prog, pass_pipeline
        )


    print(dynamic_shape_prog)
    dynamic_shape_prog.export_as_multifunction = True
    dynamic_shape_prog.skip_all_passes = True
    materialized_mlmodel = _mil_convert(
        dynamic_shape_prog,
        convert_from="milinternal",
        convert_to="mlprogram",
        specification_version=ct.target.iOS18,
        compute_units=ct.ComputeUnit.CPU_ONLY,
        skip_model_load=True,
    )
    # materialized_mlmodel.save(args.save_path)


def parse_args():
    parser = argparse.ArgumentParser(prog="PROG")

    # Create subparsers object
    subparsers = parser.add_subparsers(dest="command")

    # create the parser for the "info" command
    parser_info = subparsers.add_parser("info", help="Get information of mlpackage")
    parser_info.add_argument("model_path", type=str, help="Path to mlpackage")
    parser_info.set_defaults(func=info)

    parser_merge = subparsers.add_parser("merge", help="Get information of mlpackage")
    parser_merge.add_argument(
        "model_paths", type=str, nargs="+", help="Path to mlpackage"
    )
    parser_merge.add_argument(
        "--save_path",
        type=str,
        help="Path to save merged mlpackage",
        required=True,
    )
    parser_merge.set_defaults(func=merge)

    parser_materialize = subparsers.add_parser("materialize", help="")
    parser_materialize.add_argument(
        "model_path",
        type=str,
        help="Path to mlpackage",
    )
    parser_materialize.add_argument(
        "lengths",
        type=int,
        nargs="+",
    )
    parser_materialize.add_argument(
        "--save_path",
        type=str,
        help="Path to save materialized mlpackage",
        required=True,
    )
    parser_materialize.set_defaults(func=materialize_dynamic_shape)

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    args.func(args)
