DATA_TYPE_INT_TO_NAME = {
    # Common values, adjust as needed based on your specific proto enum definition
    # Check your Model_pb2.py or .proto file for the correct integer values
    # Example values derived from typical ML frameworks / Core ML:
    # Core ML specific values might be different - check documentation or generated code
    65552: "FLOAT16",  # Common representation, Core ML might use a different int
    65568: "FLOAT32",  # Common representation
    131104: "INT32",  # Common representation
    # Add other types defined in your ArrayDataType enum here
    # 0: "INVALID",
    # 2: "FLOAT64",
    # 4: "INT8",
    # 5: "INT16",
    # 6: "INT64",
    # 7: "UINT8",
    # ... etc.
}


def _get_data_type_name_direct_map(dtype_enum_value):
    """Helper to get the string name of the data type using a direct map."""
    return DATA_TYPE_INT_TO_NAME.get(
        dtype_enum_value, f"UNKNOWN_DTYPE({dtype_enum_value})"
    )


# --- End of Direct Mapping Approach ---


def _format_shape(shape_list):
    """Formats a list/repeated field of dimensions into a tuple string."""
    if not shape_list:
        return "scalar ()"
    return str(tuple(shape_list))


def _format_tensor_info(tensor_type):
    """Formats the type information for a tensor (MultiArray or State Array)."""
    type_str = "Unknown Type"
    shape_str = ""
    dtype_str = ""
    enum_shapes_str = ""

    if tensor_type.HasField("multiArrayType"):
        ma_type = tensor_type.multiArrayType
        # Use the direct mapping function here
        dtype_str = _get_data_type_name_direct_map(ma_type.dataType)
        shape_str = _format_shape(ma_type.shape)

        if ma_type.HasField("enumeratedShapes") and ma_type.enumeratedShapes.shapes:
            enum_shapes = [
                _format_shape(s.shape) for s in ma_type.enumeratedShapes.shapes
            ]
            enum_shapes_str = f" {{Enum Shapes: [{', '.join(enum_shapes)}]}}"

        type_str = f"{dtype_str} {shape_str}{enum_shapes_str}"

    elif tensor_type.HasField("stateType"):
        st_type = tensor_type.stateType
        if st_type.HasField("arrayType"):
            arr_type = st_type.arrayType
            # Use the direct mapping function here
            dtype_str = _get_data_type_name_direct_map(arr_type.dataType)
            shape_str = _format_shape(arr_type.shape)
            type_str = f"State<{dtype_str} {shape_str}>"
        else:
            type_str = "State<Unknown>"

    return type_str


# def _format_tensor_info(tensor_type):
#     """Formats the type information for a tensor (MultiArray or State Array)."""
#     type_str = "Unknown Type"
#     shape_str = ""
#     dtype_str = ""
#     enum_shapes_str = ""
#     dtype_value = -999 # Default invalid value

#     if tensor_type.HasField("multiArrayType"):
#         ma_type = tensor_type.multiArrayType
#         dtype_value = ma_type.dataType # Get the integer value
#         # --- TEMPORARY DEBUG PRINT ---
#         print(f"DEBUG: multiArrayType - Found dataType integer value: {dtype_value}")
#         # -----------------------------
#         dtype_str = _get_data_type_name_direct_map(dtype_value)
#         shape_str = _format_shape(ma_type.shape)
#         # ... (rest of multiArrayType handling) ...
#         type_str = f"{dtype_str} {shape_str}{enum_shapes_str}"

#     elif tensor_type.HasField("stateType"):
#         st_type = tensor_type.stateType
#         if st_type.HasField("arrayType"):
#             arr_type = st_type.arrayType
#             dtype_value = arr_type.dataType # Get the integer value
#             # --- TEMPORARY DEBUG PRINT ---
#             print(f"DEBUG: stateType.arrayType - Found dataType integer value: {dtype_value}")
#             # -----------------------------
#             dtype_str = _get_data_type_name_direct_map(dtype_value)
#             shape_str = _format_shape(arr_type.shape)
#             type_str = f"State<{dtype_str} {shape_str}>"
#         else:
#              type_str = "State<Unknown>"
#              print("DEBUG: stateType found, but no arrayType field.") # Added debug

#     # --- Optional: Print if no known type was found ---
#     # if dtype_value == -999 and type_str == "Unknown Type":
#     #      print(f"DEBUG: No known tensor type field found in: {tensor_type}")
#     # ---------------------------------------------------

#     return type_str


def print_model_description(desc):
    """
    Processes and prints a ModelDescription protobuf object concisely.
    Uses a direct map for data type names.

    Args:
        desc: An instance of the ModelDescription protobuf message.
    """
    if not hasattr(desc, "functions"):
        print("Invalid description object: Missing 'functions' field.")
        return

    if len(desc.functions) == 0:
        functions = [desc]
        print("Single Function Model")
    else:
        print(
            f"Model Description (Default Function: {desc.defaultFunctionName if hasattr(desc, 'defaultFunctionName') else 'N/A'})"
        )
        functions = desc.functions
    print("=" * 80)

    for i, func in enumerate(functions):
        print(f"\n--- Function {i+1}: {func.name if hasattr(func, 'name') else 'main'} ---")

        # --- Inputs ---
        if func.input:
            print("  Inputs:")
            max_name_len = max(len(inp.name) for inp in func.input) if func.input else 0
            for inp in func.input:
                info_str = _format_tensor_info(inp.type)
                print(f"    - {inp.name:<{max_name_len}} : {info_str}")
        else:
            print("  Inputs: None")

        # --- Outputs ---
        if func.output:
            print("  Outputs:")
            max_name_len = (
                max(len(out.name) for out in func.output) if func.output else 0
            )
            for out in func.output:
                info_str = _format_tensor_info(out.type)
                if (
                    out.type.HasField("multiArrayType")
                    and not out.type.multiArrayType.shape
                ):
                    if "Enum Shapes" not in info_str:
                        info_str += " (Shape Undefined/Dynamic?)"
                print(f"    - {out.name:<{max_name_len}} : {info_str}")
        else:
            print("  Outputs: None")

        # --- States ---
        if func.state:
            print("  States:")
            max_name_len = max(len(st.name) for st in func.state) if func.state else 0
            for st in func.state:
                info_str = _format_tensor_info(st.type)
                print(f"    - {st.name:<{max_name_len}} : {info_str}")
        else:
            print("  States: None")

    print("\n" + "=" * 80)
