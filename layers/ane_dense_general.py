from typing import Optional, Tuple
import torch
import torch.nn as nn
from torch import Tensor
from ane_models.dia.dia.dia.layers import DenseGeneral

class ANEDenseGeneral(nn.Module):
    """
    Wrapper for DenseGeneral that reshapes weights for better compatibility with Apple Neural Engine.

    This wrapper transforms the DenseGeneral layer to use flattened weight matrices and
    convolution operations which are more ANE-friendly.

    This implementation assumes:
    - Inputs and outputs dimensions are flattened
    - Inputs are in NCHW format: [batch_size, hidden_dimension, 1, sequence_length]
    - Weights are reshaped at inference time (not during initialization)

    Attributes:
        layer (DenseGeneral): The original DenseGeneral layer being wrapped.
        in_features_flat (int): Flattened input features dimension.
        out_features_flat (int): Flattened output features dimension.
    """

    def __init__(self, layer: DenseGeneral):
        super().__init__()
        self.layer = layer

        # Store original shapes
        self.original_in_shapes = layer.in_shapes
        self.original_out_features = layer.out_features
        self.original_axis = layer.axis

        # Compute the flat input and output dimensions
        self.in_features_flat = 1
        for dim in self.original_in_shapes:
            self.in_features_flat *= dim

        self.out_features_flat = 1
        for dim in self.original_out_features:
            self.out_features_flat *= dim

    def forward(self, inputs: Tensor) -> Tensor:
        """
        Forward pass for the ANEDenseGeneral layer.

        Args:
            inputs (Tensor): The input tensor in NCHW format
                            (shape: [batch_size, hidden_dimension, 1, sequence_length])

        Returns:
            Tensor: The output tensor with shape [batch_size, out_features_flat, 1, sequence_length]
        """
        # Save original dtype and shape
        original_dtype = inputs.dtype
        batch_size, _, _, seq_length = inputs.shape

        # Reshape weights at inference time
        # From original shape to [out_features_flat, in_features_flat, 1, 1]
        conv_weight = self.layer.weight.reshape(
            self.in_features_flat, self.out_features_flat
        ).transpose(0, 1).reshape(
            self.out_features_flat, self.in_features_flat, 1, 1
        )

        # Apply convolution - this preserves the NCHW format
        # Input: [batch_size, hidden_dimension, 1, sequence_length]
        # Weight: [out_features_flat, in_features_flat, 1, 1]
        # Output: [batch_size, out_features_flat, 1, sequence_length]
        outputs = torch.nn.functional.conv2d(inputs, conv_weight)

        return outputs.to(original_dtype)


def ane_dense_general(x: Tensor, dense_general: DenseGeneral) -> Tensor:
    """
    Utility function to apply an ANE-friendly dense general operation.

    Args:
        x (Tensor): Input tensor in NCHW format.
        dense_general (DenseGeneral): The DenseGeneral layer to wrap.

    Returns:
        Tensor: The result of applying the wrapped dense general operation.
    """
    wrapper = ANEDenseGeneral(dense_general)
    return wrapper(x)
