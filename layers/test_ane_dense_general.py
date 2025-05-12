import torch
import torch.nn as nn
import unittest
from ane_models.dia.dia.dia.layers import DenseGeneral
from layers.ane_dense_general import ANEDenseGeneral

class TestANEDenseGeneral(unittest.TestCase):
    def test_k_proj_like_layer(self):
        # Create sample inputs and DenseGeneral layer similar to k_proj
        batch_size = 2
        seq_len = 8
        in_dim = 1024
        out_dim = 2048  # total output dimension (16*128)

        # Create input tensor: [batch_size, in_dim, 1, seq_len] (NCHW format)
        x = torch.randn(batch_size, in_dim, 1, seq_len)

        # Create DenseGeneral layer with flat in_shapes and structured out_features
        dense_layer = DenseGeneral(
            in_shapes=(in_dim,),
            out_features=(16, 128),
            axis=(-1,)
        )

        # Initialize weights with a recognizable pattern for testing
        nn.init.ones_(dense_layer.weight)

        # Original output using standard DenseGeneral
        # First reshape to [batch_size, seq_len, in_dim] for standard DenseGeneral
        x_reshaped = x.permute(0, 3, 2, 1).reshape(batch_size, seq_len, in_dim)
        # Output will have shape [batch_size, seq_len, 16, 128] due to structured out_features
        original_output = dense_layer(x_reshaped)
        # Reshape back to match ANE output: [batch_size, out_dim, 1, seq_len]
        original_output = original_output.permute(0, 2, 3, 1).reshape(batch_size, out_dim, 1, seq_len)

        # Create ANEDenseGeneral wrapper
        ane_layer = ANEDenseGeneral(dense_layer)

        # ANE output should match the original output
        ane_output = ane_layer(x)

        # Check shapes and values
        self.assertEqual(original_output.shape, ane_output.shape)
        self.assertTrue(torch.allclose(original_output, ane_output, atol=1e-5))

        # Test with different batch sizes
        x2 = torch.randn(1, in_dim, 1, 4)
        x2_reshaped = x2.permute(0, 3, 2, 1).reshape(1, 4, in_dim)
        # Apply DenseGeneral and reshape with proper structured dimensions
        original_output2 = dense_layer(x2_reshaped).permute(0, 2, 3, 1).reshape(1, out_dim, 1, 4)
        ane_output2 = ane_layer(x2)
        self.assertEqual(original_output2.shape, ane_output2.shape)
        self.assertTrue(torch.allclose(original_output2, ane_output2, atol=1e-5))

    def test_o_proj_like_layer(self):
        # Create sample inputs and DenseGeneral layer similar to o_proj
        batch_size = 2
        seq_len = 8
        in_dim = 2048  # total input dimension (16*128)
        out_dim = 2048

        # Create input tensor: [batch_size, in_dim, 1, seq_len] (NCHW format)
        x = torch.randn(batch_size, in_dim, 1, seq_len)

        # Create DenseGeneral layer with structured in_shapes and flat out_features
        dense_layer = DenseGeneral(
            in_shapes=(16, 128),
            out_features=(out_dim,),
            axis=(-2, -1)  # Contract both dimensions of the structured input
        )

        # Initialize weights with a recognizable pattern for testing
        nn.init.ones_(dense_layer.weight)

        # Original output using standard DenseGeneral
        # For structured in_shapes, reshape maintaining batch and sequence dimensions
        x_reshaped = x.permute(0, 3, 2, 1).reshape(batch_size, seq_len, 16, 128)
        # Output will have shape [batch_size, seq_len, out_dim]
        original_output = dense_layer(x_reshaped)
        # Reshape back to match ANE output: [batch_size, out_dim, 1, seq_len]
        original_output = original_output.permute(0, 2, 1).unsqueeze(2)

        # Create ANEDenseGeneral wrapper
        ane_layer = ANEDenseGeneral(dense_layer)

        # ANE output should match the original output
        ane_output = ane_layer(x)

        # Check shapes and values
        self.assertEqual(original_output.shape, ane_output.shape)
        self.assertTrue(torch.allclose(original_output, ane_output, atol=1e-5))

        # Test with different batch sizes
        x2 = torch.randn(1, in_dim, 1, 4)
        # Reshape with proper structured dimensions (16, 128)
        x2_reshaped = x2.permute(0, 3, 2, 1).reshape(1, 4, 16, 128)
        # Apply DenseGeneral and reshape appropriately
        original_output2 = dense_layer(x2_reshaped).permute(0, 2, 1).unsqueeze(2)
        ane_output2 = ane_layer(x2)
        self.assertEqual(original_output2.shape, ane_output2.shape)
        self.assertTrue(torch.allclose(original_output2, ane_output2, atol=1e-5))


if __name__ == "__main__":
    unittest.main()