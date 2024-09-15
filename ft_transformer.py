# Import necessary libraries
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
from transformer import SimpleTransformer

class NumericalEmbedder(nn.Module):
    """
    This class embeds numerical (continuous) inputs by using learned weights and biases.
    
    Parameters:
    - dim (int): Dimension of the embeddings for each numerical feature.
    - num_numerical_types (int): Number of different continuous features (numerical types).

    Inputs:
    - x (tensor): A tensor of shape (batch_size, num_continuous) containing continuous values.
    
    Output:
    - A tensor of shape (batch_size, num_continuous, dim), representing embedded continuous features.
    """
    def __init__(self, dim, num_numerical_types):
        super().__init__()
        # Define learnable weights and biases for embedding continuous features
        self.weights = nn.Parameter(torch.randn(num_numerical_types, dim))
        self.biases = nn.Parameter(torch.randn(num_numerical_types, dim))

    def forward(self, x):
        # Reshape the input to (batch_size, num_continuous, 1)
        x = rearrange(x, 'b n -> b n 1')
        # Apply embedding using learned weights and biases
        return x * self.weights + self.biases


class FTTransformer(nn.Module):
    """
    This class implements a Transformer model for processing tabular data, consisting of both categorical
    and continuous features.

    Parameters:
    - categories (list): List specifying the number of unique values for each categorical feature.
    - num_continuous (int): Number of continuous (numerical) features.
    - dim (int): Dimension of embeddings.
    - depth (int): Number of Transformer layers (depth).
    - heads (int): Number of attention heads in each layer.
    - dim_head (int, optional): Dimension of each attention head. Default is 16.
    - dim_out (int, optional): Output dimension (for classification/regression). Default is 1.
    - num_special_tokens (int, optional): Number of special tokens (e.g., CLS token). Default is 2.
    - attn_dropout (float, optional): Dropout rate for attention. Default is 0.
    - ff_dropout (float, optional): Dropout rate for feedforward layers. Default is 0.
    
    Inputs:
    - x_categ (tensor): Tensor of categorical inputs, shape (batch_size, num_categories).
    - x_numer (tensor): Tensor of continuous (numerical) inputs, shape (batch_size, num_continuous).
    - return_attn (bool, optional): If True, the model will return attention weights. Default is False.

    Output:
    - logits (tensor): The output prediction of the model, shape (batch_size, dim_out).
    - If return_attn is True, it also returns attention weights.
    """
    def __init__(
        self,
        *,
        categories,
        num_continuous,
        dim,
        depth,
        heads,
        dim_head = 16,
        dim_out = 1,
        num_special_tokens = 2,
        attn_dropout = 0.,
        ff_dropout = 0.
    ):
        super().__init__()
        
        # Sanity checks for input dimensions
        assert all(map(lambda n: n > 0, categories)), 'number of each category must be positive'
        assert len(categories) + num_continuous > 0, 'input shape must not be null'

        # Store the number of categorical features and total number of unique categories
        self.num_categories = len(categories)
        self.num_unique_categories = sum(categories)

        # Total number of tokens in the embedding (unique categories + special tokens like CLS)
        self.num_special_tokens = num_special_tokens
        total_tokens = self.num_unique_categories + num_special_tokens

        # Offset to handle embeddings for categorical features (if present)
        if self.num_unique_categories > 0:
            categories_offset = F.pad(torch.tensor(list(categories)), (1, 0), value = num_special_tokens)
            categories_offset = categories_offset.cumsum(dim = -1)[:-1]
            self.register_buffer('categories_offset', categories_offset)

            # Create an embedding layer for categorical features
            self.categorical_embeds = nn.Embedding(total_tokens, dim)

        # Store the number of continuous features
        self.num_continuous = num_continuous

        # Create a numerical embedder if there are continuous features
        if self.num_continuous > 0:
            self.numerical_embedder = NumericalEmbedder(dim, self.num_continuous)

        # CLS token for Transformer, initialized randomly
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))

        # Transformer for feature processing
        self.transformer = SimpleTransformer(
            dim = dim,
            depth = depth,
            heads = heads,
            dim_head = dim_head,
            attn_dropout = attn_dropout,
            ff_dropout = ff_dropout
        )

        # Layer for output prediction (to logits)
        self.to_logits = nn.Sequential(
            nn.LayerNorm(dim),
            nn.ReLU(),
            nn.Linear(dim, dim_out)
        )

    def forward(self, x_categ, x_numer, return_attn = False):
        """
        Forward method processes the categorical and numerical inputs through the Transformer model.

        Inputs:
        - x_categ (tensor): Categorical input, shape (batch_size, num_categories).
        - x_numer (tensor): Numerical input, shape (batch_size, num_continuous).
        - return_attn (bool, optional): Whether to return attention weights. Default is False.

        Output:
        - logits (tensor): Model prediction, shape (batch_size, dim_out).
        - If return_attn is True, also returns attention weights.
        """
        # Ensure the input matches the expected number of categories
        assert x_categ.shape[-1] == self.num_categories, f'you must pass in {self.num_categories} values for your categories input'

        # List to store processed inputs (categorical and numerical)
        xs = []
        
        # Embed categorical inputs if available
        if self.num_unique_categories > 0:
            x_categ = x_categ + self.categories_offset
            x_categ = self.categorical_embeds(x_categ)
            xs.append(x_categ)

        # Embed continuous inputs if available
        if self.num_continuous > 0:
            x_numer = self.numerical_embedder(x_numer)
            xs.append(x_numer)

        # Concatenate categorical and numerical embeddings
        x = torch.cat(xs, dim = 1)

        # Add CLS token to the sequence
        b = x.shape[0]  # batch size
        cls_tokens = repeat(self.cls_token, '1 1 d -> b 1 d', b = b)
        x = torch.cat((cls_tokens, x), dim = 1)

        # Pass through Transformer
        x, attns = self.transformer(x, return_attn = True)

        # Extract the CLS token output for classification/regression
        x = x[:, 0]

        # Pass the output through the final linear layer
        logits = self.to_logits(x)

        # Return logits and (optionally) attention weights
        if not return_attn:
            return logits
        return logits, attns
    
# Unit Test
import unittest

class TestFTTransformer(unittest.TestCase):
    
    def setUp(self):
        # Define some example parameters
        self.categories = [10, 20, 15]  # 3 categorical features with different numbers of categories
        self.num_continuous = 5         # 5 continuous features
        self.dim = 32                   # Embedding dimension
        self.depth = 4                  # Transformer depth
        self.heads = 4                  # Number of attention heads
        self.batch_size = 8             # Example batch size

        # Create an instance of the FTTransformer
        self.model = FTTransformer(
            categories=self.categories,
            num_continuous=self.num_continuous,
            dim=self.dim,
            depth=self.depth,
            heads=self.heads
        )

    def test_forward_pass(self):
        # Create random inputs (categorical and continuous)
        x_categ = torch.randint(0, 10, (self.batch_size, len(self.categories)))  # Categorical inputs
        x_numer = torch.randn(self.batch_size, self.num_continuous)              # Continuous inputs

        # Perform a forward pass through the model
        output = self.model(x_categ, x_numer)

        # Check if the output has the correct shape (batch_size, dim_out)
        self.assertEqual(output.shape, (self.batch_size, 1), "Output shape is incorrect.")

if __name__ == "__main__":
    unittest.main()
