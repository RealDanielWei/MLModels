import torch
from torch import nn, einsum
import torch.nn.functional as F
from einops import rearrange, repeat

class GEGLU(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        # Split the input tensor into two equal parts along the last dimension
        x, gates = x.chunk(2, dim=-1)
        # Apply the GELU activation function to 'gates' and multiply element-wise with 'x'
        return x * F.gelu(gates)

def FeedForward(dim, mult=4, dropout=0.):
    """
    Constructs a feed-forward neural network module with GEGLU activation.

    Parameters:
    - dim (int): Input and output dimensionality of the network.
    - mult (int): Multiplicative factor for the hidden layer size. Default is 4.
    - dropout (float): Dropout rate. Default is 0.

    Returns:
    - nn.Sequential: A sequential container of neural network layers.
    """
    return nn.Sequential(
        nn.LayerNorm(dim),                       # Layer normalization
        nn.Linear(dim, dim * mult * 2),          # Linear projection to higher-dimensional space
        GEGLU(),                                 # GEGLU activation function
        nn.Dropout(dropout),                     # Dropout layer for regularization
        nn.Linear(dim * mult, dim)               # Linear projection back to original dimension
    )

class MultiHeadSelfAttention(nn.Module):
    """
    The multi-head self-attention module.

    Parameters:
    - dim (int): Input embedding dimension.
    - heads (int, optional): Number of attention heads. Default is 8.
    - dim_head (int, optional): Dimension of each attention head. Default is 64.
    - dropout (float, optional): Dropout rate for regularization. Default is 0.0.
    """
    def __init__(
        self,
        dim,
        heads=8,
        dim_head=64,
        dropout=0.
    ):
        super().__init__()
        inner_dim = dim_head * heads  # Total dimension after projecting into multiple heads
        self.heads = heads
        self.scale = dim_head ** -0.5  # Scaling factor for dot-product attention

        # Layer normalization to stabilize training
        self.norm = nn.LayerNorm(dim)

        # Linear layer to project input embeddings to queries, keys, and values
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)

        # Linear layer to project concatenated attention heads back to output dimension
        self.to_out = nn.Linear(inner_dim, dim, bias=False)

        # Dropout layer applied to attention probabilities
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        """
        Performs a forward pass of the attention mechanism.

        Parameters:
        - x (Tensor): Input tensor of shape [batch_size, seq_len, dim].

        Returns:
        - out (Tensor): Output tensor of shape [batch_size, seq_len, dim].
        - attn (Tensor): Attention weights of shape [batch_size, heads, seq_len, seq_len].
        """
        h = self.heads  # Number of attention heads

        # Apply layer normalization to the input
        x = self.norm(x)  # Shape: [batch_size, seq_len, dim]

        # Compute queries, keys, and values in a single linear projection and split them
        qkv = self.to_qkv(x)  # Shape: [batch_size, seq_len, inner_dim * 3]
        q, k, v = qkv.chunk(3, dim=-1)  # Each has shape: [batch_size, seq_len, inner_dim]

        # Reshape and transpose for multi-head attention
        # New shape: [batch_size, heads, seq_len, dim_head]
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=h), (q, k, v))

        # Scale queries to prevent large dot-product values
        q = q * self.scale

        # Compute attention scores via scaled dot-product
        # sim shape: [batch_size, heads, seq_len, seq_len]
        sim = torch.einsum('b h i d, b h j d -> b h i j', q, k)

        # Apply softmax to get attention probabilities
        attn = sim.softmax(dim=-1)

        # Apply dropout to attention probabilities
        attn = self.dropout(attn)

        # Compute weighted sum of values
        # out shape: [batch_size, heads, seq_len, dim_head]
        out = torch.einsum('b h i j, b h j d -> b h i d', attn, v)

        # Concatenate attention heads
        # New shape: [batch_size, seq_len, inner_dim]
        out = rearrange(out, 'b h n d -> b n (h d)')

        # Project back to output dimension
        out = self.to_out(out)  # Shape: [batch_size, seq_len, dim]

        return out, attn  # Return the output and attention weights
    
class SimpleTransformer(nn.Module):
    """
    Transformer model consisting of stacked attention and feed-forward layers.

    Args:
        dim (int): Dimension of the input embeddings.
        depth (int): Number of layers in the Transformer.
        heads (int): Number of attention heads.
        dim_head (int): Dimension of each attention head.
        attn_dropout (float): Dropout rate for attention layers.
        ff_dropout (float): Dropout rate for feed-forward layers.
    """
    def __init__(
        self,
        dim,
        depth,
        heads,
        dim_head,
        attn_dropout,
        ff_dropout
    ):
        super().__init__()
        # Initialize a list to hold the layers of the Transformer
        self.layers = nn.ModuleList([])

        # Build each layer consisting of an attention and a feed-forward module
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                # Multi-Head Self-Attention layer
                MultiHeadSelfAttention(dim, heads=heads, dim_head=dim_head, dropout=attn_dropout),
                # Feed-forward layer
                FeedForward(dim, dropout=ff_dropout),
            ]))

    def forward(self, x, return_attn=False):
        """
        Forward pass through the Transformer.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_len, dim).
            return_attn (bool, optional): If True, returns attention maps. Defaults to False.

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, seq_len, dim).
            torch.Tensor (optional): Stack of attention maps if return_attn is True.
        """
        # List to collect attention maps from each layer
        post_softmax_attns = []

        # Iterate over each layer in the Transformer
        for attn, ff in self.layers:
            # Apply attention layer
            attn_out, post_softmax_attn = attn(x)
            # Collect attention maps
            post_softmax_attns.append(post_softmax_attn)

            # Residual connection for attention layer
            x = attn_out + x
            # Apply feed-forward layer with residual connection
            x = ff(x) + x

        if not return_attn:
            # Return the final output
            return x

        # Return the final output and the stack of attention maps
        return x, torch.stack(post_softmax_attns)

# Unit Test
import unittest

class TestSimpleTransformer(unittest.TestCase):
    def setUp(self):
        # Set up the Transformer model with some arbitrary but valid dimensions
        self.model = SimpleTransformer(
            dim=64,        # Input embedding dimension
            depth=2,       # Number of transformer layers
            heads=4,       # Number of attention heads
            dim_head=16,   # Dimension of each attention head
            attn_dropout=0.1,  # Dropout for attention layers
            ff_dropout=0.1     # Dropout for feed-forward layers
        )
    
    def test_forward_pass(self):
        # Create a dummy input tensor of shape (batch_size, seq_len, dim)
        batch_size = 2
        seq_len = 10
        dim = 64
        x = torch.randn(batch_size, seq_len, dim)
        
        # Run a forward pass through the model
        output = self.model(x)

        # Check that the output has the correct shape
        self.assertEqual(output.shape, (batch_size, seq_len, dim))

if __name__ == "__main__":
    unittest.main()