import torch.nn as nn
from attention import MultiHeadAttention

class EncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff=512):
        super(EncoderLayer, self).__init__()
        self.attn = MultiHeadAttention(d_model, num_heads)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Linear(d_ff, d_model)
        )

    def forward(self, x, mask=None):
        # Attention + Residual + Norm
        attn_out, weights = self.attn(x, x, x, mask)
        x = self.norm1(x + attn_out)
        # Feed Forward + Residual + Norm
        ff_out = self.ff(x)
        x = self.norm2(x + ff_out)
        return x, weights