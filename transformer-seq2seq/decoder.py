import torch.nn as nn
from attention import MultiHeadAttention

class DecoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff=512):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads)
        self.cross_attn = MultiHeadAttention(d_model, num_heads)
        self.ff = nn.Sequential(nn.Linear(d_model, d_ff), nn.ReLU(), nn.Linear(d_ff, d_model))
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)

    def forward(self, x, enc_out, src_mask, tgt_mask):
        # 1. Masked Self-Attention [cite: 63]
        x_attn, _ = self.self_attn(x, x, x, tgt_mask)
        x = self.norm1(x + x_attn)
        # 2. Cross-Attention with Encoder Output [cite: 78]
        x_cross, attn_map = self.cross_attn(x, enc_out, enc_out, src_mask)
        x = self.norm2(x + x_cross)
        # 3. Feed Forward
        x = self.norm3(x + self.ff(x))
        return x, attn_map