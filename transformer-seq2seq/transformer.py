import torch.nn as nn
from positional_encoding import PositionalEncoding
# Add these two lines to fix the NameError
from encoder import EncoderLayer 
from decoder import DecoderLayer

class TransformerSeq2Seq(nn.Module):
    def __init__(self, src_vocab, tgt_vocab, d_model=128, num_heads=8):
        super().__init__()
        self.src_emb = nn.Embedding(src_vocab, d_model)
        self.tgt_emb = nn.Embedding(tgt_vocab, d_model)
        self.pos_enc = PositionalEncoding(d_model)
        
        # Now these will work because they are imported above
        self.encoder = EncoderLayer(d_model, num_heads)
        self.decoder = DecoderLayer(d_model, num_heads)
        self.fc_out = nn.Linear(d_model, tgt_vocab)

    def encode(self, src, src_mask):
        return self.encoder(self.pos_enc(self.src_emb(src)), src_mask)

    def decode(self, tgt, enc_out, src_mask, tgt_mask):
        return self.decoder(self.pos_enc(self.tgt_emb(tgt)), enc_out, src_mask, tgt_mask)

    def forward(self, src, tgt, src_mask, tgt_mask):
        enc_out = self.encode(src, src_mask)
        out, attn = self.decode(tgt, enc_out, src_mask, tgt_mask)
        return self.fc_out(out), attn