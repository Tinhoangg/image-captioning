import torch
from torchvision import models
import torch.nn as nn
import torch.optim as optim
import math

# Feature extraction
class ViTEncoder(nn.Module):

    def __init__(self, embed_dim):
        super().__init__()
        vit = models.vit_b_16(weights=models.ViT_B_16_Weights.IMAGENET1K_V1)
        vit.heads = nn.Identity() # remove classification head
        self.vit = vit
        self.embed_dim = embed_dim
        self.proj = nn.Linear(768, embed_dim)

    def forward(self, x):
        x = self.vit(x)
        x = self.proj(x)
        return x
    

# Positional Encoding

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()

        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000)/d_model))

        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)

        self.pe = pe.unsqueeze(1)

    def forward(self, x):
        return x + self.pe[:x.size(0)].to(x.device)



# Transformer Decoder
class Decoder(nn.Module):
    def __init__(self, embed_dim, vocab_size, num_layers=3, num_heads=4, hidden_dim=512, max_len=5000, dropout=0.2):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.pos_enc = PositionalEncoding(d_model=embed_dim, max_len=max_len)
        decoder_layer = nn.TransformerDecoderLayer(d_model=embed_dim, nhead=num_heads, dim_feedforward=hidden_dim,dropout=dropout)
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers)

        self.fc = nn.Linear (embed_dim, vocab_size)

    def forward(self, tgt, memory):
        '''
        tgt: (seq_len, B) token ID input
        memory: (B, embed_dim) image features
        '''
        tgt_emb = self.embed(tgt)       # (seq_len, B, embed_dim)
        tgt_emb = self.pos_enc(tgt_emb) # add positional encoding

        memory = memory.unsqueeze(0)

        seq_len = tgt_emb.size(0)
        mask = nn.Transformer.generate_square_subsequent_mask(seq_len).to(tgt.device)
        out = self.transformer_decoder(tgt_emb, 
                                       memory, 
                                       tgt_mask=mask)
        
        out = self.fc(out)          # (seq_len, B, vocab_size)

        return out



