import torch
from torchvision import models
import torch.nn as nn
import torch.optim as optim
import math
# Feature extraction
class Encoder(nn.Module):

    def __init__(self, embed_size):
        super().__init__()
        resnet = models.resnet50(pretrained=True)
        modules = list(resnet.children())[:-1] # remove last fully connected
        self.resnet = nn.Sequential(*modules)
        self.linear = nn.Linear(resnet.fc.in_features, embed_size)
        self.bn = nn.BatchNorm1d(embed_size, momentum=0.01)

    def forward(self,imgs):
        with torch.no_grad(): # freeze resnet
            features = self.resnet(imgs)
        features = features.view(features.size(0), -1)
        features = self.bn(self.linear(features))

        return features # (B, embed_size)

# Positional Encoding

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1)]
        return x


# Transformer Decoder
class Decoder(nn.Module):
    def __init__(self, embed_size, vocab_size, num_layers=3, num_heads=4, hidden_dim=512, max_len=50):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_size,padding_idx=0)
        self.pos_enc = PositionalEncoding(d_model=embed_size, max_len=max_len)
        decoder_layer = nn.TransformerDecoderLayer(d_model=embed_size, nhead=num_heads, dim_feedforward=hidden_dim)
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers)

        self.fc = nn.Linear (embed_size, vocab_size)

    def forward(self, tgt, memory):
        '''
        tgt: (seq_len, B) token ID input
        memory: (B, embed_size) image features
        '''
        tgt_emb = self.embed(tgt)       # (seq_len, B, embed_size)
        tgt_emb = self.pos_enc(tgt_emb) # add positional encoding
        memory = memory.unsqueeze(0)
        seq_len = tgt_emb.size(0)
        mask = nn.Transformer.generate_square_subsequent_mask(seq_len).to(tgt.device)
        out = self.transformer_decoder(tgt_emb, memory, tgt_mask=mask)
        out = self.fc(out)          # (seq_len, B, vocab_size)

        return out



