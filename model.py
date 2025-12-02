import torch
import torch.nn as nn
from torchvision import models


# ENCODER (ViT)
class ViTEncoder(nn.Module):
    def __init__(self, embed_dim=512, train_last_block=False):
        super().__init__()

        vit = models.vit_b_16(weights=models.ViT_B_16_Weights.IMAGENET1K_V1)

        # remove classification head
        vit.heads = nn.Identity()
        self.vit = vit

        # freeze toàn bộ backbone
        for p in self.vit.parameters():
            p.requires_grad = False

        # nếu muốn train block cuối
        if train_last_block:
            for p in self.vit.encoder.layers[-1].parameters():
                p.requires_grad = True

        # ViT output = 768 → project về embed_dim
        self.proj = nn.Linear(768, embed_dim)

    def forward(self, images):
        feat = self.vit(images)          # [B, 768]
        feat = self.proj(feat)           # [B, embed_dim]
        return feat.unsqueeze(1)         # [B, 1, embed_dim] 


# DECODER (Transformer)
class TransformerCaptionDecoder(nn.Module):
    def __init__(self, embed_dim, vocab_size, num_layers=4, num_heads=8, dropout=0.1):
        super().__init__()

        self.embed_dim = embed_dim
        self.vocab_size = vocab_size

        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.pos_embedding = nn.Embedding(500, embed_dim)

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=2048,
            dropout=dropout,
            batch_first=True
        )

        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
        self.fc_out = nn.Linear(embed_dim, vocab_size)

    def forward(self, captions, memory):
        """
        captions: [B, T]
        memory:   [B, 1, E]  (encoder output)
        """
        B, T = captions.size()

        positions = torch.arange(0, T, device=captions.device).unsqueeze(0)
        tgt = self.embedding(captions) + self.pos_embedding(positions)

        # mask decoder
        causal_mask = nn.Transformer.generate_square_subsequent_mask(T).to(captions.device)

        out = self.decoder(
            tgt=tgt,
            memory=memory,
            tgt_mask=causal_mask
        )  # [B, T, E]

        logits = self.fc_out(out)  # [B, T, vocab]
        return logits


# IMAGE CAPTIONING MODEL
class ImageCaptioningModel(nn.Module):
    def __init__(self, vocab_size, embed_dim=512, train_last_block=False):
        super().__init__()

        self.encoder = ViTEncoder(embed_dim, train_last_block=train_last_block)
        self.decoder = TransformerCaptionDecoder(embed_dim, vocab_size)

    def forward(self, images, captions):
        memory = self.encoder(images)      # [B, 1, E]
        outputs = self.decoder(captions, memory)
        return outputs
