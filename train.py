from dataset import CaptionDataset
import json
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
import torch
from torchvision import models
import torch.nn as nn
import torch.optim as optim
from model import Encoder, Decoder
# SETUP
BATCH_SIZE = 32
PAD_IDX = 0

captions_json = "data/captions_img.json"
with open("data/word2idx.json", "r", encoding="utf-8") as f:
    w2i = json.load(f)

train_data = CaptionDataset("data/processed/train",
                            captions_json, w2i)
val_data = CaptionDataset("data/processed/val",
                            captions_json, w2i)
test_data = CaptionDataset("data/processed/test",
                            captions_json, w2i)

# split batch

def collate_fn(batch):

    imgs, captions = zip(*batch) # unzip batch: list of img and list of caption
    imgs = torch.stack(imgs, 0) # stack img to tensor: (B, 3, H, W)
    # convert caption to tensor
    captions = [torch.tensor(c) for c in captions]
    captions = pad_sequence(captions, batch_first=True, padding_value=PAD_IDX)
    
    return imgs, captions

# Data loader
train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, 
                          shuffle=True, collate_fn=collate_fn)
val_loader = DataLoader(val_data, batch_size=BATCH_SIZE, 
                        shuffle=True, collate_fn=collate_fn)
test_loader = DataLoader(test_data, batch_size=BATCH_SIZE, 
                         shuffle=True, collate_fn=collate_fn)

# Hyperparameters 
embed_size = 256
hidden_dim = 512
vocab_size = len(w2i)
num_epochs = 5
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Model, Loss, Optimizer
encoder = Encoder(embed_size=embed_size).to(device)
decoder = Decoder(embed_size,vocab_size)
criterian = nn.CrossEntropyLoss(ignore_index=PAD_IDX)
params = list(decoder.parameters()) + list(encoder.linear.parameters()) + list(encoder.bn.parameters())
optimizer = optim.AdamW(params, lr=1e-3)

# Training loop
for epoch in range(num_epochs):
    encoder.train()
    decoder.train()
    for i, (images, captions) in enumerate(train_loader):
        images = images.to(device)
        captions = captions.to(device)
        
        optimizer.zero_grad()
        tgt_input = captions[:, :-1].transpose(0, 1) # (seq_len, B)
        target = captions[:, 1:].reshape(-1) # (seq_len*B)

        output = decoder(tgt_input, encoder(images))
        loss = criterian(output.reshape(-1, vocab_size), target)
        loss.backward()
        optimizer.step()

        if i % 100 == 0:
            print(f"Epoch [{epoch+1}/{num_epochs}], Step [{i}/{len(train_loader)}], Loss: {loss.item():.4f}")
