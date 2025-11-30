from dataset import CaptionDataset
import json
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
import torch
from torchvision import models
import torch.nn as nn
import torch.optim as optim
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
vocabsize = len(w2i)
num_epochs = 5
device = torch.device("cuda"if torch.cuda.is_available else "cpu")

# Model, Loss, Optimizer
