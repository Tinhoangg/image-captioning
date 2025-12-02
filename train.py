import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
import torch.optim as optim
from dataset import CaptionDataset
from model import Encoder, Decoder   # chỉnh lại nếu khác
import json

# ======= CONFIG =======
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 32
LR = 1e-4
EPOCHS = 50
PAD_IDX = 0
SAVE_PATH = "best_model.pt"

# ======= COLLATE FN =======
def collate_fn(batch):
    imgs, caps = zip(*batch)
    imgs = torch.stack(imgs)

    # pad captions
    caps = pad_sequence(caps, batch_first=True, padding_value=PAD_IDX)

    return imgs, caps

# ======= LOAD VOCAB =======
with open("data/word2idx.json", "r", encoding="utf-8") as f:
    w2i = json.load(f)

vocab_size = len(w2i)
print("Vocab size:", vocab_size)

# ======= LOAD DATA =======
train_dataset = CaptionDataset("data/processed/train", "data/captions_img.json", w2i)
val_dataset   = CaptionDataset("data/processed/val", "data/captions_img.json", w2i)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
val_loader   = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)

# ======= MODEL =======
encoder = Encoder().to(DEVICE)
decoder = Decoder(vocab_size=vocab_size, pad_idx=PAD_IDX).to(DEVICE)

criterion = nn.CrossEntropyLoss(ignore_index=PAD_IDX)
optimizer = optim.Adam(list(encoder.parameters()) + list(decoder.parameters()), lr=LR)


# ======= TRAIN ONE EPOCH =======
def train_one_epoch(epoch):
    encoder.train()
    decoder.train()

    running_loss = 0

    for imgs, caps in train_loader:
        imgs = imgs.to(DEVICE)
        caps = caps.to(DEVICE)

        optimizer.zero_grad()

        # teacher forcing: decoder input = caption[:-1]
        tgt_input = caps[:, :-1]
        tgt_output = caps[:, 1:]

        enc_out = encoder(imgs)
        logits = decoder(tgt_input, enc_out)   # (B, T, V)

        loss = criterion(logits.reshape(-1, vocab_size), tgt_output.reshape(-1))
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    print(f"Epoch {epoch} - Train Loss: {running_loss / len(train_loader):.4f}")
    return running_loss / len(train_loader)


# ======= VALIDATION =======
def validate():
    encoder.eval()
    decoder.eval()

    val_loss = 0

    with torch.no_grad():
        for imgs, caps in val_loader:
            imgs = imgs.to(DEVICE)
            caps = caps.to(DEVICE)

            tgt_input = caps[:, :-1]
            tgt_output = caps[:, 1:]

            enc_out = encoder(imgs)
            logits = decoder(tgt_input, enc_out)

            loss = criterion(logits.reshape(-1, vocab_size), tgt_output.reshape(-1))
            val_loss += loss.item()

    return val_loss / len(val_loader)


# ======= TRAIN LOOP =======
best_val = 1e9

for epoch in range(1, EPOCHS + 1):

    train_loss = train_one_epoch(epoch)
    val_loss = validate()

    print(f"Epoch {epoch} - Val Loss: {val_loss:.4f}")

    # save best model
    if val_loss < best_val:
        best_val = val_loss
        torch.save({
            "encoder": encoder.state_dict(),
            "decoder": decoder.state_dict()
        }, SAVE_PATH)
        print(" Saved Best Model")

print("Done Training!")
