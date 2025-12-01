import torch
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
import torch.nn as nn
import torch.optim as optim
from model import ViTEncoder, Decoder
from dataset import CaptionDataset
import json
from tqdm import tqdm
import warnings

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# ==================== SETTINGS ====================
BATCH_SIZE = 32
PAD_IDX = 0
NUM_EPOCHS = 10
EMBED_SIZE = 256
HIDDEN_DIM = 512

# ==================== LOAD DATA ====================
captions_json = "/kaggle/input/caption-img/captions_img.json"
with open("/kaggle/input/caption-img/word2idx.json", "r", encoding="utf-8") as f:
    w2i = json.load(f)
idx2w = {v: k for k, v in w2i.items()}

train_data = CaptionDataset("/kaggle/input/caption-img/processed/train", captions_json, w2i)
val_data   = CaptionDataset("/kaggle/input/caption-img/processed/val", captions_json, w2i)

def collate_fn(batch):
    imgs, captions = zip(*batch)
    imgs = torch.stack(imgs, 0)
    captions = [torch.tensor(c) for c in captions]
    captions = pad_sequence(captions, batch_first=True, padding_value=PAD_IDX)
    return imgs, captions

train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True,
                          collate_fn=collate_fn, num_workers=2)
val_loader   = DataLoader(val_data, batch_size=BATCH_SIZE, shuffle=False,
                          collate_fn=collate_fn, num_workers=2)

# ==================== MODEL ====================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
print(f"Vocabulary size: {len(w2i)}")

encoder = ViTEncoder(embed_dim=EMBED_SIZE).to(device)
decoder = Decoder(embed_dim=EMBED_SIZE, vocab_size=len(w2i)).to(device)

criterion = nn.CrossEntropyLoss(ignore_index=PAD_IDX)
optimizer = optim.AdamW(list(decoder.parameters()) +
                        list(encoder.linear.parameters()) +
                        list(encoder.bn.parameters()), lr=3e-4)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2, verbose=True)

best_val_loss = float('inf')

# ==================== TRAINING LOOP ====================
for epoch in range(NUM_EPOCHS):
    encoder.train()
    decoder.train()
    train_loss = 0
    pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS}")
    
    for images, captions in pbar:
        images = images.to(device)
        captions = captions.to(device)

        optimizer.zero_grad()
        tgt_input = captions[:, :-1].transpose(0,1)   # (seq_len, B)
        target = captions[:, 1:].reshape(-1)          # flatten

        output = decoder(tgt_input, encoder(images))
        loss = criterion(output.reshape(-1, len(w2i)), target)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        pbar.set_postfix({'loss': loss.item()})
    
    avg_train_loss = train_loss / len(train_loader)

    # Validation
    encoder.eval()
    decoder.eval()
    val_loss = 0
    with torch.no_grad():
        for images, captions in val_loader:
            images = images.to(device)
            captions = captions.to(device)

            tgt_input = captions[:, :-1].transpose(0,1)
            target = captions[:, 1:].reshape(-1)

            output = decoder(tgt_input, encoder(images))
            loss = criterion(output.reshape(-1, len(w2i)), target)
            val_loss += loss.item()
    avg_val_loss = val_loss / len(val_loader)
    scheduler.step(avg_val_loss)

    print(f"\nEpoch [{epoch+1}/{NUM_EPOCHS}] Train Loss: {avg_train_loss:.4f} Val Loss: {avg_val_loss:.4f}")

    # Save best model
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        torch.save({
            'epoch': epoch,
            'encoder_state_dict': encoder.state_dict(),
            'decoder_state_dict': decoder.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'val_loss': avg_val_loss,
        }, 'best_model.pth')
        print(f"Saved best model with Val Loss: {avg_val_loss:.4f}")
