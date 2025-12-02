import torch
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
import torch.nn as nn
import torch.optim as optim
from dataset import CaptionDataset
from model import ViTEncoder as Encoder, Decoder
import json
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")

# ==================== SETTINGS ====================
BATCH_SIZE = 32
PAD_IDX = 0
EPOCHS = 10
EMBED = 256
HIDDEN = 512

# ==================== LOAD DATA ====================
captions_json = "/kaggle/input/caption-img/captions_img.json"
with open("/kaggle/input/caption-img/word2idx.json","r",encoding="utf-8") as f:
    w2i = json.load(f)
idx2w = {v:k for k,v in w2i.items()}

train_data = CaptionDataset("/kaggle/input/caption-img/processed/train", captions_json, w2i)
val_data   = CaptionDataset("/kaggle/input/caption-img/processed/val", captions_json, w2i)

def collate_fn(batch):
    imgs, caps = zip(*batch)
    imgs = torch.stack(imgs)
    caps = pad_sequence([torch.tensor(c) for c in caps], batch_first=True, padding_value=PAD_IDX)
    return imgs, caps

train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True,
                          collate_fn=collate_fn, num_workers=2)
val_loader   = DataLoader(val_data, batch_size=BATCH_SIZE, shuffle=False,
                          collate_fn=collate_fn, num_workers=2)

# ==================== MODEL ====================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)

encoder = Encoder(embed_dim=EMBED).to(device)
decoder = Decoder(embed_dim=EMBED, vocab_size=len(w2i)).to(device)

# Unfreeze block cuối của ViT → rất quan trọng
for name, param in encoder.vit.named_parameters():
    if "encoder.layers.encoder_layer_11" in name or "encoder.layer.11" in name:
        param.requires_grad = True
    else:
        param.requires_grad = False

criterion = nn.CrossEntropyLoss(ignore_index=PAD_IDX, label_smoothing=0.1)

optimizer = optim.AdamW(
    list(encoder.parameters()) + list(decoder.parameters()),
    lr=3e-4, weight_decay=1e-4
)

# Warmup cho Transformer
def warmup(step, warmup_steps=4000):
    step = max(step, 1)
    return min(step ** (-0.5), step * warmup_steps ** (-1.5))

scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=warmup)

best_val_loss = float("inf")
global_step = 0

# ==================== TRAIN LOOP ====================
for epoch in range(EPOCHS):
    encoder.train()
    decoder.train()
    train_loss = 0

    pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}")

    for images, caps in pbar:
        images = images.to(device)
        caps = caps.to(device)

        optimizer.zero_grad()

        tgt_in  = caps[:, :-1]       # (B, L-1)
        tgt_out = caps[:, 1:]        # (B, L-1)

        tgt_in  = tgt_in.transpose(0,1)   # (L, B)
        tgt_out = tgt_out.reshape(-1)

        tgt_mask = nn.Transformer.generate_square_subsequent_mask(tgt_in.size(0)).to(device)
        tgt_key_padding = (tgt_in.transpose(0,1) == PAD_IDX)

        memory = encoder(images)          # (B, EMBED)
        memory = memory.unsqueeze(0)      # (1, B, EMBED)

        logits = decoder(
            tgt_in, 
            memory.squeeze(0)
        )  # (L, B, vocab)

        loss = criterion(logits.reshape(-1, logits.size(-1)), tgt_out)
        loss.backward()

        torch.nn.utils.clip_grad_norm_(decoder.parameters(), 1.0)

        optimizer.step()
        scheduler.step()
        global_step += 1

        train_loss += loss.item()
        pbar.set_postfix({"loss": loss.item()})

    avg_train_loss = train_loss / len(train_loader)

    # ==================== VALIDATION ====================
    encoder.eval()
    decoder.eval()
    val_loss = 0
    with torch.no_grad():
        for images, caps in val_loader:
            images = images.to(device)
            caps = caps.to(device)

            tgt_in  = caps[:, :-1].transpose(0,1)
            tgt_out = caps[:, 1:].reshape(-1)

            memory = encoder(images).unsqueeze(0)
            logits = decoder(tgt_in, memory.squeeze(0))

            loss = criterion(logits.reshape(-1, logits.size(-1)), tgt_out)
            val_loss += loss.item()

    avg_val_loss = val_loss / len(val_loader)
    print(f"\nEpoch {epoch+1}: Train Loss = {avg_train_loss:.4f} | Val Loss = {avg_val_loss:.4f}")

    # Save model
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        torch.save({
            "encoder": encoder.state_dict(),
            "decoder": decoder.state_dict(),
            "val_loss": avg_val_loss
        }, "best_vit_caption.pth")
        print(f"Saved BEST model (Val Loss = {avg_val_loss:.4f})")
