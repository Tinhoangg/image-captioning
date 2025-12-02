import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from model import ImageCaptioningModel   # model bạn đã tạo
from dataset import CaptionDataset       # dataset của bạn


# ------------------------------
# MAKE PAD MASK
# ------------------------------
def create_masks(captions, pad_idx):
    """
    captions: [B, T]
    """
    # target mask để decoder không nhìn tương lai
    T = captions.size(1)
    causal_mask = nn.Transformer.generate_square_subsequent_mask(T).to(captions.device)

    # padding mask: 1 = pad, 0 = real token
    pad_mask = (captions == pad_idx)  # [B, T]

    return causal_mask, pad_mask


# ------------------------------
# TRAIN ONE EPOCH
# ------------------------------
def train_one_epoch(model, dataloader, optimizer, criterion, pad_idx):
    model.train()
    total_loss = 0

    for images, captions in tqdm(dataloader, desc="Training"):
        images = images.to(device)
        captions = captions.to(device)

        # Input = tất cả trừ token cuối
        # Target = tất cả trừ token đầu
        inputs = captions[:, :-1]
        targets = captions[:, 1:]

        optimizer.zero_grad()

        outputs = model(images, inputs)  # [B, T, vocab]

        loss = criterion(outputs.reshape(-1, outputs.size(-1)),
                         targets.reshape(-1))

        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(dataloader)


# ------------------------------
# VALIDATION
# ------------------------------
def validate(model, dataloader, criterion, pad_idx):
    model.eval()
    total_loss = 0

    with torch.no_grad():
        for images, captions in tqdm(dataloader, desc="Validating"):
            images = images.to(device)
            captions = captions.to(device)

            inputs = captions[:, :-1]
            targets = captions[:, 1:]

            outputs = model(images, inputs)

            loss = criterion(outputs.reshape(-1, outputs.size(-1)),
                             targets.reshape(-1))

            total_loss += loss.item()

    return total_loss / len(dataloader)


# ------------------------------
# MAIN TRAIN LOOP
# ------------------------------
if __name__ == "__main__":
    # ----------------
    # CONFIG
    # ----------------
    BATCH_SIZE = 32
    EPOCHS = 10
    LR = 1e-4
    PAD_IDX = 0

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # ----------------
    # LOAD VOCAB
    # ----------------
    import json
    with open("/kaggle/input/caption-img/word2idx.json", "r") as f:
        w2i = json.load(f)
    vocab_size = len(w2i)

    # ----------------
    # DATASET
    # ----------------
    train_data = CaptionDataset(
        "/kaggle/input/caption_img/processed/train",
        "/kaggle/input/caption-img/captions_img.json",
        w2i
    )
    val_data = CaptionDataset(
        "/kaggle/input/caption_img/processed/val",
        "/kaggle/input/caption-img/captions_img.json",
        w2i
    )

    train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
    val_loader   = DataLoader(val_data,   batch_size=BATCH_SIZE)

    # ----------------
    # MODEL
    # ----------------
    model = ImageCaptioningModel(
        vocab_size=vocab_size,
        embed_dim=512,
        train_last_block=False
    ).to(device)

    criterion = nn.CrossEntropyLoss(ignore_index=PAD_IDX)
    optimizer = optim.Adam(model.parameters(), lr=LR)

    # ----------------
    # TRAINING LOOP
    # ----------------
    best_val_loss = float("inf")

    for epoch in range(1, EPOCHS + 1):
        print(f"\n===== EPOCH {epoch}/{EPOCHS} =====")

        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, PAD_IDX)
        val_loss   = validate(model, val_loader, criterion, PAD_IDX)

        print(f"Train Loss: {train_loss:.4f}")
        print(f"Val   Loss: {val_loss:.4f}")

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), "best_model.pth")
            print(">>> Saved best model!")

    print("Training complete.")
