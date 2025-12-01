from dataset import CaptionDataset
import json
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
import torch
from torchvision import models
import torch.nn as nn
import torch.optim as optim
from model import ViTEncoder, Decoder
from tqdm import tqdm
import warnings
from nltk.translate.bleu_score import corpus_bleu, sentence_bleu
import random

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# SETUP
BATCH_SIZE = 32
PAD_IDX = 0

captions_json = "data/captions_img.json"
with open("data/word2idx.json", "r", encoding="utf-8") as f:
    w2i = json.load(f)

# Tạo idx2word để decode predictions
idx2w = {v: k for k, v in w2i.items()}

train_data = CaptionDataset("data/processed/train",
                            captions_json, w2i)
val_data = CaptionDataset("data/processed/val",
                            captions_json, w2i)
test_data = CaptionDataset("data/processed/test",
                            captions_json, w2i)

# split batch
def collate_fn(batch):
    imgs, captions = zip(*batch)
    imgs = torch.stack(imgs, 0)
    captions = [torch.tensor(c) for c in captions]
    captions = pad_sequence(captions, batch_first=True, padding_value=PAD_IDX)
    return imgs, captions

def decode_caption(indices, idx2word):
    """Chuyển đổi indices thành text"""
    words = []
    for idx in indices:
        if idx == PAD_IDX:
            continue
        if idx in idx2word:
            word = idx2word[idx]
            if word == '<end>':
                break
            if word != '<start>':
                words.append(word)
    return ' '.join(words)

def generate_caption(encoder, decoder, image, max_len=50, device='cuda'):
    encoder.eval()
    decoder.eval()
    
    with torch.no_grad():
        features = encoder(image.unsqueeze(0).to(device))  # (1, embed_dim)
        features = features.unsqueeze(0)  # (1, 1, embed_dim) -> seq_len_memory=1, batch=1, embed_dim
        
        # Bắt đầu với <start> token
        start_token = w2i.get('<start>', 1)
        caption = [start_token]
        
        for _ in range(max_len):
            tgt_input = torch.tensor(caption).unsqueeze(1).to(device)  # (seq_len, 1)
            output = decoder(tgt_input, features)  # (seq_len, 1, vocab_size)
            
            # Lấy token có xác suất cao nhất
            pred = output[-1, 0].argmax().item()
            caption.append(pred)
            
            # Dừng nếu gặp <end> token
            if pred == w2i.get('<end>', 2):
                break
                
    return caption


def calculate_bleu(encoder, decoder, data_loader, device, max_samples=None):
    """Tính BLEU score trên toàn bộ dataset"""
    encoder.eval()
    decoder.eval()
    
    references = []
    hypotheses = []
    
    with torch.no_grad():
        for i, (images, captions) in enumerate(tqdm(data_loader, desc="Calculating BLEU")):
            if max_samples and i >= max_samples:
                break
                
            images = images.to(device)
            
            # Generate captions cho batch
            for j in range(images.size(0)):
                # Generate caption
                pred_caption = generate_caption(encoder, decoder, images[j], device=device)
                
                # Ground truth caption
                true_caption = captions[j].cpu().tolist()
                
                # Chuyển thành words và loại bỏ special tokens
                pred_words = [idx2w.get(idx, '<unk>') for idx in pred_caption 
                             if idx not in [PAD_IDX, w2i.get('<start>', 1), w2i.get('<end>', 2)]]
                true_words = [idx2w.get(idx, '<unk>') for idx in true_caption 
                             if idx not in [PAD_IDX, w2i.get('<start>', 1), w2i.get('<end>', 2)]]
                
                references.append([true_words])
                hypotheses.append(pred_words)
    
    # Tính BLEU score
    bleu1 = corpus_bleu(references, hypotheses, weights=(1, 0, 0, 0))
    bleu2 = corpus_bleu(references, hypotheses, weights=(0.5, 0.5, 0, 0))
    bleu3 = corpus_bleu(references, hypotheses, weights=(0.33, 0.33, 0.33, 0))
    bleu4 = corpus_bleu(references, hypotheses, weights=(0.25, 0.25, 0.25, 0.25))
    
    return bleu1, bleu2, bleu3, bleu4, references, hypotheses

def show_predictions(encoder, decoder, data_loader, device, num_samples=5):
    """Hiển thị một số predictions"""
    encoder.eval()
    decoder.eval()
    
    print("\n" + "="*80)
    print("SAMPLE PREDICTIONS")
    print("="*80)
    
    # Lấy random samples
    images, captions = next(iter(data_loader))
    indices = random.sample(range(len(images)), min(num_samples, len(images)))
    
    for idx in indices:
        image = images[idx]
        true_caption = captions[idx].cpu().tolist()
        
        # Generate prediction
        pred_caption = generate_caption(encoder, decoder, image, device=device)
        
        # Decode captions
        true_text = decode_caption(true_caption, idx2w)
        pred_text = decode_caption(pred_caption, idx2w)
        
        # Tính BLEU cho caption này
        true_words = [idx2w.get(idx, '<unk>') for idx in true_caption 
                     if idx not in [PAD_IDX, w2i.get('<start>', 1), w2i.get('<end>', 2)]]
        pred_words = [idx2w.get(idx, '<unk>') for idx in pred_caption 
                     if idx not in [PAD_IDX, w2i.get('<start>', 1), w2i.get('<end>', 2)]]
        
        bleu = sentence_bleu([true_words], pred_words, weights=(0.25, 0.25, 0.25, 0.25))
        
        print(f"\nSample {idx + 1}:")
        print(f"Ground Truth: {true_text}")
        print(f"Predicted:    {pred_text}")
        print(f"BLEU-4:       {bleu:.4f}")
        print("-" * 80)

def main():
    # Data loader
    train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, 
                            shuffle=True, collate_fn=collate_fn, num_workers=4)
    val_loader = DataLoader(val_data, batch_size=BATCH_SIZE, 
                            shuffle=False, collate_fn=collate_fn, num_workers=4)
    test_loader = DataLoader(test_data, batch_size=BATCH_SIZE, 
                            shuffle=False, collate_fn=collate_fn)

    # Hyperparameters 
    embed_size = 256
    hidden_dim = 512
    vocab_size = len(w2i)
    num_epochs = 10
    global_step = 0
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"Vocabulary size: {vocab_size}")
    
    # Model, Loss, Optimizer
    encoder = ViTEncoder(embed_dim=embed_size).to(device)
    decoder = Decoder(embed_size, vocab_size).to(device)
    criterian = nn.CrossEntropyLoss(ignore_index=PAD_IDX)
    params = list(decoder.parameters()) + list(encoder.proj.parameters())
    optimizer = optim.AdamW(params, lr=3e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2, verbose=True)

    best_val_loss = float('inf')
    
    # Training loop
    for epoch in range(num_epochs):
        encoder.train()
        decoder.train()
        train_loss = 0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")

        for i, (images, captions) in enumerate(pbar):
            images = images.to(device)
            captions = captions.to(device)
            
            optimizer.zero_grad()
            tgt_input = captions[:, :-1].transpose(0, 1)
            target = captions[:, 1:].reshape(-1)

            output = decoder(tgt_input, encoder(images))
            loss = criterian(output.reshape(-1, vocab_size), target)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            global_step = epoch * len(train_loader) + i
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

                tgt_input = captions[:, :-1].transpose(0, 1)
                target = captions[:, 1:].reshape(-1)
                
                features = encoder(images)
                output = decoder(tgt_input, features)

                loss = criterian(output.reshape(-1, vocab_size), target)
                val_loss += loss.item()
                
        avg_val_loss = val_loss / len(val_loader)
        
        # Update learning rate
        scheduler.step(avg_val_loss)
        
        print(f"Epoch [{epoch+1}/{num_epochs}]")
        print(f"Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")
        
        # Hiển thị một số predictions sau mỗi epoch để kiểm tra
        if (epoch + 1) % 2 == 0:
            show_predictions(encoder, decoder, val_loader, device, num_samples=2)
        
        # Save best model based on validation loss
        if epoch == 0 or avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save({
                'epoch': epoch,
                'encoder_state_dict': encoder.state_dict(),
                'decoder_state_dict': decoder.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': avg_val_loss,
            }, 'best_model.pth')
            print(f"Saved best model with Val Loss: {avg_val_loss:.4f}")
    
    # Final evaluation on test set
    print("\n" + "="*80)
    print("FINAL EVALUATION ON TEST SET")
    print("="*80)
    
    # Load best model
    checkpoint = torch.load('best_model.pth')
    encoder.load_state_dict(checkpoint['encoder_state_dict'])
    decoder.load_state_dict(checkpoint['decoder_state_dict'])
    
    bleu1, bleu2, bleu3, bleu4, _, _ = calculate_bleu(
        encoder, decoder, test_loader, device
    )
    
    print(f"\nTest Set Results:")
    print(f"BLEU-1: {bleu1:.4f}")
    print(f"BLEU-2: {bleu2:.4f}")
    print(f"BLEU-3: {bleu3:.4f}")
    print(f"BLEU-4: {bleu4:.4f}")
    
    # Show test predictions
    show_predictions(encoder, decoder, test_loader, device, num_samples=5)
    
if __name__ == "__main__":
    main()