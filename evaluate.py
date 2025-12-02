import torch
from torch.utils.data import DataLoader
from dataset import CaptionDataset
from model import Encoder, Decoder
import json
from nltk.translate.bleu_score import corpus_bleu

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
PAD_IDX = 0
MAX_LEN = 20
NUM_EXAMPLES = 5  # số ảnh in ví dụ

# ===== Load vocab =====
with open("/kaggle/input/caption-img/word2idx.json", "r", encoding="utf-8") as f:
    w2i = json.load(f)
i2w = {v:k for k,v in w2i.items()}
vocab_size = len(w2i)

# ===== Load test dataset =====
test_dataset = CaptionDataset("/kaggle/input/caption-img/processed/test", "/kaggle/input/caption-img/captions_img.json", w2i)

def collate_fn(batch):
    imgs, caps = zip(*batch)
    imgs = torch.stack(imgs)
    return imgs, caps

test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, collate_fn=collate_fn)

# ===== Load model =====
encoder = Encoder().to(DEVICE)
decoder = Decoder(vocab_size=vocab_size, pad_idx=PAD_IDX).to(DEVICE)

checkpoint = torch.load("best_model.pt", map_location=DEVICE)
encoder.load_state_dict(checkpoint['encoder'])
decoder.load_state_dict(checkpoint['decoder'])

encoder.eval()
decoder.eval()

# ===== Generate caption (greedy) =====
def generate_caption(encoder, decoder, image, max_len=MAX_LEN):
    with torch.no_grad():
        image = image.unsqueeze(0).to(DEVICE)
        features = encoder(image)
        caption_ids = [w2i["<start>"]]

        for _ in range(max_len):
            input_ids = torch.tensor(caption_ids, dtype=torch.long).unsqueeze(0).to(DEVICE)
            outputs = decoder(input_ids, features)
            next_id = outputs[0,-1].argmax().item()
            caption_ids.append(next_id)
            if next_id == w2i["<end>"]:
                break

    words = [i2w[idx] for idx in caption_ids[1:] if idx != w2i["<end>"]]
    return words

# ===== Evaluate & print examples =====
references = []
hypotheses = []

print("\n=== Example Captions ===")
for idx, (imgs, caps_list) in enumerate(test_loader):
    if idx >= NUM_EXAMPLES:
        break
    img = imgs[0]
    refs_words = [
        [i2w[idx.item()] for idx in cap.view(-1) if idx.item() not in (PAD_IDX, w2i["<start>"], w2i["<end>"])]
        for cap in caps_list
    ]
    pred_words = generate_caption(encoder, decoder, img)

    references.append(refs_words)
    hypotheses.append(pred_words)

    print(f"\nImage {idx+1}:")
    print("Reference Captions:")
    for r in refs_words:
        print(" ", " ".join(r))
    print("Predicted Caption:")
    print(" ", " ".join(pred_words))

# ===== Compute BLEU =====
bleu1 = corpus_bleu(references, hypotheses, weights=(1,0,0,0))
bleu2 = corpus_bleu(references, hypotheses, weights=(0.5,0.5,0,0))
bleu3 = corpus_bleu(references, hypotheses, weights=(0.33,0.33,0.33,0))
bleu4 = corpus_bleu(references, hypotheses, weights=(0.25,0.25,0.25,0.25))

print("\n=== BLEU Scores ===")
print(f"BLEU-1: {bleu1*100:.2f}")
print(f"BLEU-2: {bleu2*100:.2f}")
print(f"BLEU-3: {bleu3*100:.2f}")
print(f"BLEU-4: {bleu4*100:.2f}")
