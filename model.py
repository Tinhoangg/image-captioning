import torch
import torch.nn as nn
from torchvision import models

class Encoder(nn.Module):
    def __init__(self, embed_dim=512, train_cnn=False):
        super().__init__()

        # load resnet50 pre-trained model
        resnet = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)

        # remove last fc layer
        self.cnn = nn.Sequential(*list(resnet.children())[:-1]) # output(B,2048,1,1)

        # Linear projection 2048 -> embed dim
        self.fc = nn.Linear(2048, embed_dim)
        self.bn = nn.BatchNorm1d(embed_dim, momentum=0.01)
        self.relu = nn.ReLU()

        # freeze cnn layer
        for p in self.cnn.parameters():
            p.requires_grad = train_cnn
    def forward(self, images):
        '''
        images: (B,3,H,W)
        return: (B,embed_dim)
        '''
        features = self.cnn(images)  # (B,2048,1,1)
        features = features.view(features.size(0), -1)  # (B,2048)
        out = self.fc(features)  # (B,embed_dim)
        out = self.bn(out)
        out = self.relu(out)
        return out
class Decoder(nn.Module):
    def __init__(self,vocab_size, embed_dim=512, hidden_dim=512,encoder_dim=512, num_layer=3,pad_idx=0, dropout=0.3):
        super().__init__()
        
        self.vocab_size = vocab_size
        self.pad_idx = pad_idx

        #LSTM
        self.embed = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_idx)
        self.lstm = nn.LSTM(embed_dim, 
                            hidden_dim, 
                            num_layers=num_layer,
                            batch_first=True, 
                            dropout=dropout)
        
        self.fc = nn.Linear(hidden_dim, vocab_size)

        self.init_h = nn.Linear(encoder_dim, hidden_dim)
        self.init_c = nn.Linear(encoder_dim, hidden_dim)
        self.num_layers = num_layer


    def forward(self, captions, features):
        '''
        captions: (B, max_len)
        features: (B, encoder_dim)
        return logit: (B, max_len, vocab_size)
        '''
        # embedding token
        embeddings = self.embed(captions)  # (B, max_len, embed_dim)

        h0 = self.init_h(features)  # (B, hidden_dim)
        c0 = self.init_c(features)  # (B, hidden_dim)

        # expand to (num_layers, B, hidden_dim)
        h0 = h0.unsqueeze(0).repeat(self.num_layers, 1, 1)
        c0 = c0.unsqueeze(0).repeat(self.num_layers, 1, 1)

        #LSTM Output
        lstm_out, _ = self.lstm(embeddings, (h0, c0))  # (B, max_len, hidden_dim)

        # predict token
        logits = self.fc(lstm_out)  # (B, max_len, vocab_size)

        return logits

