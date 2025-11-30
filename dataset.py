import random
from torch.utils.data import Dataset
import json
import torch
import os
class CaptionDataset(Dataset):
    def __init__(self, img_dir, caption_json, w2i):
        '''
        img_dir: preprocess img folder
        caption_json: captions data file
        w2i: vocab dictionary '''

        self.img_dir = img_dir
        with open(caption_json, 'r', encoding='utf-8') as f:
            all_captions = json.load(f)
        available_img = [f.replace(".pt", ".jpg") for f in os.listdir(img_dir)]


        self.caption_json = {img: caps 
                             for img, caps in all_captions.items()
                             if img in available_img
                             }
        self.w2i = w2i
        self.image_list = list(self.caption_json.keys())
        self.UNK = self.w2i['<unk>']


    def __len__(self):
        return len(self.image_list)
    
    def __getitem__(self, idx):

        img_name = self.image_list[idx]

        # load preprocess image
        img_pt_name = img_name.replace(".jpg", ".pt")
        img_path = os.path.join(self.img_dir, img_pt_name)
        img_tensor = torch.load(img_path)

        # get 1 random caption for image
        raw_caption = random.choice(self.caption_json[img_name])

        # tokenize
        tokens = raw_caption.split()

        # convert word -> ID
        token_ids = [self.w2i.get(t, self.UNK) for t in tokens]

        token_ids = torch.tensor(token_ids, dtype=torch.long)
        
        return img_tensor, token_ids
