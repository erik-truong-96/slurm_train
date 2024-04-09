import os
import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from transformers import DistilBertTokenizer

# # Initialize BERT tokenizer and model
tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')


def get_bert_tokens(text):
    return tokenizer.encode(text, add_special_tokens=True)

def get_padded_bert_tokens(text, max_seq_len):
    return tokenizer.encode(text, return_tensors="pt", padding="max_length", truncation=True, max_length=max_seq_len).squeeze(0)


class Multi_Modal_Dataset_Tensors(Dataset):

    def __init__(self, img_dir, classes=2, df_labels=None, transform=None):

        # make sure directory exists
        if not os.path.exists(img_dir):
            raise ValueError(f"{img_dir} does not exist")


        if transform is None:
            raise ValueError("transform of image to tensor must be provided")

        # Argument validations
        if df_labels is None:
            raise ValueError("dataframe with labels must be provided")
        if classes not in [2, 3, 6]:
            raise ValueError("classes must be 2, 3, or 6")

        # transform to apply to images
        self.transform = transform

        # get images in a directory
        self.image_paths = [os.path.join(img_dir, file_name) for file_name in os.listdir(img_dir)]


        # save tokens for each text sample
        # save max sequence length
        #self.token_map = {}
        self.max_seq_len = 0
        for i in range(len(self.image_paths)):
            id = os.path.basename(self.image_paths[i]).strip().split(".")[0]
            text = df_labels.loc[id, "clean_title"]
            tokens = get_bert_tokens(text)
            #self.token_map[id] = tokens
            self.max_seq_len = max(self.max_seq_len, len(tokens))
           # print(f"Creating word to token mapping for data: {i+1}/{len(self.image_paths)}", end="\r")
        #self.max_seq_len = np.floor(1.5*self.max_seq_len).astype(int)


        # dataframe to fetch labels
        self.df_labels = df_labels

        # keep track of distribution of classes
        self.n_way_label = "2_way_label"
        if classes == 3:
            self.n_way_label = "3_way_label"
        elif classes == 6:
            self.n_way_label = "6_way_label"

        # process images
        # self.images = [
        #         self.transform(Image.open(image_path).convert("RGB"))
        #         for image_path in self.image_paths
        #     ]
    
    def get_class_label(self, image_path):
        id = os.path.basename(image_path).strip().split(".")[0]
        label = self.df_labels.loc[id, self.n_way_label]
        return torch.tensor(label, dtype=torch.long)

    def get_tokens(self, image_path):
        id = os.path.basename(image_path).strip().split(".")[0]
        text = self.df_labels.loc[id, "clean_title"]
        return get_padded_bert_tokens(text, self.max_seq_len)

    def get_images(self, image_path):
        x = Image.open(image_path).convert("RGB")
        return self.transform(x)


    def __getitem__(self, index):
        image_path = self.image_paths[index]
        #images = self.images[index]
        images = self.get_images(image_path)
        tokens = self.get_tokens(image_path)
        labels = self.get_class_label(image_path)
        return tokens, images, labels
 
    def __len__(self):
        return len(self.image_paths)

    
if __name__ == "__main__":
    pass