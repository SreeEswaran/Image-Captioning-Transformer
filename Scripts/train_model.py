import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertModel, BertConfig
from transformers import AdamW
import os
from tqdm import tqdm

class ImageCaptionDataset(Dataset):
    def __init__(self, captions_file, img_folder, tokenizer, max_length):
        self.captions_df = pd.read_csv(captions_file)
        self.img_folder = img_folder
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.captions_df)
    
    def __getitem__(self, idx):
        caption = self.captions_df.iloc[idx]['caption']
        img_id = self.captions_df.iloc[idx]['image_id']
        img_path = os.path.join(self.img_folder, img_id.replace('.jpg', '.pt'))
        image = torch.load(img_path)
        inputs = self.tokenizer.encode_plus(
            caption,
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            return_tensors='pt',
            truncation=True
        )
        return {'image': image, 'input_ids': inputs['input_ids'].squeeze(), 'attention_mask': inputs['attention_mask'].squeeze()}

def train_model(train_file, val_file, img_folder, model_dir, epochs=3, batch_size=16):
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    config = BertConfig.from_pretrained('bert-base-uncased')
    config.is_decoder = True
    model = BertModel.from_pretrained('bert-base-uncased', config=config)
    
    train_dataset = ImageCaptionDataset(train_file, os.path.join(img_folder, 'train_images'), tokenizer, max_length=64)
    val_dataset = ImageCaptionDataset(val_file, os.path.join(img_folder, 'val_images'), tokenizer, max_length=64)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    optimizer = AdamW(model.parameters(), lr=5e-5)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    
    model.train()
    for epoch in range(epochs):
        for batch in tqdm(train_loader):
            optimizer.zero_grad()
            images = batch['image'].to(device)
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, encoder_hidden_states=images)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
        
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in val_loader:
                images = batch['image'].to(device)
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                outputs = model(input_ids=input_ids, attention_mask=attention_mask, encoder_hidden_states=images)
                val_loss += outputs.loss.item()
        val_loss /= len(val_loader)
        print(f'Epoch {epoch + 1}, Validation Loss: {val_loss}')
    
    model.save_pretrained(model_dir)
    tokenizer.save_pretrained(model_dir)

if __name__ == "__main__":
    train_model('data/processed/train.csv', 'data/processed/val.csv', 'data/processed', 'models/transformer_model')
