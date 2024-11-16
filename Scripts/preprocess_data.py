import os
import pandas as pd
import torch
from torchvision import transforms
from PIL import Image
from sklearn.model_selection import train_test_split

def preprocess_data(input_dir, captions_file, output_dir):
    df = pd.read_csv(captions_file)
    train_df, val_test_df = train_test_split(df, test_size=0.2, random_state=42)
    val_df, test_df = train_test_split(val_test_df, test_size=0.5, random_state=42)
    
    os.makedirs(output_dir, exist_ok=True)
    train_df.to_csv(os.path.join(output_dir, 'train.csv'), index=False)
    val_df.to_csv(os.path.join(output_dir, 'val.csv'), index=False)
    test_df.to_csv(os.path.join(output_dir, 'test.csv'), index=False)
    
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])
    
    def save_images(df, folder):
        os.makedirs(folder, exist_ok=True)
        for _, row in df.iterrows():
            img_path = os.path.join(input_dir, row['image_id'])
            image = Image.open(img_path).convert('RGB')
            image = transform(image)
            torch.save(image, os.path.join(folder, row['image_id'].replace('.jpg','.pt')))
    
    save_images(train_df, os.path.join(output_dir, 'train_images'))
    save_images(val_df, os.path.join(output_dir, 'val_images'))
    save_images(test_df, os.path.join(output_dir, 'test_images'))

if __name__ == "__main__":
    preprocess_data('data/raw/images', 'data/captions.csv', 'data/processed')