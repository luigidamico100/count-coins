import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
from torchvision import models
from PIL import Image
import os
import numpy as np
from sklearn.model_selection import train_test_split
from skimage.io import imread
from countcoins.data.IO import read_image



def get_transforms(image_size, mode='training'):

    assert mode in ['training', 'inference']

    if mode == 'training':
        # PIL (H, W) -> torch (1, 128, 128)
        transform = transforms.Compose([
            transforms.Resize([image_size, image_size]),
            transforms.ToTensor(),
        ])
    elif mode == 'inference':
        # PIL (H, W) -> torch (1, 128, 128)
        transform = transforms.Compose([
            transforms.Resize([image_size, image_size]),
            transforms.ToTensor(),
        ])

    return transform


class CoinDataset(Dataset):
    def __init__(self, df_annotations, data_path, image_size=128, set_='train', mode='training'):
        assert set_ in ['train', 'test']
        assert mode in ['training', 'inference']

        self.df_annotations = df_annotations.query("set==@set_")
        self.transform = get_transforms(image_size=image_size, mode=mode)
        self.data_path = data_path
        

    def __len__(self):
        return len(self.df_annotations)

    def __getitem__(self, idx):

        currency = self.df_annotations.iloc[idx]['currencies']
        filename = self.df_annotations.index[idx]
        image_path = os.path.join(self.data_path, 'coins_images', currency, filename)
        image = read_image(image_path, as_gray=True, as_PIL=True, plot_image=False,)  # shape: (H, W)
        # assert image.shape[2] == 3

        if self.transform:
            image = self.transform(image)

        target = torch.tensor(self.df_annotations['coins_count'].iloc[idx], dtype=torch.float32)

        return image, target
    