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
import torch
import pandas as pd
from countcoins.data.dataset import get_transforms
import time
from countcoins.data.IO import read_image
import pickle


def number_of_parameters(model):
    total_params = sum(p.numel() for p in model.parameters())
    return total_params


def model_memory_mb(model):
    param_size = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    buffer_size = 0
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()

    size_all_mb = (param_size + buffer_size) / (1024**2)

    return size_all_mb


class CoinCounterCNN(nn.Module):
    def __init__(self, initial_channels=1, reduce_factor=1, image_size=128):
        super(CoinCounterCNN, self).__init__()

        assert 16 % reduce_factor == 0
        assert image_size % 8 == 0
        base_dim = 16 // reduce_factor

        self.conv_layers = nn.Sequential(
            nn.Conv2d(initial_channels, base_dim, kernel_size=3, padding=1),  # Conv1
            nn.BatchNorm2d(base_dim),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            # (4, 64, 64)

            nn.Conv2d(base_dim, base_dim*2, kernel_size=3, padding=1),  # Conv2
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            # (8, 32, 32)

            nn.Conv2d(base_dim*2, base_dim*4, kernel_size=3, padding=1),  # Conv3
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
            # (16, 16, 16)
        )

        C, H, W = base_dim*4, image_size // 8, image_size // 8
        self.fc_layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(C*H*W, base_dim*8),
            nn.ReLU(),
            nn.Linear(base_dim*8, 1) 
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = self.fc_layers(x)
        return x
    

def train(model, epochs, optimizer, criterion, additional_criterion, train_loader, test_loader, device):
    training_info_list = []
    criterion.to(device)

    for epoch in range(1, epochs+1):
        time_start = time.time()
        model.train()
        running_loss, running_additional_loss = 0.0, 0.0

        for images, targets in train_loader:
            # images.shape = torch.Size([B, 1, image_size, image_size])
            # targets.shape = torch.suze([B])
            images, targets = images.to(device), targets.to(device)

            outputs = model(images)
            loss = criterion(outputs.squeeze(), targets)
            additional_loss = additional_criterion(outputs.squeeze(), targets)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            running_additional_loss += additional_loss.item()

        epoch_train_loss = running_loss/len(train_loader)
        epoch_train_additional_loss = running_additional_loss/len(train_loader)

        training_info_list.append({
            'Epoch': epoch,
            'Set': 'Train',
            'Metric': type(criterion).__name__,
            'Value': epoch_train_loss,
        })

        training_info_list.append({
            'Epoch': epoch,
            'Set': 'Train',
            'Metric': type(additional_criterion).__name__,
            'Value': epoch_train_additional_loss,
        })

        model.eval()
        with torch.no_grad():
            total_loss, total_additional_loss = 0., 0.
            for images, targets in test_loader:
                images, targets = images.to(device), targets.to(device)
                outputs = model(images)
                loss = criterion(outputs.squeeze(), targets)
                additional_loss = additional_criterion(outputs.squeeze(), targets)
                total_loss += loss.item()
                total_additional_loss += additional_loss.item()

            epoch_test_loss = total_loss / len(test_loader)
            epoch_test_additional_loss = total_additional_loss / len(test_loader)
            # print(f"Test Loss: {epoch_test_loss:.4f}")
            time_end = time.time()
            print(f"Epoch [{epoch}/{epochs}], Train loss: {epoch_train_loss:.2f}, Test loss: {epoch_test_loss:.2f}. Time: {time_end-time_start:.1f}")

            training_info_list.append({
                'Epoch': epoch,
                'Set': 'Test',
                'Metric': type(criterion).__name__,
                'Value': epoch_test_loss,
            })

            training_info_list.append({
                'Epoch': epoch,
                'Set': 'Test',
                'Metric': type(additional_criterion).__name__,
                'Value': epoch_test_additional_loss,
            })

        

    df_training_info = pd.DataFrame(training_info_list)

    return model, df_training_info


class TrainedDeepLearningModel():

    def __init__(self, models_path, experiment_name):
        experiment_folder = os.path.join(models_path, experiment_name)
        with open(os.path.join(experiment_folder, 'params.pkl'), 'rb') as f:
            params = pickle.load(f)

        self.model = CoinCounterCNN(reduce_factor=params['model_reduce_factor'], image_size=params['image_size'])
        self.model.load_state_dict(torch.load(os.path.join(experiment_folder, 'model_state_dict.pth'), weights_only=True))
        self.model.eval()

        self.transform = get_transforms(image_size=params['image_size'], mode='inference')

    def predict(self, image_path):
        image = read_image(image_path, as_gray=True, as_PIL=True, plot_image=False,)  # shape: (H, W)
        image = self.transform(image)
        image = image.unsqueeze(0)  # torch.Size([1, 1, H, W])

        with torch.no_grad():
            y = self.model(image)
            y = y.item()

        return y
