import os
from pathlib import Path
import json
import numpy as np
import torch
import random

project_root_path = os.path.dirname(Path(os.path.abspath(__file__)).parent)
data_path = os.path.join(project_root_path, 'data')
df_annotations_path = os.path.join(data_path, 'coins_count_values.csv')
models_path = os.path.join(project_root_path, 'models')


if torch.cuda.is_available():
    device = torch.device("cuda")
    print('Using CUDA')
elif torch.backends.mps.is_available():
    device = torch.device("mps")
    print('Using MPS')
else:
    device = torch.device("cpu")
    print('Using CPU')


def set_random_seed(random_seed=42):
    random.seed(random_seed)
    
    np.random.seed(random_seed)
    
    # Set the PyTorch random seed for CPU and CUDA
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed)  # If using multi-GPU

    # Ensure deterministic behavior in PyTorch
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False