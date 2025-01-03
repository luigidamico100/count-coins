import pandas as pd
import os
from skimage import io
import matplotlib.pyplot as plt
from PIL import Image
import pickle
import torch


def read_df_annotation(csv_path):
    df = pd.read_csv(csv_path, index_col='image_name')
    df = df.rename(columns={'folder': 'currencies'})
    return df


def read_image(image_path, as_gray=False, as_PIL=False, plot_image=False, coins_count=None):
    image = io.imread(image_path, as_gray=as_gray)
    if plot_image:
        plt.imshow(image)
        plt.axis('off')
        if coins_count is not None:
            plt.title(f'Coins count: {coins_count}')
    if as_PIL:
        image = Image.fromarray(image)
    return image


def save_deeplearning_experiment(model, df_training_info, fig_training, models_path, params, experiment_name):
    experiment_folder_path = os.path.join(models_path, experiment_name)
    os.mkdir(experiment_folder_path)
    torch.save(model.state_dict(), os.path.join(experiment_folder_path, 'model_state_dict.pth'))
    torch.save(model, os.path.join(experiment_folder_path, 'model.pth'))
    df_training_info.to_csv(os.path.join(experiment_folder_path, 'df_training_info.csv'))
    fig_training.savefig(os.path.join(experiment_folder_path, 'fig_training.png'))

    with open(os.path.join(experiment_folder_path, 'params.pkl'), 'wb') as f:
        pickle.dump(params, f)

    print(f'Experiment saved in: {experiment_folder_path}')

