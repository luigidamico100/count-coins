import matplotlib.pyplot as plt
import seaborn as sns


def visualize_distributions(df_annotations):

    sns.set_theme()
    fig, axs = plt.subplots(2, 1, figsize=(12, 8))
    axs[0].set_title('Coin count distribution for different currencies')
    sns.histplot(df_annotations, x='coins_count', hue='currencies', bins=30, ax=axs[0])
    axs[1].set_title('Currencies counts for train and test set')
    sns.histplot(df_annotations, x='currencies', hue='set', multiple='dodge', shrink=.8, ax=axs[1])
    plt.tight_layout()
    plt.close()
    plt.style.use('default')
    return fig


def visualize_training_info(df_training_info, plt_close=True):
    
    sns.set_theme()
    fig, axs = plt.subplots(2, 1, figsize=(7, 6))
    axs[0].set_title('MSE Loss')
    sns.lineplot(df_training_info.query("Metric=='MSELoss'"), x='Epoch', y='Value', hue='Set', markers=True, ax=axs[0])
    axs[1].set_title('MAE Loss')
    sns.lineplot(df_training_info.query("Metric=='L1Loss'"), x='Epoch', y='Value', hue='Set', markers=True, ax=axs[1])
    plt.style.use('default')
    plt.tight_layout()

    if plt_close:
        plt.close()
    return fig