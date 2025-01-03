# Count coins project

This is a little project made for the aulab classes.


## Description
The aim of the project is to show project data science project structure folders for a common data science project taken from [Cookiecutter](https://github.com/drivendataorg/cookiecutter-data-science). 

Beside this, the project is about Computer vision, specifically the task is to count the number of coins present in images. The dataset is the [Count Coins Image dataset](https://www.kaggle.com/datasets/balabaskar/count-coins-image-dataset) from Kaggle. 

In particular the task is solved in two different way.

- **Model based approach**: A classic computer vision algorithm is developed using skimage library.

- **Data driven approach**: Deep learning methods are used, in particular a Convolutional Neural Network is trained using PyTorch library.

## Get started

First, make sure to create a new virtual environment.

Install all the required libraries.

Install the current project source code as a python package. Place in the main directory and run:

    pip install -e .

You are now able to run the notebooks in the folder `notebooks/`.

### Notebooks description

You can perform the main things through the notebooks, since they are a very high-level representation of all the code in the project. They mainly uses the low-level code developed in the python package `countcoins`.

The notebooks are:

- `EDA.ipynb`: Simple data exploration
- `predict_classical_cv.ipynb`: Perform a single prediction using the developed classical computer vision model. 
- `train_deeplearning.ipynb`: Perform the training of a deep learning model and save the experiment in the folder `/models/`.
- `evaluate.ipynb`: Evaluate a model (both the classical and the deep learning one) over all the dataset and finally show the resulting metrics.