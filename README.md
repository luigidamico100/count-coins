# Count coins project

This is a little project made for the aulab classes.


## Description  
This project aims to demonstrate the folder structure for a typical data science project, following the framework provided by [Cookiecutter](https://github.com/drivendataorg/cookiecutter-data-science).  

In addition, the project focuses on a computer vision task—specifically, counting the number of coins in images. The dataset used is the [Count Coins Image dataset](https://www.kaggle.com/datasets/balabaskar/count-coins-image-dataset) available on Kaggle.  

The task is approached using two distinct methods:  

- **Model-Based Approach**: A traditional computer vision algorithm is implemented utilizing the `skimage` library.  
- **Data-Driven Approach**: Deep learning techniques are applied by training a Convolutional Neural Network (CNN) with the PyTorch library.  



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