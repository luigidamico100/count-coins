# Count coins project

This is a little project made for Aulab class.


# Description  
This project aims to demonstrate the folder structure for a typical data science project, following the framework provided by [Cookiecutter](https://github.com/drivendataorg/cookiecutter-data-science).  

The project focuses on a computer vision task- Specifically, counting the number of coins in images. The dataset used is the [Count Coins Image dataset](https://www.kaggle.com/datasets/balabaskar/count-coins-image-dataset) available on Kaggle.  

The task is approached using two distinct methods:  

- **Model-Based Approach**: A traditional computer vision algorithm is implemented using the `skimage` library.  
- **Data-Driven Approach**: Deep learning techniques are applied by training a Convolutional Neural Network (CNN) with `PyTorch` library.  



# Get started

## Download the dataset
Download the dataset from [kaggle](https://www.kaggle.com/datasets/balabaskar/count-coins-image-dataset) and extract the content in the folder  `/data/`. The data folder should be structured like this:

```
├── data
    ├── coins_count_values.csv
    ├── coins_images
        ├── all_coins
        ├── china_coins
        ├── ...

```

## Prepare the environment

First, make sure to create a new virtual environment, using `conda` or `penv`.

Install all the required libraries using

    pip install -r requirements.txt

Install the current project source code as a python package. Place in the main directory and run:

    pip install -e .

You are now able to run the notebooks in the folder `/notebooks/`.

## Notebooks description

You can perform the main things through the notebooks, since they are a very high-level representation of all the code in the project. They mainly uses the low-level code developed in the python package `countcoins`.

The notebooks are:

- `EDA.ipynb`: Simple data exploration
- `predict_classical_cv.ipynb`: Perform a single prediction using the developed classical computer vision model. 
- `train_deeplearning.ipynb`: Perform the training of a deep learning model and save the experiment in the folder `/models/`.
- `evaluate.ipynb`: Evaluate a model (both the classical and the deep learning one) over all the dataset and finally shows the resulting metrics.


## Project structure


```

├── data                        <- Folder containing all the data
│
├── models                      <- Contains the deep learning experiments
│
├── notebooks                   <- Jupyter notebooks
│
├── requirements.txt            <- Python package requirements
│
└── countcoins                  <- Source code for use in this project.
    │
    ├── config.py               <- Store useful variables and configuration
    │
    ├── visualization.py        <- Visualization functions
    │
    ├── data                    <- functions to deal with data
    │   │
    │   ├── IO.py               <- Input/Output functions
    │   ├── dataset.py          <- dataset functionalities for pytorch
    │   ├── preprocess.py       <- preprocessing functionalities
    │
    └── modelling               <- functions for the models
        │
        ├── classical_cv.py     <- classical computer vision algorithm functions
        ├── deep_learning.py    <- Functions for the deep learning model
        └── evaluating.py       <- Functions for evaluating generic models

        

```
