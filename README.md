# Landmark Detection using Google Landmark Detection Dataset (GLDv2)

## Project Overview

This project involves building a deep learning model for landmark detection using the Google Landmark Detection Dataset (GLDv2). The dataset contains over 5 million images with more than 200,000 unique landmarks. The model is trained to accurately identify and categorize landmarks across diverse images.

## Table of Contents

- [Dataset](#dataset)
- [Download Dataset](#download-dataset)
- [Extracting the data](#extracting-the-data)
- [Model Architecture](#model-architecture)
- [Training](#training)
- [Evaluation](#evaluation)
- [Installation](#installation)

## Dataset

The Google Landmark Detection Dataset (GLDv2) is a large-scale dataset with:
- **5 million+ images**
- **200,000+ unique landmarks**

This dataset is a benchmark for large-scale image retrieval and classification, providing a rich and diverse set of landmarks from around the world.

## Download Dataset

## Training list

-   `train.csv`: CSV with id,url,landmark_id fields. `id` is a 16-character
    string, `url` is a string, `landmark_id` is an integer. Available at:
    [`https://s3.amazonaws.com/google-landmark/metadata/train.csv`](https://s3.amazonaws.com/google-landmark/metadata/train.csv).

## Training data

The `train` set is split into 500 TAR files (each of size ~1GB) containing
JPG-encoded images. The files are located in the `train/` directory, and are
named `images_000.tar`, `images_001.tar`, ..., `images_499.tar`. To download
them, access the following link:

[`https://s3.amazonaws.com/google-landmark/train/images_000.tar`](https://s3.amazonaws.com/google-landmark/train/images_000.tar)

And similarly for the other files.

## Test dataset

-   `test.csv`: single-column CSV with id field. `id` is a 16-character string.
    Available at:
    [`https://s3.amazonaws.com/google-landmark/metadata/test.csv`](https://s3.amazonaws.com/google-landmark/metadata/test.csv).

## Test data

The `test` set is split into 20 TAR files (each of size ~500MB) containing
JPG-encoded images. The files are located in the `test/` directory, and are
named `images_000.tar`, `images_001.tar`, ..., `images_019.tar`. To download
them, access the following link:

[`https://s3.amazonaws.com/google-landmark/test/images_000.tar`](https://s3.amazonaws.com/google-landmark/test/images_000.tar)

And similarly for the other files.

## Extracting the data

We recommend that the set of TAR files corresponding to each dataset split be
extracted into a directory per split; ie, the `index` TARs extracted into an
`index` directory; `train` TARs extracted into a `train` directory; `test` TARs
extracted into a `test` directory. This is done automatically if you use the
above download instructions/script.

The directory structure of the image data is as follows: Each image is stored in
a directory `${a}`/`${b}`/`${c}`/`${id}`.jpg, where `${a}`, `${b}` and `${c}`
are the first three letters of the image id, and `${id}` is the image id found
in the csv files. For example, an image with the id `0123456789abcdef` would be
stored in `0/1/2/0123456789abcdef.jpg`.

## Model Architecture

The model is built using state-of-the-art deep learning techniques. The architecture includes:
- **Convolutional Neural Networks (CNNs)** for feature extraction.
- **Attention mechanisms** to focus on distinctive parts of the landmarks.
- **Transfer Learning** using pre-trained models to leverage existing knowledge.

## Training

The model was trained using the following techniques:
- **Data Augmentation** to increase the variety of images and prevent overfitting.
- **Batch Normalization** for faster convergence and stable training.
- **Optimizer**: Adam optimizer was used with a learning rate scheduler.
- **Loss Function**: Categorical cross-entropy for classification tasks.

## Evaluation

The model's performance was evaluated using:
- **Accuracy**: Overall accuracy on the validation set.
- **Precision, Recall, and F1-Score**: For detailed analysis of model performance.
- **Confusion Matrix**: To visualize the classification results and errors.


## Installation

To run this project locally, follow these steps:

1. Clone the repository:
    ```bash
     git clone https://github.com/vitthalnamdev/Landmark-Detection.git
    ```
