# Deep Learning Projects

This repository showcases a collection of deep learning projects, each illustrating unique aspects of neural network architectures and applications. Explore these projects to see practical implementations of deep learning techniques across diverse tasks.

---

## Table of Contents

1. [Project 1: Fashion MNIST Classification with Residual Networks](#project-1-fashion-mnist-classification-with-residual-networks)
2. [Project 2: Sentiment Analysis and Time Series Prediction with RNNs and LSTMs](#project-2-sentiment-analysis-and-time-series-prediction-with-rnns-and-lstms)
3. [Project 3: VGG16 Implementation and Fine-Tuning on CIFAR-100](#project-3-vgg16-implementation-and-fine-tuning-on-cifar-100)
4. [Project 4: Autoencoders and PCA on MNIST Dataset](#project-4-autoencoders-and-pca-on-mnist-dataset)

---

## Project 1: Fashion MNIST Classification with Residual Networks

### Overview

This project focuses on classifying images from the Fashion MNIST dataset using convolutional neural networks (CNNs). Residual connections are employed to enhance the performance and stability of the model.

### Highlights

- **Dataset**: Fashion MNIST, consisting of 10 classes of grayscale images.
- **Architecture**:
  - Initial model: Standard CNN with multiple convolutional layers.
  - Enhanced model: CNN with residual connections for improved gradient flow and reduced training loss.
- **Results**:
  - Initial Model Accuracy: 91%.
  - Residual Model Accuracy: 92%.
- **Visualization**:
  - Training Accuracy and Loss Curves.
  - Confusion Matrix.
  - ROC Curves with AUC-ROC Score: 0.99.

### Files

- `Ex2/Ex2.py`: Python script implementing the models.
- `Ex2/Ex2.pdf`: Report with detailed analysis and visuals.

---

## Project 2: Sentiment Analysis and Time Series Prediction with RNNs and LSTMs

### Overview

This project applies Recurrent Neural Networks (RNNs) and Long Short-Term Memory (LSTM) networks for sentiment analysis and time series prediction.

### Highlights

- **Part 1: Sentiment Analysis**
  - Dataset: IMDB movie reviews labeled as positive or negative.
  - Model: RNN for sentiment classification.
- **Part 2: Time Series Prediction**
  - Dataset: Reuters Newswire topics sequence.
  - Model: LSTM for predicting sequential data trends.

### Files

- `Ex3/Ex3.py`: Python script for RNN and LSTM implementations.
- `Ex3/Ex3.pdf`: Report detailing methodology and results.

---

## Project 3: VGG16 Implementation and Fine-Tuning on CIFAR-100

### Overview

Train and fine-tune VGG16 models on the CIFAR-100 dataset to explore transfer learning and model optimization.

### Highlights

- **Dataset**: CIFAR-100, containing 100 classes of color images.
- **Approach**:
  - Train a VGG16 model from scratch on CIFAR-100.
  - Fine-tune a pre-trained VGG16 model with Early Stopping to prevent overfitting.
- **Results**:
  - Improved classification accuracy with fine-tuning.

### Files

- `Ex4/Ex4.py`: Python script for VGG16 training and fine-tuning.
- `Ex4/Ex4.pdf`: Report analyzing the training process and outcomes.

---

## Project 4: Autoencoders and PCA on MNIST Dataset

### Overview

This project compares the performance of autoencoders and Principal Component Analysis (PCA) for dimensionality reduction on the MNIST dataset.

### Highlights

- **Dataset**: MNIST, consisting of handwritten digit images.
- **Approach**:
  - Train an autoencoder to learn efficient data representations.
  - Evaluate PCA for dimensionality reduction and reconstruction.
- **Results**:
  - Insights into reconstruction quality and representation efficiency.

### Files

- `Ex5/Ex5.py`: Python script implementing autoencoder and PCA comparisons.
- `Ex5/Ex5.pdf`: Report with evaluation metrics and visualizations.

---

## Prerequisites

- Python 3.8 or higher
- Required Python libraries:
  - TensorFlow
  - Keras
  - NumPy
  - pandas
  - matplotlib
  - scikit-learn
  - seaborn
