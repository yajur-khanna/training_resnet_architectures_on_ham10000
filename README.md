# ResNet-50 Image Classification (PyTorch)

A deep learning–based image classification project implementing and fine-tuning a **ResNet-50 convolutional neural network** using PyTorch.  
The project focuses on understanding **residual learning, training dynamics of deep CNNs, and end-to-end ML pipelines**, rather than treating the architecture as a black-box model.

---

## Overview

This project implements a deep convolutional neural network based on **ResNet-50**, designed to address optimization challenges in very deep networks using **residual connections**.

The pipeline covers the complete workflow:

- Data extraction and preprocessing (ETL)
- Model implementation and fine-tuning
- Training and evaluation
- Performance analysis

---

## Features -

- Implementation of **ResNet-50** using PyTorch
- End-to-end ETL pipeline for image datasets
- Image normalization using ImageNet statistics
- Training and validation performance tracking
- Modular structure for extending to other datasets

---

## Architecture -

### Model Overview

![ResNet-50 Architecture Overview](./images/ResNet50.png)

---

## Model Architecture Summary

The model is based on the **ResNet-50** architecture, which uses deep residual learning to enable stable optimization of very deep convolutional networks.

| Component | Description |
|---------|-------------|
| Backbone | ResNet-50 |
| Block Type | Bottleneck residual blocks |
| Convolutions | 1×1 → 3×3 → 1×1 |
| Skip Connections | Identity / projection shortcuts |
| Normalization | Batch Normalization |
| Activation | ReLU |
| Pooling | Adaptive Global Average Pooling |
| Output Layer | Fully Connected classification head |

Residual connections allow the network to learn **residual mappings** instead of direct mappings, significantly reducing degradation and vanishing gradient issues as depth increases.

---

## Dataset Preparation

### ETL Pipeline

The project follows a structured **ETL (Extract, Transform, Load)** pipeline to ensure reproducibility and clean experimentation.

### Extract
- Load image data from kaggle input directory and store it in a pandas dataframe
- Create new columns for full path for each image using image_id and store label encoded class labels

### Transform and Custom Dataset
- Create transform function to convert images to model-compatible resolution (3x224x224)
- Implement the PyTorch `Dataset` interface to enable efficient batching, shuffling, and scalable training via `DataLoader`
- The features passed to the dataset are the full paths to image column, labels are encoded class labels, and the custom transform created above is also passed as a parameter

### Load
- Construct batched DataLoaders for training and validation
- Shuffling enabled for training, disabled for evaluation

This separation allows easy substitution of datasets without modifying model logic.

---

## Model Training

### Loss Function -
- Cross-Entropy Loss for multi-class classification

### Optimization -
- Optimizer: Adam / SGD (configurable)
- Optional learning rate scheduling

### Training Configuration -

- Batch size: Configurable
- Epochs: Configurable
- Device: GPU (CUDA supported)

Training and validation metrics are tracked per epoch to monitor convergence and detect overfitting.

---

## Evaluation

Model performance is evaluated using:

- Classification accuracy
- Training vs validation loss trends
- Epoch-wise performance analysis

These metrics help diagnose optimization issues and generalization behavior.

---
