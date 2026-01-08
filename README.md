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

```mermaid
graph TD
    A[Input Image] --> B[Conv + BN + ReLU]
    B --> C[Residual Bottleneck Blocks]
    C --> D[Global Average Pooling]
    D --> E[Fully Connected Layer]
    E --> F[Class Probabilities]
