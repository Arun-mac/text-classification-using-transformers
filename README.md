# Transformer-Based Text Classification

## Overview
This repository contains a TensorFlow implementation of a transformer-based neural network for text classification. The model is designed for sentiment analysis using the IMDB movie reviews dataset.

## Architecture
The architecture consists of custom layers for transformer blocks and positional embedding.

## Dataset
The IMDB movie reviews dataset is used for training and validation. The dataset is preprocessed to limit the vocabulary to the top 20,000 words, and sequences are truncated or padded to a maximum length of 400.

## Model Training
The model is defined with an embedding layer, a transformer block, global average pooling, dropout for regularization, and dense layers with ReLU activation. It uses softmax activation for binary classification.
