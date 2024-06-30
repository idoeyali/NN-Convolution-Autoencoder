# NN-Convolution-Autoencoder
This project implements a deep learning pipeline using PyTorch for the MNIST dataset, focusing on both autoencoder-based feature extraction and classification tasks. The pipeline includes the following components:

Encoder: Convolutional neural network (CNN) that compresses input images into a lower-dimensional latent space.
Decoder: CNN that reconstructs images from the latent space representation.
Autoencoder: Combination of the Encoder and Decoder for unsupervised learning to reconstruct input images.
Classifier: CNN that uses the pre-trained Encoder for feature extraction and a multi-layer perceptron (MLP) for digit classification.
ClassifierDecoding: Network combining a pre-trained encoder and a decoder for fine-tuning with a reconstruction-based approach.
