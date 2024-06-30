# NN-Convolution-Autoencoder
This project implements a deep learning pipeline using PyTorch for the **MNIST** dataset, focusing on both autoencoder-based feature extraction and classification tasks. The pipeline includes the following components:

* Encoder: Convolutional neural network (CNN) that compresses input images into a lower-dimensional latent space.
* Decoder: CNN that reconstructs images from the latent space representation.
* Autoencoder: Combination of the Encoder and Decoder for unsupervised learning to reconstruct input images.
* Classifier: CNN that uses the pre-trained Encoder for feature extraction and a multi-layer perceptron (MLP) for digit classification.
* ClassifierDecoding: Network combining a pre-trained encoder and a decoder for fine-tuning with a reconstruction-based approach.
# Key Features
* **Autoencoder**: Trained to minimize reconstruction loss, demonstrating effective compression and reconstruction of MNIST digits.

  First defining Encoder class:
  ```python
  class Encoder(nn.Module):
    def __init__(self, latent_dim=12):
        super(Encoder, self).__init__()
        self.latent_dim = latent_dim
        self.conv1 = nn.Conv2d(1, 16, kernel_size=5, stride=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1)
        self.fc = nn.Linear(64 * 3 * 3, latent_dim)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = nn.MaxPool2d(kernel_size=2, stride=2)(x)
        x = self.relu(self.conv2(x))
        x = nn.MaxPool2d(kernel_size=2, stride=2)(x)
        x = self.relu(self.conv3(x))
        x = torch.flatten(x, 1)  # Flatten
        x = self.fc(x)
        return x
      
  ```
  The main purpose of this **architecture** is to reduce the data size while preserving essential features. The series of convolutional and pooling layers progressively reduces the spatial dimensions (which reduce the data size)  while increasing the number of channels (which help preserve essential features). The Relu layer introduces non-linearity into the model, which helps the network learn more complex representations. Then, The fully connected (FC) layer serves as a bridge to achieve the compression of the input data into a lower-dimensional latent representation.

  Then defining decoder:
```python
class Decoder(nn.Module):
    def __init__(self, latent_dim=12):
        super(Decoder, self).__init__()
        self.latent_dim = latent_dim
        self.fc = nn.Linear(latent_dim, 64 * 3 * 3)
        self.deconv1 = nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.deconv2 = nn.ConvTranspose2d(32, 16, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.deconv3 = nn.ConvTranspose2d(16, 1, kernel_size=5, stride=2, output_padding=1)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.fc(x)
        x = x.view(-1, 64, 3, 3)  # Reshape
        x = self.relu(self.deconv1(x))
        x = self.relu(self.deconv2(x))
        x = self.sigmoid(self.deconv3(x))
        return x
```
The decoder architecture effectively reconstructs the input image by progressively upsampling the latent representation.

* **Classifier**: Trained to predict digit classes using cross-entropy loss, incorporating transfer learning for improved performance with limited labeled data.
* **Fine-tuning**: Demonstrates the use of pre-trained encoders for improved classification accuracy with a small labeled dataset.
* **Visualization**: Plots training and test losses, accuracies, and reconstructed images to evaluate model performance.
* 
