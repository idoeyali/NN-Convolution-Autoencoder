# Project Details and Explanation
  ## Modules
  * **Encoder**:  
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
  The main purpose of the **encoder architecture** is to reduce the data size while preserving essential features. 
  The series of convolutional and pooling layers progressively reduces the spatial dimensions (which reduce the data size)  while increasing the number of channels (which help preserve essential features). The Relu layer introduces non-linearity into the model, which helps the network learn more complex representations. 
  Then, The fully connected (FC) layer serves as a bridge to achieve the compression of the input data into a lower-dimensional latent representation.
  * **Decoder**: Then defining decoder:
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
The **decoder architecture** effectively reconstructs the input image by progressively upsampling the latent representation.
The fully connected layer ensures that the compact latent vector is expanded into a suitable high-dimensional representation. 
The subsequent deconvolutional layers then increase the spatial dimensions step-by-step until the original image size is restored.
The use of ReLU activations introduces non-linearity, helping the network learn complex features. Finally, the Sigmoid activation ensures the output is a valid image.
* **Autoencoder**: After defining encoder and decoder we can build an autoencoder:
```python
class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x
```
* **Classifier**: In the following code we can see the Classifier definition. This class is capable to preform fine-tuning getting pre-traind encoder.
```python
class Classifier(nn.Module):
    def __init__(self, latent_dim=12, num_classes=10, tuning_encoder=None):
        super(Classifier, self).__init__()
        if tuning_encoder:
            self.encoder = tuning_encoder
        else:
            self.encoder = Encoder(latent_dim)
        self.fc1 = nn.Linear(latent_dim, 64)
        self.fc2 = nn.Linear(64, num_classes)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.encoder(x)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x
```
