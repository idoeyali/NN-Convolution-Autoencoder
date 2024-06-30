import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import torch.utils.data as data_utils


#################################
# Helping functions
#################################
def plot_scores(train_losses, test_losses, train_accuracies, test_accuracies, num_epochs):
    """
    Plots the training and test losses and accuracies over epochs.

    Parameters:
    - train_losses (list): List of training losses for each epoch.
    - test_losses (list): List of test losses for each epoch.
    - train_accuracies (list): List of training accuracies for each epoch.
    - test_accuracies (list): List of test accuracies for each epoch.
    - num_epochs (int): Number of training epochs.
    """
    plt.figure(figsize=(12, 6))
    # Plotting Training and Test Loss
    plt.subplot(1, 2, 1)
    plt.plot(range(1, num_epochs + 1), train_losses, label='Train Loss')
    plt.plot(range(1, num_epochs + 1), test_losses, label='Test Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Training and Test Loss')
    # Plotting Training and Test Accuracy
    plt.subplot(1, 2, 2)
    plt.plot(range(1, num_epochs + 1), train_accuracies, label='Train Accuracy')
    plt.plot(range(1, num_epochs + 1), test_accuracies, label='Test Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.title('Training and Test Accuracy')

    plt.show()


def train_and_test_scores(module, criterion, num_epochs, train_loader, test_loader):
    """
    Trains and evaluates a neural network module, recording and plotting the loss and accuracy over epochs.

    Parameters:
    - module (nn.Module): The neural network module to be trained and evaluated.
    - criterion (nn.Module): The loss function.
    - num_epochs (int): The number of training epochs.
    - train_loader (DataLoader): DataLoader for the training dataset.
    - test_loader (DataLoader): DataLoader for the test dataset.

    Returns:
    - None: This function plots the training and test losses and accuracies.
    """
    optimizer = optim.Adam(module.parameters(), lr=0.001)
    # Training loop
    train_losses = []
    train_accuracies = []
    test_losses = []
    test_accuracies = []

    for epoch in range(num_epochs):
        module.train()
        train_loss = 0
        correct = 0
        total = 0
        for batch_idx, (data, targets) in enumerate(train_loader):
            optimizer.zero_grad()
            outputs = module(data)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

        train_losses.append(train_loss / len(train_loader))
        train_accuracies.append(100. * correct / total)
        print(
            f'Epoch {epoch + 1} - Train Loss: {train_loss / len(train_loader):.4f}, Accuracy: {100. * correct / total:.2f}%')

        module.eval()
        test_loss = 0
        correct = 0
        total = 0
        with torch.no_grad():
            for data, targets in test_loader:
                outputs = module(data)
                loss = criterion(outputs, targets)

                test_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()

        test_losses.append(test_loss / len(test_loader))
        test_accuracies.append(100. * correct / total)
        print(f'Test Loss: {test_loss / len(test_loader):.4f}, Accuracy: {100. * correct / total:.2f}%')
    plot_scores(train_losses, test_losses, train_accuracies, test_accuracies, num_epochs)


#################################
# Data Loading
#################################
transform = transforms.Compose([transforms.ToTensor()])

train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
batch_size = 64
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)


#################################
# Encoder
#################################
class Encoder(nn.Module):
    """
    Convolutional Encoder for extracting latent representations from input images.

    Args:
        latent_dim (int): The dimensionality of the latent space.

    Attributes:
        conv1 (nn.Conv2d): First convolutional layer.
        conv2 (nn.Conv2d): Second convolutional layer.
        conv3 (nn.Conv2d): Third convolutional layer.
        fc (nn.Linear): Fully connected layer for mapping to the latent space.
        relu (nn.ReLU): ReLU activation function.
    """

    def __init__(self, latent_dim=12):
        super(Encoder, self).__init__()
        self.latent_dim = latent_dim
        self.conv1 = nn.Conv2d(1, 16, kernel_size=5, stride=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1)
        self.fc = nn.Linear(64 * 3 * 3, latent_dim)
        self.relu = nn.ReLU()

    def forward(self, x):
        """
        Forward pass through the encoder.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, 1, 28, 28).

        Returns:
            torch.Tensor: Output latent vector of shape (batch_size, latent_dim).
        """
        x = self.relu(self.conv1(x))
        x = nn.MaxPool2d(kernel_size=2, stride=2)(x)
        x = self.relu(self.conv2(x))
        x = nn.MaxPool2d(kernel_size=2, stride=2)(x)
        x = self.relu(self.conv3(x))
        x = torch.flatten(x, 1)  # Flatten
        x = self.fc(x)
        return x


#################################
# Decoder
#################################
class Decoder(nn.Module):
    """
    Convolutional Decoder for reconstructing images from latent representations.

    Args:
        latent_dim (int): The dimensionality of the latent space.

    Attributes:
        fc (nn.Linear): Fully connected layer for mapping from the latent space.
        deconv1 (nn.ConvTranspose2d): First transposed convolutional layer.
        deconv2 (nn.ConvTranspose2d): Second transposed convolutional layer.
        deconv3 (nn.ConvTranspose2d): Third transposed convolutional layer.
        relu (nn.ReLU): ReLU activation function.
        sigmoid (nn.Sigmoid): Sigmoid activation function for the final layer.
    """

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
        """
        Forward pass through the decoder.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, latent_dim).

        Returns:
            torch.Tensor: Reconstructed image tensor of shape (batch_size, 1, 28, 28).
        """
        x = self.fc(x)
        x = x.view(-1, 64, 3, 3)  # Reshape
        x = self.relu(self.deconv1(x))
        x = self.relu(self.deconv2(x))
        x = self.sigmoid(self.deconv3(x))
        return x


#################################
# Autoencoder
#################################
class Autoencoder(nn.Module):
    """
    Convolutional Autoencoder composed of an Encoder and a Decoder.

    Attributes:
        encoder (Encoder): Instance of the Encoder class.
        decoder (Decoder): Instance of the Decoder class.
    """

    def __init__(self):
        super(Autoencoder, self).__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()

    def forward(self, x):
        """
        Defines the forward pass through the Autoencoder.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, 1, 28, 28).

        Returns:
            torch.Tensor: Reconstructed image tensor of shape (batch_size, 1, 28, 28).
        """
        x = self.encoder(x)
        x = self.decoder(x)
        return x


autoencoder = Autoencoder()

#################################
# Training the Autoencoder
#################################
criterion = nn.MSELoss()  # Change to MSE loss for reconstruction
optimizer = optim.Adam(autoencoder.parameters(), lr=0.001)  # Decrease learning rate

num_epochs = 10
epochs_loss = []

for epoch in range(num_epochs):
    autoencoder.train()
    train_loss = 0
    for batch_idx, (data, _) in enumerate(train_loader):
        optimizer.zero_grad()
        reconstructions = autoencoder(data)
        loss = criterion(reconstructions, data)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()

    epoch_loss = train_loss / len(train_loader)
    epochs_loss.append(epoch_loss)
    print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {epoch_loss:.4f}')

# After training, save the Autoencoder
torch.save(autoencoder.state_dict(), 'autoencoder.pth')
print("Autoencoder saved.")

#################################
# Testing the Autoencoder
#################################
autoencoder.eval()
with torch.no_grad():
    for data, _ in test_loader:
        reconstructions = autoencoder(data)
        break

#################################
# Display original and reconstructed images using Autoencoder
#################################
n = 10
plt.figure(figsize=(20, 4))
for i in range(n):
    # Display original
    ax = plt.subplot(2, n, i + 1)
    plt.imshow(data[i].squeeze().numpy(), cmap='gray')
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    # Display reconstruction
    ax = plt.subplot(2, n, i + 1 + n)
    plt.imshow(reconstructions[i].squeeze().numpy(), cmap='gray')
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
plt.show()


#################################
# Classifier definition
#################################
class Classifier(nn.Module):
    """
    Classifier network composed of an optional pre-trained encoder and MLP layers for classification.

    Attributes:
        encoder (Encoder or None): Instance of the Encoder class for feature extraction. If None, a new Encoder will be instantiated.
        fc1 (nn.Linear): Fully connected layer 1 for feature mapping.
        fc2 (nn.Linear): Fully connected layer 2 for class prediction.
        relu (nn.ReLU): ReLU activation function.
    """

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
        """
        Defines the forward pass through the Classifier.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, 1, 28, 28).

        Returns:
            torch.Tensor: Predicted class scores tensor of shape (batch_size, num_classes).
        """
        x = self.encoder(x)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x


# Instantiate the classifier
classifier = Classifier()

#################################
# Training and testing the Classifier and plot the result
#################################
train_and_test_scores(module=classifier, criterion=nn.CrossEntropyLoss(), num_epochs=10, train_loader=train_loader,
                      test_loader=test_loader)

# After training, save the classifier
torch.save(classifier.state_dict(), 'classifier.pth')
print("Classifier saved.")

#################################
# Load the Pre-trained Classifier
#################################
# Instantiate the classifier
classifier = Classifier()

# Load the pre-trained classifier
classifier.load_state_dict(torch.load('classifier.pth'))
print("Classifier loaded.")

# Set the encoder to non-trainable
for param in classifier.encoder.parameters():
    param.requires_grad = False


#################################
# ClassifierDecoding definition
#################################
class ClassifierDecoding(nn.Module):
    """
    Classifier with Decoder network for reconstruction-based fine-tuning.

    Attributes:
        encoder (Encoder): Pre-trained Encoder instance for feature extraction.
        decoder (Decoder): Decoder instance for reconstructing input images.
    """

    def __init__(self, encoder, decoder):
        super(ClassifierDecoding, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, x):
        """
        Defines the forward pass through the ClassifierDecoding.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, 1, 28, 28).

        Returns:
            torch.Tensor: Reconstructed images tensor of shape (batch_size, 1, 28, 28).
        """
        x = self.encoder(x)
        x = self.decoder(x)
        return x


#################################
# Training the ClassifierDecoding
#################################
# Instantiate the decoder and the combined model
decoder = Decoder()
autoencoder_with_fixed_encoder = ClassifierDecoding(classifier.encoder, decoder)

# Define the loss function and the optimizer
criterion = nn.MSELoss()
optimizer = optim.Adam(decoder.parameters(), lr=0.001)

# Training loop
num_epochs = 10
epochs_loss = []
for epoch in range(num_epochs):
    autoencoder_with_fixed_encoder.train()
    train_loss = 0
    for batch_idx, (data, _) in enumerate(train_loader):
        optimizer.zero_grad()
        outputs = autoencoder_with_fixed_encoder(data)
        loss = criterion(outputs, data)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()

    epoch_loss = train_loss / len(train_loader)
    epochs_loss.append(epoch_loss)
    print(f'Epoch {epoch + 1} loss: {epoch_loss:.4f}')

#################################
# Testing the ClassifierDecoding
#################################
# Testing and visualization
autoencoder_with_fixed_encoder.eval()
test_loss = 0.0
flag = True
with torch.no_grad():
    for data, _ in test_loader:
        outputs = autoencoder_with_fixed_encoder(data)
        test_loss += criterion(outputs, data).item()
        if flag:
            pre_reconstruction = data
            reconstructed = outputs
            flag = False

test_loss /= len(test_loader)
print(f'Test Loss: {test_loss:.4f}')

#################################
# Display original and reconstructed 50 images using classifier-based autoencoder
#################################
# Display an array of reconstructed digits
n = 50
plt.figure(figsize=(20, 8))
for i in range(n):
    # Display original
    ax = plt.subplot(5, n // 5, i + 1)
    plt.imshow(pre_reconstruction[i].squeeze().numpy(), cmap='gray')
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

plt.figure(figsize=(20, 8))
for i in range(n):
    # Display reconstruction
    ax = plt.subplot(5, n // 5, i + 1)
    plt.imshow(reconstructed[i].squeeze().numpy(), cmap='gray')
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
plt.show()

#################################
# Define a subset of 100 sample from training set and training the Classifier
#################################
indices = torch.arange(100)
train_loader_CLS = data_utils.Subset(train_dataset, indices)
train_loader_CLS = torch.utils.data.DataLoader(train_loader_CLS, batch_size=batch_size, shuffle=True, num_workers=0)

classifier = Classifier()
# Training the Classifier
train_and_test_scores(module=classifier, criterion=nn.CrossEntropyLoss(), num_epochs=50, train_loader=train_loader_CLS,
                      test_loader=test_loader)

#################################
# Training pre-trained encoder Classifier over 100 sample from training set
#################################
# Load the pre-trained autoencoder and extract the encoder
autoencoder.load_state_dict(torch.load('autoencoder.pth'))
pretrained_encoder = autoencoder.encoder

# Instantiate the classifier with the pre-trained encoder
pretrained_encoder_classifier = Classifier(tuning_encoder=pretrained_encoder)

# Training the Classifier with Fine-tuning
train_and_test_scores(module=pretrained_encoder_classifier, criterion=nn.CrossEntropyLoss(), num_epochs=50,
                      train_loader=train_loader_CLS, test_loader=test_loader)
