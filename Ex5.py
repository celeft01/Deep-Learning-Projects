import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import MeanSquaredError
import tensorflow as tf
from sklearn.decomposition import PCA
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import mnist
from sklearn.metrics import mean_squared_error
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"



# Load MNIST dataset
(x_train, _), (x_test, _) = mnist.load_data()
x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0

# Flatten the images to 784-dimensional vectors
x_train = x_train.reshape(-1, 784)
x_test = x_test.reshape(-1, 784)


#--------------------- Exercise 1: Training an Autoencoder --------------------------------------------------------

# Encoder
input_img = Input(shape=(784,))
encoded = Dense(128, activation='relu')(input_img)
encoded = Dense(64, activation='relu')(encoded)
encoded = Dense(32, activation='relu')(encoded)  # Latent space with 32 neurons

# Decoder
decoded = Dense(64, activation='relu')(encoded)
decoded = Dense(128, activation='relu')(decoded)
decoded = Dense(784, activation='sigmoid')(decoded)  # Output layer for reconstruction

# Autoencoder Model
autoencoder = Model(input_img, decoded)
autoencoder.compile(optimizer=Adam(), loss=MeanSquaredError(), metrics=['accuracy'])
autoencoder.summary()

# Train the autoencoder
history = autoencoder.fit(
    x_train, x_train,
    epochs=50,
    batch_size=256,
    shuffle=True,
    validation_data=(x_test, x_test)
)

# Plot training & validation loss values
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend(loc='upper right')
plt.show()

# Predict reconstructed images
reconstructed_imgs = autoencoder.predict(x_test)

# Plot original and PCA-reconstructed images
n = 10  # Number of images to display
plt.figure(figsize=(20, 4))
plt.suptitle("Comparison of Original and Autoencoder Reconstructed Images")
for i in range(n):
    # Display original images
    ax = plt.subplot(2, n, i + 1)
    plt.imshow(x_test[i].reshape(28, 28), cmap='gray')
    plt.title("Original")
    plt.axis('off')

    # Display PCA reconstructed images
    ax = plt.subplot(2, n, i + 1 + n)
    plt.imshow(reconstructed_imgs[i].reshape(28, 28), cmap='gray')
    plt.title("Reconstructed")
    plt.axis('off')
plt.show()

# Calculate the Mean Squared Error for Autoencoder reconstruction
autoencoder_reconstruction_error = mean_squared_error(x_test, reconstructed_imgs)
print("Autoencoder Reconstruction Error:", autoencoder_reconstruction_error)


# Add Gaussian noise
noise_factor = 0.5
x_test_noisy = x_test + noise_factor * np.random.normal(loc=0.0, scale=1.0, size=x_test.shape)
x_test_noisy = np.clip(x_test_noisy, 0., 1.)

# Predict denoised images
denoised_imgs = autoencoder.predict(x_test_noisy)

# Plot original, noisy, and denoised images
plt.figure(figsize=(20, 6))
plt.suptitle("Comparison of Noisy, Original, and Autoencoder Noisy Reconstructed Images")
for i in range(n):
    # Display noisy images
    ax = plt.subplot(3, n, i + 1)
    plt.imshow(x_test_noisy[i].reshape(28, 28), cmap='gray')
    plt.title("Noisy")
    plt.axis('off')

    # Display original images
    ax = plt.subplot(3, n, i + 1 + n)
    plt.imshow(x_test[i].reshape(28, 28), cmap='gray')
    plt.title("Original")
    plt.axis('off')

    # Display denoised images
    ax = plt.subplot(3, n, i + 1 + 2 * n)
    plt.imshow(denoised_imgs[i].reshape(28, 28), cmap='gray')
    plt.title("Denoised")
    plt.axis('off')
plt.show()

# Calculate the Mean Squared Error for Autoencoder Noisy reconstruction
autoencoder_noisy_reconstruction_error = mean_squared_error(x_test, denoised_imgs)
print("Autoencoder Noisy Reconstruction Error:", autoencoder_noisy_reconstruction_error)

#----------------------------------------------------------------------------------------------------------------------------




#--------------------- Exercise 2: Comparison with PCA --------------------------------------------------------
# Initialize PCA with 32 components
pca = PCA(n_components=32)

# Fit PCA on training data and transform both train and test data
x_train_pca = pca.fit_transform(x_train)
x_test_pca = pca.transform(x_test)

# Inverse transform the reduced data to reconstruct images
x_test_pca_reconstructed = pca.inverse_transform(x_test_pca)

# Plot original and PCA-reconstructed images
n = 10  # Number of images to display
plt.figure(figsize=(20, 4))
plt.suptitle("Comparison of Original and PCA Reconstructed Images")
for i in range(n):
    # Display original images
    ax = plt.subplot(2, n, i + 1)
    plt.imshow(x_test[i].reshape(28, 28), cmap='gray')
    plt.title("Original")
    plt.axis('off')

    # Display PCA reconstructed images
    ax = plt.subplot(2, n, i + 1 + n)
    plt.imshow(x_test_pca_reconstructed[i].reshape(28, 28), cmap='gray')
    plt.title("Reconstructed")
    plt.axis('off')
plt.show()

# Calculate the Mean Squared Error for PCA-based reconstruction
pca_reconstruction_error = mean_squared_error(x_test, x_test_pca_reconstructed)
print("PCA Reconstruction Error:", pca_reconstruction_error)



# Add Gaussian noise to the test images
noise_factor = 0.5
x_test_noisy = x_test + noise_factor * np.random.normal(loc=0.0, scale=1.0, size=x_test.shape)
x_test_noisy = np.clip(x_test_noisy, 0., 1.)

# Reconstruct noisy images with PCA
x_test_noisy_pca = pca.transform(x_test_noisy)
x_test_noisy_pca_reconstructed = pca.inverse_transform(x_test_noisy_pca)



# Plot original, noisy, and denoised images
plt.figure(figsize=(20, 6))
plt.suptitle("Comparison of Noisy, Original, and PCA Noisy Reconstructed Images")
for i in range(n):
    # Display noisy images
    ax = plt.subplot(3, n, i + 1)
    plt.imshow(x_test_noisy[i].reshape(28, 28), cmap='gray')
    plt.title("Noisy")
    plt.axis('off')

    # Display original images
    ax = plt.subplot(3, n, i + 1 + n)
    plt.imshow(x_test[i].reshape(28, 28), cmap='gray')
    plt.title("Original")
    plt.axis('off')

    # Display denoised images
    ax = plt.subplot(3, n, i + 1 + 2 * n)
    plt.imshow(x_test_noisy_pca_reconstructed[i].reshape(28, 28), cmap='gray')
    plt.title("Denoised")
    plt.axis('off')
plt.show()

# Calculate the Mean Squared Error for PCA-based Noisy reconstruction
pca_noisy_reconstruction_error = mean_squared_error(x_test, x_test_noisy_pca_reconstructed)
print("PCA Noisy Reconstruction Error:", pca_noisy_reconstruction_error)
#----------------------------------------------------------------------------------------------------------------------------