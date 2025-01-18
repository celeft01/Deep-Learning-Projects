import tensorflow as tf
from tensorflow.keras.datasets import fashion_mnist
from tensorflow.keras.utils import to_categorical
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.preprocessing import label_binarize
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc, precision_recall_curve, precision_score, recall_score, f1_score, roc_auc_score




# Load the Fashion MNIST dataset
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

# Normalize the pixel values to be between 0 and 1
train_images = train_images.astype('float32') / 255.0
test_images = test_images.astype('float32') / 255.0


# One-hot encode the labels
num_classes = 10
train_labels = to_categorical(train_labels, num_classes)
test_labels = to_categorical(test_labels, num_classes)

# Display the shape of the training and testing data
print("Shape of training images:", train_images.shape)
print("Shape of training labels:", train_labels.shape)
print("Shape of testing images:", test_images.shape)
print("Shape of testing labels:", test_labels.shape)

class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

#---------------------Exercise 1--------------------------------------------------------

# Input Layer: 28x28 grayscale images.
# Convolutional Layer: 32 filters, 3x3 kernel, ReLU activation.
# Max Pooling Layer: 2x2 pooling.
# Convolutional Layer: 64 filters, 3x3 kernel, ReLU activation.
# Max Pooling Layer: 2x2 pooling.
# Flatten Layer: Convert 2D to 1D.
# Fully Connected Layer: 128 units, ReLU activation.
# Output Layer: 10 units (for the 10 classes), softmax activation
# Loss Function: Categorical Crossentropy (since we have multiple classes).
# Optimizer: Adam (popular for fast convergence).
# Metric: Accuracy



# Define the model
model = models.Sequential()

# Add convolutional layers, pooling layers, and fully connected layers
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))

model.add(layers.Flatten())
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(10, activation='softmax'))

# Compile the model
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Train the model using the training data
history = model.fit(train_images, train_labels, epochs=10)

# Evaluate the model using the testing data
test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=2)

print(f"Test Accuracy: {test_acc}")
print(f"Test Loss: {test_loss}")

# Plot training accuracy and loss
plt.figure(figsize=(12, 4))

# Accuracy plot
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.title('Training Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

# Loss plot
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Training Loss')
plt.title('Training Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.show()


# Predict the test data
y_pred = model.predict(test_images)
y_pred_classes = np.argmax(y_pred, axis=1)
test_labels_classes = np.argmax(test_labels, axis=1)

# 1. Confusion Matrix
conf_matrix = confusion_matrix(test_labels_classes, y_pred_classes)
plt.figure(figsize=(10, 8))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
plt.title('Confusion Matrix')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.show()

# 2. Classification Report (Precision, Recall, F1 Score)
print("Classification Report:")
class_report = classification_report(test_labels_classes, y_pred_classes, target_names=class_names)
print(class_report)

# 3. ROC Curve and AUC-ROC
# Binarize the labels for multi-class ROC curve
y_test_binarized = label_binarize(test_labels_classes, classes=np.arange(10))
fpr = {}
tpr = {}
roc_auc = {}

plt.figure(figsize=(10, 8))
for i in range(10):
    fpr[i], tpr[i], _ = roc_curve(y_test_binarized[:, i], y_pred[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])
    plt.plot(fpr[i], tpr[i], label=f'ROC curve for {class_names[i]} (AUC = {roc_auc[i]:0.2f})')

plt.plot([0, 1], [0, 1], 'k--')  # Random guessing line
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curves for Fashion MNIST Classes')
plt.legend(loc="lower right")
plt.show()

# Calculate and display the average AUC-ROC score for all classes
average_roc_auc = roc_auc_score(y_test_binarized, y_pred, multi_class="ovr")
print(f"Average AUC-ROC Score: {average_roc_auc:.2f}")



#---------------------Exercise 2--------------------------------------------------------


# Input layer
input_layer = layers.Input(shape=(28, 28, 1))

# First convolutional layer
conv1 = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(input_layer)
pool1 = layers.MaxPooling2D((2, 2))(conv1)

# Second convolutional layer with a residual connection
conv2 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(pool1)
pool2 = layers.MaxPooling2D((2, 2))(conv2)

# To add a residual connection, we need to match the dimensions:
# We upsample pool1 to match the dimensions of pool2 using a Conv2D layer
residual1 = layers.Conv2D(64, (1, 1), padding='same')(pool1)
residual1 = layers.MaxPooling2D((2, 2))(residual1)  # Match dimensions

# Add the residual connection
residual_output1 = layers.add([pool2, residual1])

# Third convolutional layer with another residual connection
conv3 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(residual_output1)

# Flatten the feature maps and apply fully connected layers
flatten = layers.Flatten()(conv3)
dense1 = layers.Dense(64, activation='relu')(flatten)
output_layer = layers.Dense(10, activation='softmax')(dense1)

# Define the model
model = models.Model(inputs=input_layer, outputs=output_layer)

# Compile the model
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Train the model
history = model.fit(train_images, train_labels, epochs=10)





# Evaluate the model using the testing data
test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=2)

print(f"Test Accuracy: {test_acc}")
print(f"Test Loss: {test_loss}")

# Plot training accuracy and loss
plt.figure(figsize=(12, 4))

# Accuracy plot
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.title('Training Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

# Loss plot
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Training Loss')
plt.title('Training Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.show()


# Predict the test data
y_pred = model.predict(test_images)
y_pred_classes = np.argmax(y_pred, axis=1)
test_labels_classes = np.argmax(test_labels, axis=1)

# 1. Confusion Matrix
conf_matrix = confusion_matrix(test_labels_classes, y_pred_classes)
plt.figure(figsize=(10, 8))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
plt.title('Confusion Matrix')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.show()

# 2. Classification Report (Precision, Recall, F1 Score)
print("Classification Report:")
class_report = classification_report(test_labels_classes, y_pred_classes, target_names=class_names)
print(class_report)

# 3. ROC Curve and AUC-ROC
# Binarize the labels for multi-class ROC curve
y_test_binarized = label_binarize(test_labels_classes, classes=np.arange(10))
fpr = {}
tpr = {}
roc_auc = {}

plt.figure(figsize=(10, 8))
for i in range(10):
    fpr[i], tpr[i], _ = roc_curve(y_test_binarized[:, i], y_pred[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])
    plt.plot(fpr[i], tpr[i], label=f'ROC curve for {class_names[i]} (AUC = {roc_auc[i]:0.2f})')

plt.plot([0, 1], [0, 1], 'k--')  # Random guessing line
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curves for Fashion MNIST Classes')
plt.legend(loc="lower right")
plt.show()

# Calculate and display the average AUC-ROC score for all classes
average_roc_auc = roc_auc_score(y_test_binarized, y_pred, multi_class="ovr")
print(f"Average AUC-ROC Score: {average_roc_auc:.2f}")