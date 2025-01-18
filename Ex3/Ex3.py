from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing import sequence
import numpy as np
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, SimpleRNN, Dense
from tensorflow.keras.losses import BinaryCrossentropy
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.datasets import fashion_mnist
from tensorflow.keras.utils import to_categorical
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.preprocessing import label_binarize
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc, precision_recall_curve, precision_score, recall_score, f1_score, roc_auc_score
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.datasets import reuters
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout
from tensorflow.keras.datasets import reuters
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Bidirectional, LSTM, Dropout, Dense, Embedding
from sklearn.utils.class_weight import compute_class_weight



#----------------------------------------------Exercise 1: Recurrent Neural Networks (RNNs)---------------------------------------------
#4


# Load the dataset
# Only considering the top 10,000 most frequently occurring words
(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=10000)
# Pad the data so that all sequences have the same length
x_train = sequence.pad_sequences(x_train, maxlen=500)
x_test = sequence.pad_sequences(x_test, maxlen=500)
print('Train set shape:', x_train.shape)
print('Test set shape:', x_test.shape)

# Build the RNN model
model = Sequential()

# Add an Embedding layer
model.add(Embedding(input_dim=10000, output_dim=32, input_length=500))

# Add a SimpleRNN layer
model.add(SimpleRNN(units=32))

# Add a Dense layer with sigmoid activation for binary classification
model.add(Dense(1, activation='sigmoid'))

# Compile the model
model.compile(optimizer=Adam(), loss=BinaryCrossentropy(), metrics=['accuracy'])

# Summary of the model architecture
model.summary()


# Train the model
history = model.fit(x_train, y_train, epochs=3, batch_size=64, validation_split=0.2)


# Evaluate the model using the testing data
test_loss, test_acc = model.evaluate(x_test, y_test, verbose=2)

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

# Predict the test data (returns probabilities)
y_pred_prob = model.predict(x_test)
# Convert probabilities to binary predictions (0 or 1)
y_pred = np.where(y_pred_prob > 0.5, 1, 0)


# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
# Plot Confusion Matrix
plt.figure(figsize=(6, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Negative', 'Positive'], yticklabels=['Negative', 'Positive'])
plt.title('Confusion Matrix')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.show()


# Classification Report
print("Classification Report:")
print(classification_report(y_test, y_pred, target_names=['Negative', 'Positive']))


# ROC Curve and AUC
fpr, tpr, _ = roc_curve(y_test, y_pred_prob)  # Using probabilities for ROC
roc_auc = auc(fpr, tpr)
# Plot ROC Curve
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc="lower right")
plt.show()
#--------------------------------------------------------------------------------------------------------------------------



#----------------------------------------------Exercise 2: Long Short-Term Memory Networks (LSTM)---------------------------------------------
#5

# Load the dataset
# Only considering the top 10,000 most frequently occurring words
(x_train, y_train), (x_test, y_test) = reuters.load_data(num_words=10000)
# Pad the data so that all sequences have the same length
x_train = pad_sequences(x_train, maxlen=200)
x_test = pad_sequences(x_test, maxlen=200)
print('Train set shape:', x_train.shape)
print('Test set shape:', x_test.shape)


# Define the model
model = Sequential()

# Embedding layer (turns integers into dense vectors of fixed size)
embedding_dim = 128
model.add(Embedding(input_dim=10000, output_dim=embedding_dim, input_length=200))

# LSTM layer
model.add(LSTM(128, return_sequences=False))  # 128 LSTM units
# model.add(LSTM(64, return_sequences=False))  # Second LSTM layer

# Dropout for regularization
model.add(Dropout(0.5))

# Dense output layer with softmax activation
model.add(Dense(46, activation='softmax'))

# Compile the model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Model summary
model.summary()

# # Compute class weights
# class_weights = compute_class_weight('balanced', classes=np.unique(np.argmax(y_train, axis=1)), y=np.argmax(y_train, axis=1))
# class_weight_dict = dict(enumerate(class_weights))

#Train the model
history = model.fit(x_train, y_train, epochs=10, batch_size=64, validation_split=0.2)











# Evaluate the model using the testing data
test_loss, test_acc = model.evaluate(x_test, y_test, verbose=2)
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



# One-hot encode the labels for the ROC-AUC curve
y_test_one_hot = to_categorical(y_test, num_classes=46)

# Predict the classes for the test data
y_pred = model.predict(x_test)
y_pred_classes = np.argmax(y_pred, axis=1)

# 1. Confusion Matrix
cm = confusion_matrix(y_test, y_pred_classes)

plt.figure(figsize=(10, 7))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
plt.title('Confusion Matrix')
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.show()




# 2. Classification Report
print("Classification Report:")
print(classification_report(y_test, y_pred_classes))




# 3. ROC Curve and AUC-ROC

# For multiclass, you need to binarize the output labels
y_test_binarized = label_binarize(y_test, classes=np.arange(46))

# Initialize variables for micro-averaged ROC curve
fpr = dict()
tpr = dict()
roc_auc = dict()

# Compute ROC curve and ROC area for each class
for i in range(46):
    fpr[i], tpr[i], _ = roc_curve(y_test_binarized[:, i], y_pred[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

# Plot ROC curve for a few classes (or aggregate if you prefer)
plt.figure(figsize=(10, 8))
for i in range(5):  # Limit to 5 classes for clarity; can be adjusted
    plt.plot(fpr[i], tpr[i], label=f'Class {i} (AUC = {roc_auc[i]:.2f})')

# Plotting micro-average ROC curve (aggregating all classes)
fpr_micro, tpr_micro, _ = roc_curve(y_test_binarized.ravel(), y_pred.ravel())
roc_auc_micro = auc(fpr_micro, tpr_micro)
plt.plot(fpr_micro, tpr_micro, label=f'Micro-average (AUC = {roc_auc_micro:.2f})', color='navy', linestyle='--')

# Plot settings
plt.plot([0, 1], [0, 1], 'k--')  # Diagonal line
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend(loc='lower right')
plt.show()