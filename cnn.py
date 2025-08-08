# CNN for Digit Classification (MNIST)
# This code builds a simple CNN to classify MNIST digits as either '3 or 9
#' or 'not 3 or 9'.
# It uses TensorFlow and Keras for model building and training.
# It also includes data preprocessing, model training, and prediction.
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.datasets import mnist
import matplotlib.pyplot as plt

# 1. Load MNIST dataset
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# 2. Normalize pixel values to [0, 1]
x_train = x_train / 255.0
x_test = x_test / 255.0

# 3. Label: 1 if digit is 3 or 9, else 0
y_train_binary = np.isin(y_train, [4]).astype(np.float32)
y_test_binary = np.isin(y_test, [4]).astype(np.float32)

# 4. Build model
model = Sequential([
    Flatten(input_shape=(28, 28)),          # Convert 28x28 to 784
    #   Flatten(input_shape=(28, 28)),          # To test other input sample
    Dense(64, activation='relu'), # Fully connected layer
    Dense(1, activation='sigmoid')          # Binary output
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# 5. Train model
model.fit(x_train, y_train_binary, epochs=5, validation_data=(x_test, y_test_binary))

# 6. Predict a sample
index = 6  # change index to test other digits
sample = x_test[index]
pred = model.predict(sample.reshape(1, 28, 28), verbose=0)
pred_class = int(pred[0][0] >= 0.5)

print(f"Actual digit: {y_test[index]}")
print(f"Predicted: {'3 or 9' if pred_class else 'Not 3 or 9'} (Confidence: {pred[0][0]:.4f})")

# 7. Show the image
plt.imshow(sample, cmap='gray')
plt.title(f"Predicted: {'3 or 9' if pred_class else 'Not 3 or 9'}")
plt.axis('off')
plt.show()
