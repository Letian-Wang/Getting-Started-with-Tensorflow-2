import tensorflow as tf
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten, Dense, Conv2D, MaxPooling2D, BatchNormalization, Dropout

print(tf.__version__)
diabetes_dataset = load_diabetes()
data = diabetes_dataset['data']
targets = diabetes_dataset['target']

# Normalise the target data (this will make clearer training curves)
targets = (targets - targets.mean(axis=0)) / (targets.std())

# Split the dataset into training and test datasets 
train_data, test_data, train_targets, test_targets = train_test_split(data, targets, test_size=0.1)

model = Sequential([
    Dense(64, input_shape=[train_data.shape[1],], activation="relu"),
    BatchNormalization(),  # <- Batch normalisation layer
    Dropout(0.5),
    BatchNormalization(),  # <- Batch normalisation layer
    Dropout(0.5),
    Dense(256, activation='relu'),
])

# Add a customised batch normalisation layer
model.add(tf.keras.layers.BatchNormalization(
    momentum=0.95, 
    epsilon=0.005,
    axis = -1,
    beta_initializer=tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.05), 
    gamma_initializer=tf.keras.initializers.Constant(value=0.9)
))
# Recall that there are some parameters and hyperparameters associated with batch normalisation.
    # 1. The hyperparameter momentum is the weighting given to the previous running mean when re-computing it with an extra minibatch. By default, it is set to 0.99.
    # 2. The hyperparameter  𝜖  is used for numeric stability when performing the normalisation over the minibatch. By default it is set to 0.001.
    # 3. The parameters  𝛽  and  𝛾  are used to implement an affine transformation after normalisation. By default,  𝛽  is an all-zeros vector, and  𝛾  is an all-ones vector.
model.add(Dense(1))

# Compile the model
model.compile(optimizer='adam', loss='mse', metrics=['mae'])
history = model.fit(train_data, train_targets, epochs=100, validation_split=0.15, batch_size=64,verbose=False)

# Plot the learning curves

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

frame = pd.DataFrame(history.history)
epochs = np.arange(len(frame))

fig = plt.figure(figsize=(12,4))

# Loss plot
ax = fig.add_subplot(121)
ax.plot(epochs, frame['loss'], label="Train")
ax.plot(epochs, frame['val_loss'], label="Validation")
ax.set_xlabel("Epochs")
ax.set_ylabel("Loss")
ax.set_title("Loss vs Epochs")
ax.legend()

# Accuracy plot
ax = fig.add_subplot(122)
ax.plot(epochs, frame['mae'], label="Train")
ax.plot(epochs, frame['val_mae'], label="Validation")
ax.set_xlabel("Epochs")
ax.set_ylabel("Mean Absolute Error")
ax.set_title("Mean Absolute Error vs Epochs")
ax.legend()
plt.show()