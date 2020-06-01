import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D
'''
model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['mape'])
history = model.fit(X_train, y_train, epochs=10, batch_size=16)           # default: epochs=1, batch_size=32
print(hisotry.history.keys())                   # dict_keys(['loss', 'mape', 'val_loss', 'val_mape'])
'''
# X_train: numpy array: (num_samples, num_features)
# y_train: numpy array: (num_samples, num_classes)  (categorical_crossentropy) (label is a vector) 
# y_train: numpy array: (num_samples, )  (sparse_categorical_crossentropy)  (label is a single integer)
# verbose: 0 (silent), 1 (all), 2 (one line for one epoch)


''' Example '''
from tensorflow.keras.preprocessing import image
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# 1. load data
fashion_mnist_data = tf.keras.datasets.fashion_mnist
(train_images, train_labels), (test_images, test_labels) = fashion_mnist_data.load_data()
print(train_images.shape)
print(train_labels[0])

train_images = train_images / 255
test_images = test_images / 255

labels = [
    'T-shirt/top',
    'Trouser',
    'Pullover',
    'Dress',
    'Coat',
    'Sandal',
    'Shirt',
    'Sneaker',
    'Bag',
    'Ankle boot'
]
i=0
img=train_images[i,:,:]
plt.imshow(img)
print("img: \n", img)

# 2. Build model
model = Sequential([
    Conv2D(16, (3,3), activation='relu' ,input_shape=(28, 28, 1)),
    MaxPooling2D((3, 3)),
    Flatten(),
    Dense(10, activation='softmax')
])
model.summary()

# 3. Compile and train
opt = tf.keras.optimizers.Adam(learning_rate=0.005)
acc = tf.keras.metrics.SparseCategoricalAccuracy()
mae = tf.keras.metrics.MeanAbsoluteError()
model.compile(optimizer=opt, loss='sparse_categorical_crossentropy', metrics=[acc, mae])

history = model.fit(train_images[...,np.newaxis], train_labels, epochs=8, batch_size=256)       # need a channel_dim
# X_train: numpy array: (num_samples, num_features)
# y_train: numpy array: (num_samples, num_classes)  (categorical_crossentropy) (label is a vector) 
# y_train: numpy array: (num_samples, )  (sparse_categorical_crossentropy)  (label is a single integer)
# verbose: 0 (silent), 1 (all), 2 (one line for one epoch)

# 4. Plot
df = pd.DataFrame(history.history)
print(df.head())

loss_plot = df.plot(y='loss', title="Loss vs. Epoch", legend=False)
loss_plot = df.plot(y='mean_absolute_error', title="Mae vs. Epoch", legend=False)
loss_plot.set(xlabel='Epochs', ylabel='Loss')
plt.show()