from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Softmax, Conv2D, MaxPooling2D

model = Sequential([
    Flatten(input_shape=(28, 28)),              # input_shape=() is optional
    Dense(16, activation='relu', name='layer_1'),            
    Dense(16, activation='relu'),      
    Dense(10, activation='softmax')
])
# # Alternative way of modelling
# model = Sequential()
# model.add(Dense(64, activation='relu', input_shape=(784,)))
# model.add(Dense(10, activation='softmax'))

print(model.weights[1].numpy())
print(model.layers)
model.summary()


model2 = Sequential([
    Conv2D(16, (3,3), padding='SAME', strides = 2, activation='relu', input_shape=(28, 28, 1), data_format='channels_first'),
    MaxPooling2D((3,3)),        # (None, 5, 4, 1) (First dimension is batch_size)
    Flatten(),
    Dense(10, activation='softmax')
])
model2.summary()