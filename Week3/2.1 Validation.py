from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam

''' 3 ways to prepare validation data '''

# # 1: fit split
# hisotry = model.fit(inputs, targets, validation_split=0.2)

# # 2: dataset help to split
# (X_train, y_train), (X_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()
# model.fit(X_train, y_train, validation_data=(X_test, y_test))

# # 3: use function 
# from sklearn.model_selection import train_test_split
# X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.1)
# model.fit(X_train, y_train, validation_data=(X_val, y_val))

''' example '''
from sklearn.datasets import load_diabetes
diabetes_dataset = load_diabetes()
print(diabetes_dataset["DESCR"])        # description

print(diabetes_dataset.keys())
data = diabetes_dataset["data"] 
targets = diabetes_dataset["target"]

targets = (targets - targets.mean(axis=0)) / targets.std()      # normalization label
print(targets)

from sklearn.model_selection import train_test_split
train_data, test_data, train_target, test_target = train_test_split(data, targets, test_size=0.1)
print("train_data.shape: ", train_data.shape)
print("train_target.shape: ", train_target.shape)
print("test_data.shape: ", test_data.shape)
print("test_target.shape: ", test_target.shape)

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

def get_model():
    model = Sequential([
        Dense(128, activation='relu', input_shape=(train_data.shape[1],)),          # generalization
        Dense(128, activation='relu'),
        Dense(128, activation='relu'),
        Dense(128, activation='relu'),
        Dense(128, activation='relu'),
        Dense(128, activation='relu'),
        Dense(1)
    ])
    return model

model = get_model()
model.summary()

model.compile(optimizer='adam', loss='mse', metrics='mae')
history = model.fit(train_data, train_target, epochs=100, validation_split=0.15, batch_size=64, verbose=False)
model.evaluate(test_data, test_target, verbose=2)

import matplotlib.pyplot as plt
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Loss vs. Epochs')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Training', 'Validation'], loc='upper right')
plt.show()