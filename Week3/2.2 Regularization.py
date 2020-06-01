from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv1D, Dropout
import tensorflow as tf
''' knowledge '''
def knowledge():
    # Regularization applies to both Dense and Conv layer
    model = Sequential([
        Dense(64, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.001)),           
        # l2: multiply the sum of squared kernel weights by 0.001, automatically added to loss function
        # l1: multiply the sum of absolute kernel weights by 0.005, automatically added to loss function
        #     setting some of the weights to be zero
        Conv1D(64, 3, activation='relu', kernel_regularizer=tf.keras.regularizers.l1(0.005)),
        Dense(64, activation='relu', kernel_regularizer=tf.keras.regularizers.l1_l2(l1=0.005, l2=0.001), 
                                    bias_regularizer=tf.keras.regularizers.l2(0.001)),
        Dropout(0.5),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adadelta', loss='binary_crossentropy', metrics=['acc'])
    model.fit(inputs, targets, validation_split=0.25)               # Training mode, with dropout
    model.evaluate(val_inputs, val_targets)                         # Testing mode, no dropout
    model.predict(test_inputs)                                      # Testing mode, no dropout

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

from tensorflow.keras.layers import Dropout
from tensorflow.keras import regularizers 
def get_regularized_model(wd, rate):
    model = Sequential([
        Dense(128, kernel_regularizer=regularizers.l2(wd), activation='relu', input_shape=(train_data.shape[1],)),          # generalization
        Dropout(rate),
        Dense(128, kernel_regularizer=regularizers.l2(wd), activation='relu'),
        Dropout(rate),
        Dense(128, kernel_regularizer=regularizers.l2(wd), activation='relu'),
        Dropout(rate),
        Dense(128, kernel_regularizer=regularizers.l2(wd), activation='relu'),
        Dropout(rate),
        Dense(128, kernel_regularizer=regularizers.l2(wd), activation='relu'),
        Dropout(rate),
        Dense(128, kernel_regularizer=regularizers.l2(wd), activation='relu'),
        Dropout(rate),
        Dense(1)
    ])
    return model

model = get_regularized_model(1e-5, 0.3)
model.compile(optimizer='adam', loss='mse', metrics=['mae'])
history = model.fit(train_data, train_target, epochs=100, validation_split=0.15, batch_size=64, verbose=False)

model.evaluate(test_data, test_target, verbose=2)

import matplotlib.pyplot as plt
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Loss vs. epochs')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Training', 'Validation'], loc='upper right')
plt.show()
