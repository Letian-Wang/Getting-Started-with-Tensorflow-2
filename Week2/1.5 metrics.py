import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
import tensorflow.keras.backend as K

# Build the model
model = Sequential([
  Flatten(input_shape=(28,28)),
  Dense(32, activation='relu'),
  Dense(32, activation='tanh'),
  Dense(10, activation='softmax'),
])

# Compile the model
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])


# Sigmoid activation function
y_true = tf.constant([0.0,1.0,1.0])
y_pred = tf.constant([0.4,0.8, 0.3])
accuracy = K.mean(K.equal(y_true, K.round(y_pred)))
print("Sigmoid: ", accuracy)

# Binary classification with softmax
y_true = tf.constant([[0.0,1.0],[1.0,0.0],[1.0,0.0],[0.0,1.0]])
y_pred = tf.constant([[0.4,0.6], [0.3,0.7], [0.05,0.95],[0.33,0.67]])
accuracy =K.mean(K.equal(y_true, K.round(y_pred)))
print("Binary: ", accuracy)

# Categorical classification with m>2
y_true = tf.constant([[0.0,1.0,0.0,0.0],[1.0,0.0,0.0,0.0],[0.0,0.0,1.0,0.0]])
y_pred = tf.constant([[0.4,0.6,0.0,0.0], [0.3,0.2,0.1,0.4], [0.05,0.35,0.5,0.1]])
accuracy = K.mean(K.equal(K.argmax(y_true, axis=-1), K.argmax(y_pred, axis=-1)))
print("Categorical: ", accuracy)


# Compile the model with default threshold (=0.5)
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['binary_accuracy'])
# The threshold can be specified as follows
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=[tf.keras.metrics.BinaryAccuracy(threshold=0.5)])

# sparse categorical accuracy: label is a single interger
# categorical accuracy: label is a vector
#Two examples of compiling a model with a sparse categorical accuracy metric (label: single integer).
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=["sparse_categorical_accuracy"])
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=[tf.keras.metrics.SparseCategoricalAccuracy()])

link: https://colab.research.google.com/drive/1qRuiduPccxN2wh9Brnrlj8WuvGVhZiXq#forceEdit=true&sandboxMode=true&scrollTo=TwPlWaNUz-dx
# Compile a model with a top-k categorical accuracy metric with default k (=5)
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=["top_k_categorical_accuracy"])


# Define a custom metric
def mean_pred(y_true, y_pred):
    return K.mean(y_pred)              
# Specify k instead with the sparse top-k categorical accuracy
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=[mean_pred])

# multiple metrics
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=[mean_pred, "accuracy",tf.keras.metrics.SparseTopKCategoricalAccuracy(k=3)])