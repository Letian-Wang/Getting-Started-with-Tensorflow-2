import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten, Dense, Conv1D, MaxPooling1D 
model = Sequential([
    Dense(64, activation='relu', input_shape=(32,)),
    Dense(1, activation='sigmoid')
])
model.compile(
    optimizer='sgd',                    # 'adam', 'RMSprop', 'adadelta'
    loss='binary_crossentropy',         # 'mean_squared_error', 'categorical_crossentropy'
    metrics=['accuracy', 'mae']                
    )

# More control:
model = Sequential([
    Dense(64, activation='relu', input_shape=(32,)),
    Dense(1, activation='linear')
])
model.compile(
    optimizer = tf.keras.optimizers.SGD(learning_rate=0.001, momentum=0.9, nesterov=True),  # Default: 0.01, 0, 
    loss = tf.keras.losses.BinaryCrossentropy(from_logits=True),    # Tells the model to execute sigmoid loss (no difference, just more numerically stable)
    metrics = [tf.keras.metrics.BinaryAccuracy(threshold=0.7), tf.keras.metrics.MeanAbsoluteError()]
)