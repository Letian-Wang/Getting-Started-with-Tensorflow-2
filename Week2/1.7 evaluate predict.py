import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

model = Sequential(Dense(3, activation='softmax', input_shape=(12,)))
model.compile(optimizer='sgd', loss='categorical_crossentropy', metrics=['accuracy', 'mae'])
model.fit(X_train, y_train)

# predict/evaluate layer input: (num_sample, (feature_dim))
loss, accuracy, mae = model.evaluate(X_test, y_test)    # (loss, mean absolute error, mean absolute percentage error) evaluated on the test set
pred = model.predict(X_sample)                          # X_sample = (num_sample, 12),  output: (num_sample, 3)
# Convolution predict/evaluate layer input: (num_sample, (feature_dim), num_channels)
model.evaluate(test_images[...,np.newaxis], test_labels)  # need a channel_dim 
model.predict(test_image[np.newaxis, ..., np.newaxis])  # need a num_sample dim and a channel_dim
