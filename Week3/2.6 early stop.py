from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, Flatten, Dense, MaxPooling1D
from tensorflow.keras.callbacks import EarlyStopping

model = Sequential([
    Conv1D(16, 5, activation='relu', input_shape=(128, 1)),
    MaxPooling1D(4),
    Flatten(),
    Dense(10, activation='softmax')
])
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
early_stopping = EarlyStopping(monitor='val_loss')
early_stopping = EarlyStopping(monitor='val_accuracy', patience=5, min_delta=0.01, mode='max')
# patience: Terminate training if there is no improvement for 5 epochs in a row (Default:0)
# min_delta: How much change of performance can be regarded as a improvement   (Default:0.001)
# mode: 'max' means increase is better, 'min' means decrease is better (Default: automatically infer by name of monitor)
model.fit(X_train, Y_train, validation_split=0.2, epochs=100, callbacks=[early_stopping])

