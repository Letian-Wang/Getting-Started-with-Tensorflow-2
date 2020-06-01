from tensorflow.keras.callbacks import Callback

# class  my_callback(Callback):
#     def on_train_begin(self, logs=None):
#         # Do something at the start of training

#     def on_train_batch_begin(self, batch, logs=None):
#         # Do something at the start of every batch iteration

#     def on_epoch_end(self, epoch, logs=None):
#         # Do something at the end of every epoch

# model.fit(X_train, y_train, epochs=5, callbacks=[my_callback()])

class TrainingCallback(Callback):
    def on_train_begin(self, logs=None):
        print("Starting training....")

    def on_epoch_begin(self, epoch, logs=None):
        print("Starting epoch {0}".format(epoch))

    def on_train_batch_begin(self, batch, logs=None):
        print("Training: Starting batch {0}".format(batch))

    def on_train_batch_end(self, batch, logs=None):
        print("Training: Finished batch {0}".format(batch))

    def on_epoch_end(self, epoch, logs=None):
        print("Finished epoch {0}".format(epoch))
    
    def on_train_end(self, batch, logs=None):
        print("Finished training")

class TestingCallback(Callback):
    def on_test_begin(self, logs=None):
        print("Starting Testing....")

    def on_test_batch_begin(self, batch, logs=None):
        print("Testing: Starting batch {0}".format(batch))

    def on_test_batch_end(self, batch, logs=None):
        print("Testing: Finished batch {0}".format(batch))
    
    def on_test_end(self, batch, logs=None):
        print("Finished Testing")

class PredictionCallback(Callback):
    def on_predict_begin(self, logs=None):
        print("Starting Prediction....")

    def on_predict_batch_begin(self, batch, logs=None):
        print("Prediction: Starting batch {0}".format(batch))

    def on_predict_batch_end(self, batch, logs=None):
        print("Prediction: Finished batch {0}".format(batch))
    
    def on_predict_end(self, batch, logs=None):
        print("Finished Prediction")
from tensorflow.keras.layers import Dropout, Dense, Conv2D
from tensorflow.keras.models import Sequential
from tensorflow.keras import regularizers 
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_diabetes
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

diabetes_dataset = load_diabetes()
data = diabetes_dataset["data"] 
targets = diabetes_dataset["target"]
targets = (targets - targets.mean(axis=0)) / targets.std()      # normalization label
train_data, test_data, train_target, test_target = train_test_split(data, targets, test_size=0.1)
model = get_regularized_model(1e-5, 0.3)
model.compile(optimizer='adam', loss='mse')
print("----------------Training--------------------")
model.fit(train_data, train_target, epochs=3, batch_size=128,verbose=False, callbacks=[TrainingCallback()])
print("----------------Evaluating--------------------")
model.evaluate(test_data, test_target, verbose=False, callbacks=[TestingCallback()])
print("----------------Predicting--------------------")
model.predict(test_data, verbose=False, callbacks=[PredictionCallback()])