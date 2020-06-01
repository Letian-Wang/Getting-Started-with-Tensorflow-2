import tensorflow as tf
print(tf.__version__)

from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
diabetes_dataset = load_diabetes()
data = diabetes_dataset['data']
targets = diabetes_dataset['target']
train_data, test_data, train_targets, test_targets = train_test_split(data, targets, test_size=0.1)

# Build the model

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
model = tf.keras.Sequential([
    Dense(128, activation='relu', input_shape=(train_data.shape[1],)),
    Dense(64,activation='relu'),
    tf.keras.layers.BatchNormalization(),
    Dense(64, activation='relu'),
    Dense(64, activation='relu'),
    Dense(1)        
])
model.compile(loss='mse', optimizer="adam", metrics=['mae'])

# Create the custom callback
class LossAndMetricCallback(tf.keras.callbacks.Callback):

    # Print the loss after every second batch in the training set
    def on_train_batch_end(self, batch, logs=None):
        if batch %2 ==0:
            print('\n After batch {}, the loss is {:7.2f}.'.format(batch, logs['loss']))
    
    # Print the loss after each batch in the test set
    def on_test_batch_end(self, batch, logs=None):
        print('\n After batch {}, the loss is {:7.2f}.'.format(batch, logs['loss']))

    # Print the loss and mean absolute error after each epoch
    def on_epoch_end(self, epoch, logs=None):
        print('Epoch {}: Average loss is {:7.2f}, mean absolute error is {:7.2f}.'.format(epoch, logs['loss'], logs['mae']))
    
    # Notify the user when prediction has finished on each batch
    def on_predict_batch_end(self,batch, logs=None):
        print("Finished prediction on batch {}!".format(batch))

print("----------------Training--------------------")
history = model.fit(train_data, train_targets, epochs=20, batch_size=100, callbacks=[LossAndMetricCallback()], verbose=False)
print("----------------Evaluating--------------------")
model_eval = model.evaluate(test_data, test_targets, batch_size=10, callbacks=[LossAndMetricCallback()], verbose=False)
print("----------------Predicting--------------------")
model_pred = model.predict(test_data, batch_size=10, callbacks=[LossAndMetricCallback()], verbose=False)


