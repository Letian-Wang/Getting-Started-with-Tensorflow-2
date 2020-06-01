import tensorflow as tf
print(tf.__version__)

from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
diabetes_dataset = load_diabetes()
data = diabetes_dataset['data']
targets = diabetes_dataset['target']
train_data, test_data, train_targets, test_targets = train_test_split(data, targets, test_size=0.1)

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
model = tf.keras.Sequential([
    Dense(128, activation='relu', input_shape=(train_data.shape[1],)),
    Dense(64,activation='relu'),
    Dense(64, activation='relu'),
    Dense(64, activation='relu'),
    Dense(1)        
])
model.compile(loss='mse', optimizer="adam",metrics=["mse","mae"])

''' Learning rate scheduler '''
def lr_scheduler():
    # 1. Define the learning rate schedule function
    # def lr_function(epoch, lr):
    #     if epoch % 2 == 0:
    #         return lr
    #     else:
    #         return lr + epoch/1000
    # history = model.fit(train_data, train_targets, epochs=10,
    #                     callbacks=[tf.keras.callbacks.LearningRateScheduler(lr_function, verbose=1)], verbose=False)
    # 2. Train the model with a difference schedule
    history = model.fit(train_data, train_targets, epochs=10,
                        callbacks=[tf.keras.callbacks.LearningRateScheduler(lambda x:1/(x+1), verbose=1)], 
                        verbose=False)

''' CSV logger '''    
def CSV_logger():
    history = model.fit(train_data, train_targets, epochs=10,
                        callbacks=[tf.keras.callbacks.CSVLogger("results.csv")], verbose=False)
    import pandas as pd
    pd.read_csv("results.csv", index_col='epoch')
    # Note:
    # This callback streams the results from each epoch into a CSV file. 
    # The first line of the CSV file will be the names of pieces of information recorded on each subsequent line, beginning with the epoch and loss value. 
    # The values of metrics at the end of each epoch will also be recorded.
    # The only compulsory argument is the filename for the log to be streamed to. This could also be a filepath.
    # You can also specify the separator to be used between entries on each line.
    # The append argument allows you the option to append your results to an existing file with the same name. This can be particularly useful if you are continuing training.

''' Lambda callbacks '''
def Lambda_callbacks():
    # Print the epoch number at the beginning of each epoch
    epoch_callback = tf.keras.callbacks.LambdaCallback(
        on_epoch_begin=lambda epoch,logs: print('Starting Epoch {}!'.format(epoch+1)))
    # Print the loss at the end of each batch
    batch_loss_callback = tf.keras.callbacks.LambdaCallback(
        on_batch_end=lambda batch,logs: print('\n After batch {}, the loss is {:7.2f}.'.format(batch, logs['loss'])))
    # Inform that training is finished
    train_finish_callback = tf.keras.callbacks.LambdaCallback(
        on_train_end=lambda logs: print('Training finished!'))
    # Train the model with the lambda callbacks
    history = model.fit(train_data, train_targets, epochs=5, batch_size=100,
                        callbacks=[epoch_callback, batch_loss_callback,train_finish_callback], verbose=False)

''' Reduce learning rate on plateau '''
def Reduce_lr_on_plateau():
    # Train the model with the ReduceLROnPlateau callback
    history = model.fit(train_data, train_targets, epochs=100, batch_size=100,
                        callbacks=[tf.keras.callbacks.ReduceLROnPlateau(
                            monitor="loss", factor=0.2, min_delta=1, patience=1,verbose=2, cooldown=5)], verbose=2)

    # Usage :tf.keras.callbacks.ReduceLROnPlateau( monitor='val_loss', factor=0.1, patience=10, verbose=0, mode='auto', min_delta=0.0001, cooldown=0, min_lr=0)
    # 1. The argument monitor is used to specify which metric to base the callback on.
    # 2. The factor is the factor by which the learning rate decreases i.e., new_lr=factor*old_lr.
    # 3. The patience is the number of epochs where there is no improvement on the monitored metric before the learning rate is reduced.
    # 4. The verbose argument will produce progress messages when set to 1.
    # 5. The mode determines whether the learning rate will decrease when the monitored quantity stops increasing (max) or decreasing (min). The auto setting causes the callback to infer the mode from the monitored quantity.
    # 6. The min_delta is the smallest change in the monitored quantity to be deemed an improvement.
    # 7. The cooldown is the number of epochs to wait after the learning rate is changed before the callback resumes normal operation.
    # 8. The min_lr is a lower bound on the learning rate that the callback will produce.

# lr_scheduler()
# CSV_logger()
# Lambda_callbacks()
Reduce_lr_on_plateau()