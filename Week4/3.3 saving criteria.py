from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.losses import BinaryCrossentropy
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D

def knowledge():
    # save weight every epoch
    checkpoint = ModelCheckpoint('training_run_1/my_model', save_weigths_only=True,
                                save_freq='epoch')

    # save weight every 1000 sample (one sample is one batch )
    checkpoint = ModelCheckpoint('training_run_1/my_model', save_weigths_only=True,
                                save_freq=1000)

    # save weight when monitor is best so far
    checkpoint = ModelCheckpoint('training_run_1/my_model', save_weigths_only=True,
                                save_best_only=True, monitor='val_loss', mode='min')

    # save weight with name of epoch and batch
    checkpoint = ModelCheckpoint('training_run_1/my_model.{epoch}.{batch}', save_weigths_only=True,
                                save_freq=1000)

    # save weight with name of epoch
    checkpoint = ModelCheckpoint('training_run_1/my_model.{epoch}-{val_loss:4f}', save_weigths_only=True,
                                save_freq=1000)

    model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=10,
                batch_size=16, callbacks=[checkpoint])

''' example '''
def get_test_accuracy(model, x_test, y_test):
    test_loss, test_acc = model.evaluate(x=x_test, y=y_test, verbose=0)
    print('accuracy: {acc:0.3f}'.format(acc=test_acc))
def get_new_model():
    model = Sequential([
        Conv2D(filters=16, input_shape=(32, 32, 3), kernel_size=(3, 3), 
            activation='relu', name='conv_1'),
        Conv2D(filters=8, kernel_size=(3, 3), activation='relu', name='conv_2'),
        MaxPooling2D(pool_size=(4, 4), name='pool_1'),
        Flatten(name='flatten'),
        Dense(units=32, activation='relu', name='dense_1'),
        Dense(units=10, activation='softmax', name='dense_2')
    ])
    model.compile(optimizer='adam',
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy'])
    return model

def save_by_epoch():
    # Import the CIFAR-10 dataset and rescale the pixel values
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
    x_train = x_train / 255.0
    x_test = x_test / 255.0
    # Use smaller subset -- speeds things up
    x_train = x_train[:10000]
    y_train = y_train[:10000]
    x_test = x_test[:1000]
    y_test = y_test[:1000]
    # Plot the first 10 CIFAR-10 images
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 10, figsize=(10, 1))
    for i in range(10):
        ax[i].set_axis_off()
        ax[i].imshow(x_train[i])

    model = get_new_model()
    model.summary()
    get_test_accuracy(model, x_test, y_test)



    checkpoint_5000_path = 'model_checkpoints_5000/checkpoint_{epoch:02d}'
    checkpoint_5000 = ModelCheckpoint(filepath=checkpoint_5000_path, save_weights_only=True, save_freq=1000, verbose=1)
    model = get_new_model()
    model.fit(x_train, y_train, epochs=3, validation_data=(x_test, y_test), batch_size=10, callbacks=[checkpoint_5000])

def save_best():

    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
    x_train = x_train / 255.0
    x_test = x_test / 255.0
    x_train = x_train[:100]
    y_train = y_train[:100]
    x_test = x_test[:100]
    y_test = y_test[:100]

    model = get_new_model()
    checkpoint_best_path = 'model_checkpoints_best/checkpoint'
    checkpoint_best = ModelCheckpoint(filepath=checkpoint_best_path, save_weights_only=True, save_freq='epoch', 
                        save_best_only=True, monitor='val_accuracy', verbose=1)
    history = model.fit(x_train, y_train, epochs=50,  validation_data=(x_test, y_test), 
                       batch_size = 10, callbacks = [checkpoint_best], verbose=0)

    import pandas as pd
    df = pd.DataFrame(history.history)
    df.plot(y=['accuracy', 'val_accuracy'])
    new_model = get_new_model()
    checkpoint_best_path = 'model_checkpoints_best/checkpoint'
    new_model.load_weights(checkpoint_best_path)
    get_test_accuracy(new_model, x_test, y_test)
    return history
    
# save_by_epoch()

history = save_best()

