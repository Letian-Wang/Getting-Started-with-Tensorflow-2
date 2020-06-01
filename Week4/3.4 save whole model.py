from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.losses import BinaryCrossentropy
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D
from tensorflow.keras.models import load_model

def knowledge():
    ''' Save model '''
    # 1.SavedModel format
    checkpoint = ModelCheckpoint('my_model', save_weighs_only=False)
    model.fit(X_train, y_train, epochs=10, callbacks=[checkpoint])
    # my_model/assets/
    # my_model/saved_model.pb                                   # graph
    # my_model/variables/variables.data-00000-of=00001          # weights
    # my_model/variables/variables.index                        # weights

    # 2.keras format
    checkpoint = ModelCheckpoint('keras_model.h5', save_weighs_only=False)
    # keras_model.h5

    # 3. Manually save
    model.save('my_model')              # SavedModel format
    model.save('keras_model.h5')        # keras format

    ''' Load model '''
    from tensorflow.keras.models import load_model
    new_model = load_model('my_model')
    new_model = load_model('keras_model.h5')

    ''' example: '''
    # save the entire model in the native Tensorflow SavedModel format in separate folders for each epoch, 
    #   but only when the validation accuracy is the best so far in the training run
    checkpoint = ModelCheckpoint('model-{epoch}.h5', save_weights_only=False, save_best_only=True, monitor='val_accuracy')

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

# Import the CIFAR-10 dataset and rescale the pixel values
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
x_train = x_train / 255.0
x_test = x_test / 255.0
# Use smaller subset -- speeds things up
x_train = x_train[:10000]
y_train = y_train[:10000]
x_test = x_test[:1000]
y_test = y_test[:1000]

# callbacks save
checkpoint_path = 'model_checkpoints'
checkpoint = ModelCheckpoint(checkpoint_path, save_weights_only=False, 
            frequency='epoch', verbose=1)
model = get_new_model()
model.fit(x_train, y_train, epochs=3, callbacks=[checkpoint])
get_test_accuracy(model, x_test, y_test)

# mannually save
model.save('my_model')
model.save('keras_model.h5')

# load from scratch
model = load_model('keras_model.h5')
get_test_accuracy(model, x_test, y_test)
model = load_model('my_model')
get_test_accuracy(model, x_test, y_test)
