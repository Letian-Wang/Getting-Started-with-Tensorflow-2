from tensorflow.keras.losses import BinaryCrossentropy
from tensorflow.keras.callbacks import ModelCheckpoint
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D

print(tf.__version__)

def knowledge():
    model = Sequential([
        Dense(64, activation='sigmoid', input_shape=(10,)),
        Dense(1)
    ])
    model.compile(optimizer='sgd', loss=BinaryCrossentropy(from_logits=True))

    ''' save model weights '''
    # 1st way: native format
    checkpoint = ModelCheckpoint('/save/my_model', save_weights_only=True)
    # 3 files:
    # checkpoint
    # my_model.data-00000-of-00001
    # my_model.index

    # 2rd way: keras format
    checkpoint = ModelCheckpoint('/save/keras_model.h5', save_weights_only=True)
    # keras_model.h5

    model.fit(X_train, y_train, epochs=10, callbacks=[checkpoint])

    # 3rd way: manually save without callback
    model.save_weights('my_model')

    ''' load model '''
    model = Sequential([
        Dense(64, activation='sigmoid', input_shape=(10,)),
        Dense(1)
    ])
    model.load_weights('my_model')
    model.load_weights('keras_model.h5')

''' example '''
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

# Introduce function to test model accuracy
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

model = get_new_model()
model.summary()

get_test_accuracy(model, x_test, y_test)

checkpoint_path = 'save/checkpoint'
# save weights every epoch (overwrite)
checkpoint = ModelCheckpoint(filepath=checkpoint_path, frequency='epoch', save_weights_only=True, verbose=1) 

model.fit(x_train, y_train, epochs=3, callbacks=[checkpoint])
get_test_accuracy(model, x_test, y_test)

model=get_new_model()   # new model
get_test_accuracy(model, x_test, y_test)

model.load_weights(checkpoint_path)     # load model
get_test_accuracy(model, x_test, y_test)
# ls -lh save
# rm -r save

plt.show()    
