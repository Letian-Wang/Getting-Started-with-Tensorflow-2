import tensorflow as tf
import tensorflow_hub as hub
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.applications.resnet50 import preprocess_input, decode_predictions
import pandas as pd
import numpy as np
from os import getcwd
# load model
model = tf.keras.Sequential([
    hub.KerasLayer("https://tfhub.dev/google/imagenet/mobilenet_v1_050_160/classification/4")
])
model.build(input_shape=[None, 160, 160, 3])  # Batch input shape.
# m.save('models/Tensorflow_MobilieNet-V1')
model.summary()

lemon_img = load_img('image/lemon.jpg', target_size=(160, 160))
viaduct_img = load_img('image/viaduct.jpg', target_size=(160, 160))
water_tower_img = load_img('image/water_tower.jpg', target_size=(160, 160))


with open(getcwd()+'/ImageNetLabels.txt') as txt_file:
    categories = txt_file.read().splitlines()
def get_top_5_predictions(img):
    x = img_to_array(img)[np.newaxis, ...] / 255.0
    preds = model.predict(x)
    top_preds = pd.DataFrame(columns=['prediction'],
                             index=np.arange(5)+1)
    sorted_index = np.argsort(-preds[0])
    for i in range(5):
        ith_pred = categories[sorted_index[i]]
        top_preds.loc[i+1, 'prediction'] = ith_pred
            
    return top_preds

lemon_preds = get_top_5_predictions(lemon_img)
print("------------lemon_preds")
print(lemon_preds)
viaduct_preds = get_top_5_predictions(viaduct_img)
print("------------viaduct_preds")
print(viaduct_preds)
water_tower_preds = get_top_5_predictions(water_tower_img)
print("------------water_tower_preds")
print(water_tower_preds)