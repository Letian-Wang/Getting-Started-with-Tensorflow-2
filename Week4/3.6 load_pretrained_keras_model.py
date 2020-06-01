from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import preprocess_input, decode_predictions
import numpy as np
def knowledge():
    model = ResNet50(weights='imagenet', include_top=True)    #~/.keras.moels/

    # weights='imagenet': with weights learned from ImageNet dataset
    # weights=None: with weights randomly reinitialized for fresh training

    # include_top=True: include the whole model (Default)
    # include_top=False: exclude the CNN on the top of network (for transfer learning purpose)


    image_input = image.load_img('lemon.jpg', target_size=(224,224))  # (224,224) is for ResNet50
    image_input = image.img_to_array(image_input)
    image_input = preprocess_input(image_input[np.newaxis,...])     

    preds = model.predict(image_input)
    decoded_predictions = decode_predictions(preds, top=3)[0]       # retrive top 3 model predictions
    # list of (class, description, probability)

from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.applications.resnet50 import preprocess_input, decode_predictions
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

model = ResNet50(weights='imagenet', include_top=True)    #~/.keras.moels/
lemon_img = load_img('image/lemon.jpg', target_size=(224, 224))
viaduct_img = load_img('image/viaduct.jpg', target_size=(224, 224))
water_tower_img = load_img('image/water_tower.jpg', target_size=(224, 224))

def get_top_5_predictions(img):
    i = img_to_array(img)
    x = img_to_array(img)[np.newaxis, ...]
    x = preprocess_input(x)
    preds = decode_predictions(model.predict(x), top=5)
    top_preds = pd.DataFrame(columns=['prediction', 'probability'],
                             index=np.arange(5)+1)
    for i in range(5):
        top_preds.loc[i+1, 'prediction'] = preds[0][i][1]
        top_preds.loc[i+1, 'probability'] = preds[0][i][2] 
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