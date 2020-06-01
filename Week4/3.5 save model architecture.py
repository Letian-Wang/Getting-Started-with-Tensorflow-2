import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import json
import numpy as np
import yaml
model = Sequential([
    Dense(units=32, input_shape=(32, 32, 3), activation='relu', name='dense_1'),
    Dense(units=10, activation='softmax', name='dense_2')
])

''' Method 1 '''
# get_config returns the model's architecture as a dictionary
config_dict = model.get_config()
print("config_dict")
print(config_dict)

# Creating a new model from the config with reinitialized weights
# For models that are not Sequential models, 
# use tf.keras.Model.from_config instead of tf.keras.Sequential.from_config.
model_same_config = tf.keras.Sequential.from_config(config_dict)

# check model architecture and weights
print('Same config:', 
      model.get_config() == model_same_config.get_config())
print('Same value for first weight matrix:', 
      np.allclose(model.weights[0].numpy(), model_same_config.weights[0].numpy()))

''' Other file formats: JSON and YAML '''
''' JSON '''
json_string = model.to_json()
print("-----------------------json_string")
print(json_string)
# Write out JSON config file
with open('config.json', 'w') as f:
    json.dump(json_string, f)
del json_string
# Read in JSON config file again
with open('config.json', 'r') as f:
    json_string = json.load(f)
# Reinitialize model
model_same_config = tf.keras.models.model_from_json(json_string)


''' JSON(Not run) '''
yaml_string = model.to_yaml()
print("-----------yaml_string")
print(yaml_string)
# Write out yaml config file
with open('config.yaml', 'w') as f:
    yaml.dump(yaml_string, f)
del yaml_string
# Read in yaml config file again
with open('config.yaml', 'r') as f:
    yaml_string = yaml.load(f)
# Reinitialize model
model_same_config = tf.keras.models.model_from_yaml(yaml_string)