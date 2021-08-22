import keras
from keras import backend as K
from keras.layers.core import Dense, Activation
from keras.optimizers import Adam
from keras.metrics import categorical_crossentropy
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing import image
from keras.models import Model
from keras.applications import imagenet_utils
from keras.layers import Dense,GlobalAveragePooling2D
from keras.applications import MobileNet
from keras.applications.mobilenet import preprocess_input
import numpy as np
from keras.optimizers import Adam
import tensorflow as tf
import cv2
import json


'''This class is a wrapper for loading locally trained Mobilenet model'''
class localModel():
    def __init__(self,modelPath='localModels/shop_cat3.h5'):
        
        # Load the model from the model path
        self.model = tf.keras.models.load_model(modelPath)
    
    # Helper function to load and format image from the path
    def load_image(self,img_path, show=False):

        img = image.load_img(img_path, target_size=(150, 150))
        img_tensor = image.img_to_array(img)                    # (height, width, channels)
        img_tensor = np.expand_dims(img_tensor, axis=0)         # (1, height, width, channels), add a dimension because the model expects this shape: (batch_size, height, width, channels)
        img_tensor /= 255.                                      # imshow expects values in the range [0, 1]
        return img_tensor
    
    def run(self,pathx):
        
        CLASSES={0:"egg-fresh",1:"grocery",2:"meat-freah",3:"veg-fresh"}
        
        # Load image
        img=self.load_image(pathx)
        
        # Do inferencing
        pred=self.model.predict(img)[0].tolist()
        
        # Format the results
        out ={}
        for idx,prob in enumerate(pred):
            out[CLASSES[idx]]="{:.6f}".format(prob)
        print(out)
        return out

if __name__ == '__main__':
    model=localModel()
    pathx="sample.jpg"
    print(json.dumps(model.run(pathx)))