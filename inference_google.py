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

'''This class is a wrapper for loading autoMl trained model'''
class autoMLModel():
    def __init__(self,modelPath='autoMLModel'):
        
        # Load the model from the model path
        loaded = tf.saved_model.load(export_dir='autoMLModel')
        self.infer = loaded.signatures["serving_default"]

    def run(self,pathx):
        
        # Read the image from the path and convert it to 
        # consumable format
        img = cv2.imread(pathx)
        flag, bts = cv2.imencode('.jpg', img)
        inp = [bts[:,0].tobytes()]
        
        # Do the inferencing
        out = self.infer(key=tf.constant('something_unique'), image_bytes=tf.constant(inp))
        
        # Format the results
        labels=np.array(out['labels'][0]).tolist()
        probs=np.array(out['scores'][0]).tolist()
        out={}
        for idx,label in enumerate(labels):
            out[label.decode('utf-8')]="{:.6f}".format(probs[idx])
        print("out",out)
        return out
        


if __name__ == '__main__':
    model=autoMLModel()
    pathx="sample.jpg"
    print(json.dumps(model.run(pathx)))
    
