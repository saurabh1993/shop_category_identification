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



pathx= '/Users/saurabh_veda/workspace_kumar/bharat_pe_task/train_data/meat-fresh/6385798_5a912afe-e6c8-452b-9248-c29b4c5de4bd.jpeg'

class autoMLModel():
    def __init__(self,modelPath='autoMLModel'):
        loaded = tf.saved_model.load(export_dir='autoMLModel')
        self.infer = loaded.signatures["serving_default"]
        
    def run(self,pathx):
        img = cv2.imread(pathx)
        flag, bts = cv2.imencode('.jpg', img)
        inp = [bts[:,0].tobytes()]
        #loaded = tf.saved_model.load(export_dir='autoMLModel')
        #infer = loaded.signatures["serving_default"]
        out = self.infer(key=tf.constant('something_unique'), image_bytes=tf.constant(inp))
        labels=np.array(out['labels'][0]).tolist()
        probs=np.array(out['scores'][0]).tolist()
        #import pdb;pdb.set_trace()
        out={}
        for idx,label in enumerate(labels):
            out[label.decode('utf-8')]="{:.6f}".format(probs[idx])
        print("out",out)
        return out
        


if __name__ == '__main__':
    model=autoMLModel()
    print(json.dumps(model.run(pathx)))
    
