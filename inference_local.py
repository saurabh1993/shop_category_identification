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

pathx = '/Users/saurabh_veda/workspace_kumar/bharat_pe_task/train_data/meat-fresh/689350_f18d4ec7-3018-47ee-b4a3-ac7320d236e2.jpeg'
pathx= '/Users/saurabh_veda/workspace_kumar/bharat_pe_task/train_data/grocery/246781_88c67784-de0b-4bd6-8f85-4c4c959fb9ec.jpeg'
pathx = '/Users/saurabh_veda/workspace_kumar/bharat_pe_task/train_data/veg-fresh/3184806_c5f7de1a-eacc-4ae3-ab99-bc20b1d5e313.jpeg'
pathx = '/Users/saurabh_veda/workspace_kumar/bharat_pe_task/train_data/veg-fresh/1081018_5efe0420-6e98-4ecc-97e5-c2c8976f7b87.jpeg'
pathx = '/Users/saurabh_veda/workspace_kumar/bharat_pe_task/train_data/veg-fresh/8838357_2129ea53-a8c4-400f-ade1-c9cc4a60b50a.jpeg'
pathx = '/Users/saurabh_veda/workspace_kumar/bharat_pe_task/train_data/veg-fresh/9399798_0f4d6dd2-414b-4345-9aec-6f1327149a87.jpeg'
pathx = '/Users/saurabh_veda/workspace_kumar/bharat_pe_task/train_data/veg-fresh/8283462_fcf4afdf-b15c-48e7-b9d9-7505b9317328.jpeg'
pathx = '/Users/saurabh_veda/workspace_kumar/bharat_pe_task/train_data/veg-fresh/8563665_5c05bb9d-4479-4a97-9064-75d42395ebf7.jpeg'
pathx = '/Users/saurabh_veda/workspace_kumar/bharat_pe_task/train_data/grocery/721756_996a0845-b21f-4614-9a53-9c134eeb7685.jpeg'

class localModel():
    def __init__(self,modelPath='localModels/shop_cat3.h5'):
        self.model = tf.keras.models.load_model('localModels/shop_cat3.h5')
    
    def load_image(self,img_path, show=False):

        img = image.load_img(img_path, target_size=(150, 150))
        img_tensor = image.img_to_array(img)                    # (height, width, channels)
        img_tensor = np.expand_dims(img_tensor, axis=0)         # (1, height, width, channels), add a dimension because the model expects this shape: (batch_size, height, width, channels)
        img_tensor /= 255.                                      # imshow expects values in the range [0, 1]
        return img_tensor
    
    def run(self,pathx):
        CLASSES={0:"egg-fresh",1:"grocery",2:"meat-freah",3:"veg-fresh"}
        img=self.load_image(pathx)
        pred=self.model.predict(img)[0].tolist()
        out ={}
        for idx,prob in enumerate(pred):
            out[CLASSES[idx]]="{:.6f}".format(prob)
        print(out)
        return out

if __name__ == '__main__':
    model=localModel()
    print(json.dumps(model.run(pathx)))