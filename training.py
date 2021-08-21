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
#from IPython.display import Image
from keras.optimizers import Adam
import ssl
ssl._create_default_https_context = ssl._create_unverified_context
from keras.callbacks import ModelCheckpoint
import keras_metrics


#mobile = keras.applications.mobilenet.MobileNet()


class Mobelenet_classifier():
    def __init__(self,frozenLayers=80,dataPath="finalDataNew",modelPath=None):
        if modelPath is None:
            # Get the base mobilenet model and don't include the last layers
            # we want custom layers
            base_model=MobileNet(weights='imagenet',include_top=False) 
            x=base_model.output
            for layer in base_model.layers:
                layer.trainable=False
            
            # Do the average pooling -output should be 1024 layers
            x=GlobalAveragePooling2D()(x)
            x=Dense(1024,activation='relu')(x) #we add dense layers so that the model can learn more complex functions and classify for better results.
            x=Dense(1024,activation='relu')(x) #dense layer 2
            x=Dense(512,activation='relu')(x) #dense layer 3
            preds=Dense(4,activation='softmax')(x) #final layer with softmax activation
            self.model=Model(inputs=base_model.input,outputs=preds)
            # print("no:of layers",len(self.model.layers))
            # for layer in self.model.layers:
            #     layer.trainable=False
            # # or if we want to set the first 20 layers of the network to be non-trainable
            # for layer in self.model.layers[:frozenLayers]:
            #     layer.trainable=False
            # for layer in self.model.layers[frozenLayers:]:
            #     layer.trainable=True
        else:
            self.model = keras.models.load_model(modelPath)
        
        # Create train data generator   
        self.train_datagen=ImageDataGenerator(preprocessing_function=preprocess_input,validation_split=0.25)
        
        self.train_generator=self.train_datagen.flow_from_directory(dataPath,
                                                 target_size=(224,224),
                                                 color_mode='rgb',
                                                 batch_size=8,
                                                 class_mode='categorical',
                                                 subset='training',
                                                 shuffle=True)
        # Create validation data generator
        self.val_generator = self.train_datagen.flow_from_directory(
                                                dataPath, # same directory as training data
                                                target_size=(224,224),
                                                color_mode='rgb',
                                                batch_size=8,
                                                class_mode='categorical',
                                                subset='validation',
                                                shuffle=True
                                                )
        
        # Checkpoint to store
        self.checkpoint = ModelCheckpoint("model/model_shopcat.hdf5", monitor='loss', verbose=1,save_best_only=True, mode='auto', period=1)
        
    # Method to train    
    def train(self):
        self.model.compile(optimizer='Adam',loss='categorical_crossentropy',metrics=['accuracy',keras_metrics.precision(), keras_metrics.recall()])
        # Adam optimizer
        # loss function will be categorical cross entropy
        # evaluation metric will be accuracy,precision and recall
        self.model.fit(self.train_generator,
                        steps_per_epoch=self.train_generator.n//self.train_generator.batch_size,
                        validation_data = self.val_generator, 
                        validation_steps = self.val_generator.samples //self.val_generator.batch_size,
                        epochs=20
                        )
        self.model.save('model/shop_cat4.h5')
        
        

Mobilenet= Mobelenet_classifier(frozenLayers=80,dataPath='train_data')#modelPath='model/shop_cat.h5')
Mobilenet.train()

       
       
