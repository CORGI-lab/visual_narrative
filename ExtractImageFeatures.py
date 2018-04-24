from keras.applications.vgg16 import VGG16, decode_predictions
from keras.models import Model
import os
from tensorflow.python.platform import gfile
from keras.applications.vgg19 import preprocess_input
import numpy as np
import pickle
from keras.preprocessing import image
import cv2
import logging

logFile = 'feature_extraction.log'
logging.basicConfig(filename=logFile,format='%(asctime)s %(levelname)s: %(message)s',
                    datefmt='%m/%d/%Y %I:%M:%S %p', level=logging.DEBUG)

class ExtractImageFeatures:
    
    def __init__(self):
        vgg16_model = VGG16(weights='vgg16_weights_tf_dim_ordering_tf_kernels.h5')
        self.model = Model(inputs=vgg16_model.input, outputs=vgg16_model.get_layer('fc1').output)
    
    def extractSingleImageFeatures(self, imagePath):
        id_features = {}
        if not gfile.Exists(imagePath):
            logging.error('File does not exist %s', imagePath)
            raise Exception('File does not exist %s', imagePath)
        else:
            img = cv2.imread(imagePath)
            resized_img = cv2.resize(img, (224,224))
            x = np.expand_dims(resized_img, axis=0)
            features = self.model.predict(x)
            image_name = imagePath.split(os.sep)[0]
            id_features = {'features':features,'id':image_name.split('.')[0]}
            
        return id_features

    def extractAllImagesFeatures(self, imagesDirectory):
        isDir = os.path.isdir(imagesDirectory)
        if isDir == False:
            msg = "Image directory does not exist"
            logging.error(msg)
            raise Exception(msg)
        image_list = os.listdir(imagesDirectory)
        image_features = []
        for ind, image_name in enumerate(image_list):
            if (ind%50 == 0):
                logging.info('%d Processing %s...'  %(ind, image_name))
                print('%d Processing %s...'  %(ind, image_name))
            try:
                img_path = os.path.join(imagesDirectory,image_name)
                features = self.extractSingleImageFeatures(img_path)
                image_features.append(features)
            except Exception as e:
                msg = 'execption:' + image_name + ": " + e
                logging.error(msg)
                print (msg)
        
        return image_features
                