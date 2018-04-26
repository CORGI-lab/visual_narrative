import time
import os
import sys
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import pickle
import json

StoredImageFeaturesFile = 'img_feats_train_0_1_2_3_10_12_fc1'
ImageDescriptionMetaDataFile = 'train.description-in-isolation.json'

class SequentialImageDescription:
    def __init__(self):
        self.image_features = dict()
        self.annotations = dict()

def loadImageFeatures():
    image_features_list = pickle.load(open(StoredImageFeaturesFile, 'rb'))
    image_features = dict()
    for j in range(len(image_features_list)):
        image_id = image_features_list[j]['id']
        features = image_features_list[j]['features']
        image_features[image_id] = features
    
    return image_features
    
def loadAnnotations(image_features):
    description_metadata = json.load(open(ImageDescriptionMetaDataFile))
    annotations = description_metadata['annotations']
    description = dict()
    annotations_len = len(annotations)
    
    for i in range(annotations_len):
        for j in range(len(annotations[i])):
            photo_id = annotations[i][j]['photo_flickr_id']
            if photo_id in image_features.keys():
                image_des = annotations[i][j]['original_text']
                image_des = image_des.lower()
                image_des = image_des.strip()
                image_des = image_des.replace(',', ' ,')
                image_des = image_des.replace('.', '')
                image_des = image_des.replace('"', ' " ')
                description[photo_id] = image_des
    
    return description
                    
    
def main():
    
    image_features = loadImageFeatures()
    annotations = loadAnnotations(image_features)

if __name__ == "__main__":
    main()
        
