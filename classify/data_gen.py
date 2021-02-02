# -*- coding: utf-8 -*-
import os
import numpy as np
import cv2
import pandas as pd
from tensorflow import keras
from tensorflow.keras.preprocessing.image import img_to_array, load_img, ImageDataGenerator
from tensorflow.keras.applications import inception_v3,mobilenet,vgg16,resnet50

class DataGen():
    
    def __init__(self, filepath, batch_size=8, target_size=(224,224), preprocess_input=None, with_aug=True):
        
        self.df = pd.read_csv(filepath, sep=' ', header=None, dtype=str)
        self.batch_size = batch_size
        self.target_size = target_size
        
        y = self.df[1] 
        y_ohe = pd.get_dummies(y.reset_index(drop=True)).values
        
        l = np.array(y_ohe)
        a = l==1

        b = np.argwhere(a == True)
        c = b[:,1]

        iters = y.keys()
        labels = y[iters].to_numpy()

        # construct dict
        label_dicts = {}
        for i in range(len(labels)):
            label_dicts[c[i]] = labels[i]
        self.label_dicts = label_dicts
        self.class_num = len(label_dicts)
        
        if with_aug:
            self.datagen = ImageDataGenerator(
                preprocessing_function=preprocess_input,
                rotation_range=30,
                width_shift_range=0.2,
                height_shift_range=0.2,
                shear_range=0.2,
                zoom_range=0.2,
#                 fill_mode='constant', 
#                 cval=0.0,
                horizontal_flip=True,
#                 rescale = 1./255
                )    
        else:
            self.datagen = ImageDataGenerator(
                preprocessing_function=preprocess_input
#                 rescale = 1./255
            )            
    
    def from_frame(self, directory=None):
            
        self.df['image_path'] = self.df[0]
        self.df['label'] = self.df[1]
    
        generator = self.datagen.flow_from_dataframe(
            dataframe=self.df,
            directory=directory,
            x_col='image_path',
            y_col='label',
            batch_size=self.batch_size,
#             seed=42,
#             shuffle=True,
            seed=1,
            shuffle=True,
            class_mode="categorical", #categorical
            target_size=self.target_size)
        
        steps_per_epoch = generator.samples // self.batch_size

        return generator, steps_per_epoch
    
    def save_labels(self, filepath=None):

        if filepath is None:
            filepath = self.name()+'_labels.npy'
            
        np.save(filepath, self.label_dicts)
        print(filepath, ' saved.')
