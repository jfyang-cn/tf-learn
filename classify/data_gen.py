# -*- coding: utf-8 -*-
import os
import numpy as np
import cv2
from tensorflow import keras
from tensorflow.keras.preprocessing.image import img_to_array, load_img, ImageDataGenerator
import pandas as pd
from tensorflow.keras.applications.inception_v3 import preprocess_input
            
class DataGen():
    
    def __init__(self, filepath, batch_size=8, target_size=(128,128)):
        
        self.df = pd.read_csv(filepath, sep=' ', header=None, dtype=str)
        self.batch_size = batch_size
        self.target_size = target_size
        self.class_num = 5
        self.datagen = ImageDataGenerator(
#             preprocessing_function=preprocess_input,# ((x/255)-0.5)*2
            rotation_range=30,
            width_shift_range=0.2,
            height_shift_range=0.2,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True,
            rescale=1/255.
            )
            
    def name(self):
        return 'helmet'   
    
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
            shuffle=False,
            class_mode="categorical", #categorical
            target_size=self.target_size)
        
        steps_per_epoch = generator.samples // self.batch_size

        return generator, steps_per_epoch
    
    def save_labels(self, filepath=None):
        
        y = self.df[1] 
        y_ohe = pd.get_dummies(y.reset_index(drop=True)).as_matrix()
        
        l = np.array(y_ohe)
        a = l==1

        b = np.argwhere(a == True)
        c = b[:,1]

        iters = y.keys()
        labels = y[iters].get_values()

        # construct dict
        label_dicts = {}
        for i in range(len(labels)):
            label_dicts[c[i]] = labels[i]

        if filepath is None:
            filepath = self.name()+'_labels.npy'
            
        np.save(filepath, label_dicts)
        print(filepath, ' saved.')
