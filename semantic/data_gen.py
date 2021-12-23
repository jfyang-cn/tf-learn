# -*- coding: utf-8 -*-
import sys,os,argparse
sys.path.append('./')
sys.path.append('../')
import numpy as np
import cv2
from PIL import Image
import tensorflow as tf
import gc
from tensorflow import keras
from tensorflow.keras.preprocessing.image import img_to_array, load_img, ImageDataGenerator
from tensorflow.keras.utils import Sequence

class DataGen(Sequence):
    
    def __init__(self, filepath, batch_size=4, class_num=3, target_size=(256,256), n_channels=3,
                 preprocess_input=None, with_aug=True, shuffle=True, path_dataset=None, type_gen='train', option=None):
        
        with open(filepath) as f:
            self.filelist = f.read().splitlines()
        
        self.batch_size = batch_size
        self.target_size = target_size
        self.path_dataset = path_dataset
        self.type_gen = type_gen
        self.shuffle = shuffle
        self.class_num = class_num
        self.preprocess_input = preprocess_input
        self.X = np.zeros((batch_size, target_size[1], target_size[0], n_channels), dtype='float32')
        self.Y = np.zeros((batch_size, target_size[1]*target_size[0], 1), dtype='float32')
        
        self.on_epoch_end()
        
    def name(self):
        return 'drone'
        
    def __len__(self):
        return int(np.floor(len(self.filelist) / self.batch_size))

    def on_epoch_end(self):
        self.indexes = np.arange(len(self.filelist))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)
        tf.keras.backend.clear_session()
        gc.collect()

    def __getitem__(self, index):
        
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]
        # Find list of IDs
        ids = [self.filelist[k] for k in indexes]

        # Generate data
        self.__data_generation(ids)
        if self.type_gen == 'predict':
            return self.X
        else:
            return self.X, self.Y
        
    def __data_generation(self, ids):
        'Generates data containing batch_size samples'

        for i, ID in enumerate(ids):  # ID is name of file
            path_file = os.path.join(self.path_dataset, 'images/%s.jpg' % (ID))
#             print(path_file)
            img = cv2.imread(path_file)
            img = cv2.resize(img, self.target_size)
            img = img[:,:,[2,1,0]]    # to RGB
        
            if self.preprocess_input is not None:
                img = self.preprocess_input(img)
            
            self.X[i] = np.array(img)
            
            path_file = os.path.join(self.path_dataset, 'labels/%s.png' % (ID))
#             label = cv2.imread(path_file, cv2.IMREAD_GRAYSCALE)
            
            f = open(path_file, 'rb')
            label = Image.open(f)
            label.load()
            label = np.array(label)

            label = cv2.resize(label, self.target_size, interpolation = cv2.INTER_NEAREST)         
#             label = cv2.resize(label, self.target_size)
            y = label.flatten()
            y[y>(self.class_num-1)] = 0
#             y = np.eye(3)[y]
            self.Y[i] = np.expand_dims(y, -1)
            f.close()
            img = None
            f = None

        self.X = self.X/127.5
        self.X = self.X-1.

        return