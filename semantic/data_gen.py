# -*- coding: utf-8 -*-
import sys,os,argparse
sys.path.append('./')
sys.path.append('../')
import numpy as np
import cv2
from PIL import Image
from tensorflow import keras
from tensorflow.keras.preprocessing.image import img_to_array, load_img, ImageDataGenerator
from tensorflow.python.keras.utils.data_utils import Sequence

class DataGen(Sequence):
    
    def __init__(self, filepath, batch_size=8, class_num=2, target_size=(512,512), n_channels=3,
                 preprocess_input=None, with_aug=True, shuffle=True, path_dataset=None, type_gen='train', option=None):
        
        with open(filepath) as f:
            self.filelist = f.read().splitlines()
        
        self.batch_size = batch_size
        self.target_size = target_size
        self.path_dataset = path_dataset
        self.type_gen = type_gen
        self.shuffle = shuffle
        self.preprocess_input = preprocess_input
        
        self.on_epoch_end()
        
    def name(self):
        return 'drone'
        
    def __len__(self):
        return int(np.floor(len(self.filelist) / self.batch_size))

    def on_epoch_end(self):
        self.indexes = np.arange(len(self.filelist))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __getitem__(self, index):
        
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]
        # Find list of IDs
        ids = [self.filelist[k] for k in indexes]
        # Generate data
        X, Y = self.__data_generation(ids)
        if self.type_gen == 'predict':
            return X
        else:
            return X, Y
        
    def __data_generation(self, ids):
        'Generates data containing batch_size samples'
       
        X = []
        Y = []
        for i, ID in enumerate(ids):  # ID is name of file
            path_file = os.path.join(self.path_dataset, 'images/%s.jpg' % (ID))
#             print(path_file)
            img = cv2.imread(path_file)
            img = cv2.resize(img, self.target_size)
            img = img[:,:,[2,1,0]]    # to RGB
        
            if self.preprocess_input is not None:
                img = self.preprocess_input(img)

            X.append(img)
            
            path_file = os.path.join(self.path_dataset, 'labels/%s.png' % (ID))
            label = np.array(Image.open(path_file))
            Y.append(label)

        X = np.array(X)
        Y = np.array(Y)
        
        X = X/127.5
        X = X-1.

        return X, Y