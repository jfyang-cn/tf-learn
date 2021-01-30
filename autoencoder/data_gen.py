# -*- coding: utf-8 -*-
import os
import numpy as np
import cv2
from tensorflow import keras
from tensorflow.keras.preprocessing.image import img_to_array, load_img, ImageDataGenerator
import pandas as pd
from tensorflow.keras.applications.inception_v3 import preprocess_input

class DataGen(keras.utils.Sequence):
    
    def __init__(self, filepath, batch_size=4, target_size=(48,48)):
                
        with open(filepath, 'r') as f:
            lines = f.readlines()
        
        self.target_size = target_size
        self.data_dir = os.path.dirname(os.path.abspath(filepath))
        self.batch_size = batch_size
        self.class_num = 6
        self.fnames = [line.replace('\n', '') for line in lines]
        self.shuffle = True
        self.aug_gen = ImageDataGenerator() 
        self.on_epoch_end()
        
    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.fnames) / self.batch_size))

    def name(self):
        return 'human'
    
    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Find list of IDs
        fnames = [self.fnames[k] for k in indexes]
#         print(fnames)

#         for fname in fnames:
#             imgs = cv2.imread(os.path.join(self.data_dir, fname))
            
        data = np.array([img_to_array(load_img(os.path.join(self.data_dir, fname.replace('pick','hat')), target_size=self.target_size))
                           for fname in fnames
                      ]).astype('float32')
        data = data/255.
        
        y_data = np.array([img_to_array(load_img(os.path.join(self.data_dir, fname), target_size=self.target_size))
                           for fname in fnames
                      ]).astype('float32')
        y_data = y_data/255.

#         data = self.img_augment(data)
#         y_data = data.copy()
#         data_len = y_data.shape[0]
#         w,h=self.target_size
#         for i in range(data_len):
#             y_data[i][0:10,0:w,0:3] = (0,0,0)
#             y_data[i][h-10:h,0:w,0:3] = (0,0,0)
#             y_data[i][0:h,0:10,0:3] = (0,0,0)
#             y_data[i][0:h,w-10:w,0:3] = (0,0,0)

        return data, y_data

    def img_augment(self, data):
        name_list = ['rotate','width_shift','height_shift',
                    'brightness','flip_horizontal','width_zoom',
                    'height_zoom']
        dictkey_list = ['theta','ty','tx',
                    'brightness','flip_horizontal','zy',
                    'zx']
        # dictkey_list = ['ty','tx','zy','zx']
        random_aug = np.random.randint(2, 5) # random 2-4 augmentation method
        pick_idx = np.random.choice(len(dictkey_list), random_aug, replace=False) #

        dict_input = {}
        for i in pick_idx:
            if dictkey_list[i] == 'theta':
                dict_input['theta'] = np.random.randint(-10, 10)

#             elif dictkey_list[i] == 'ty': # width_shift
#                 dict_input['ty'] = np.random.randint(-10, 10)

#             elif dictkey_list[i] == 'tx': # height_shift
#                 dict_input['tx'] = np.random.randint(-10, 10)

            elif dictkey_list[i] == 'brightness': 
                dict_input['brightness'] = np.random.uniform(0.15,1)

            elif dictkey_list[i] == 'flip_horizontal': 
                dict_input['flip_horizontal'] = True

            elif dictkey_list[i] == 'zy': # width_zoom
                dict_input['zy'] = np.random.uniform(0.5,1.5)

            elif dictkey_list[i] == 'zx': # height_zoom
                dict_input['zx'] = np.random.uniform(0.5,1.5)

        data_len = data.shape[0]
        for i in range(data_len):
            data[i] = self.aug_gen.apply_transform(data[i], dict_input)/255.0
                
        return data
        
    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.fnames))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)
            