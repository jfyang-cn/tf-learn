# -*- coding: utf-8 -*-
import sys,os,argparse
sys.path.append('./')
sys.path.append('../')
import numpy as np
import cv2
from tensorflow import keras
from tensorflow.keras.preprocessing.image import img_to_array, load_img, ImageDataGenerator
from tensorflow.python.keras.utils.data_utils import Sequence
from utils.data_helper import readfile_to_dict

class DataGen(Sequence):
    
    def __init__(self, filepath, batch_size=8, class_num=2, target_size=(224,224), n_channels=3,
                 preprocess_input=None, with_aug=True, shuffle=True, path_dataset=None,
                type_gen='train', option=None):
        
        data_dict = readfile_to_dict(filepath)
        data_keys = list(data_dict.keys())
        
        self.shuffle = shuffle
        self.batch_size = batch_size
        self.labels = data_dict
        self.list_IDs = data_keys
        self.target_size = target_size
        self.path_dataset = path_dataset
        self.type_gen = type_gen
        self.option = option
        self.n_channels = n_channels
        self.class_num = class_num
        self.preprocess_input = preprocess_input

        self.aug_gen = ImageDataGenerator()
        self.steps_per_epoch = len(self.list_IDs) // self.batch_size
        print("all:", len(self.list_IDs), " batch per epoch", self.steps_per_epoch)
                
        ww,hh=704,np.floor(576*0.8)
        x0,y0=0,int(np.floor(576*0.1))
#         ww,hh=target_size[0]+19+x0,target_size[1]+19+y0
        step=70
        self.pos = []
        while True:
            if y0+target_size[1]>hh:
                break
            x0=0
            while True:
                if x0+target_size[0]>ww:
                    break
                self.pos.append((x0,y0))
                x0+=step                                
            y0+=step
        
        self.on_epoch_end()
 
    def __len__(self):
        'Denotes the number of batches per epoch'
#         print(int(np.floor(len(self.list_IDs) / self.batch_size)), len(self.pos))
        return int(np.floor(len(self.list_IDs) / self.batch_size))*len(self.pos)
    
    def on_epoch_end(self):
        'Updates indexes after each epoch'        
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)
    
    def __getitem__(self, index):
        'Generate one batch of data'
        
        index_i = index//len(self.pos)
        pos_i = index%len(self.pos)
        
        # Generate indexes of the batch
        indexes = self.indexes[index_i*self.batch_size:(index_i+1)*self.batch_size]
        # Find list of IDs
        ids = [self.list_IDs[k] for k in indexes]
        # Generate data
        X, y = self.__data_generation(ids, self.pos[pos_i])
        if self.type_gen == 'predict':
            return X
        else:
            return X, y

    def sequence_augment(self, sequence):
        
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

            elif dictkey_list[i] == 'ty': # width_shift
                dict_input['ty'] = np.random.randint(-5, 5)

            elif dictkey_list[i] == 'tx': # height_shift
                dict_input['tx'] = np.random.randint(-5, 5)

            elif dictkey_list[i] == 'brightness': 
                dict_input['brightness'] = np.random.uniform(0.15,1)

            elif dictkey_list[i] == 'flip_horizontal': 
                dict_input['flip_horizontal'] = True

            elif dictkey_list[i] == 'zy': # width_zoom
                dict_input['zy'] = np.random.uniform(0.5,1.5)

            elif dictkey_list[i] == 'zx': # height_zoom
                dict_input['zx'] = np.random.uniform(0.5,1.5)

        len_seq = sequence.shape[0]
        for i in range(len_seq):
            sequence[i] = self.aug_gen.apply_transform(sequence[i], dict_input)
                
        return sequence
    
    def __data_generation(self, ids, pos):
        'Generates data containing batch_size samples'
        # Initialization
        Y = np.zeros((self.batch_size, self.class_num))
        
        X = []
        x = None
        y = None
        for i, ID in enumerate(ids):  # ID is name of file
            path_file = os.path.join(self.path_dataset, ID)
#             print(path_file)
            img = cv2.imread(path_file)
            
            # 取图像中间的0.8部分
            hh,ww,_ = img.shape
            xx,yy = int(0),int(hh*0.1)
            ww,hh = int(ww),int(hh*0.8)
            img = img[yy:yy+hh,xx:xx+ww]
    
            img = img[:,:,[2,1,0]]    # to RGB
#             if self.preprocess_input is not None:
#                 img = self.preprocess_input(img)

            if i < len(ids)/2:
                x,y = pos[0],pos[1]
            else:
                # 随机切一块
                h,w,n = img.shape
#                 print(img.shape)
                x = np.random.randint(0, w-self.target_size[0])
                y = np.random.randint(0, h-self.target_size[1])

            (w,h) = self.target_size
#             print(x,y,w,h)
            X.append(img[y:y+h,x:x+w])
#             print(x,y,w,h)

        X = np.array(X)
        if self.type_gen =='train':
            X = self.sequence_augment(X)    # apply the same rule
        else:
            X = X
            
#         X = X/255.
        X = X/127.5
        X = X-1.

        return X, Y