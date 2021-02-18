import numpy as np
from tensorflow import keras
import cv2
import os
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from data_helper import calculateRGBdiff, readfile_to_dict
from tensorflow.python.keras.utils.data_utils import Sequence

class DataGen(Sequence):
    'Generates data for Keras'
    def __init__(self, filepath, batch_size=32, class_num=2, dim=(32,32), 
                n_channels=1,n_sequence=4, preprocess_input=None, with_aug=True, shuffle=True, path_dataset=None,
                type_gen='train', option=None):
        'Initialization'
        
        data_dict = readfile_to_dict(filepath)
        data_keys = list(data_dict.keys())
        
        self.dim = dim
        self.batch_size = batch_size
        self.class_num = class_num
        self.labels = data_dict
        self.list_IDs = data_keys
        self.n_channels = n_channels
        self.n_sequence = n_sequence    # get n_sequence diff image
        self.shuffle = shuffle
        self.path_dataset = path_dataset
        self.type_gen = type_gen
        self.option = option
        self.aug_gen = ImageDataGenerator() 
        self.steps_per_epoch = len(self.list_IDs) // self.batch_size
        print("all:", len(self.list_IDs), " batch per epoch", self.steps_per_epoch)
        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.list_IDs) / self.batch_size))

    def on_epoch_end(self):
        'Updates indexes after each epoch'        
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]
        # Find list of IDs
        list_IDs_temp = [self.list_IDs[k] for k in indexes]
        # Generate data
        X, y = self.__data_generation(list_IDs_temp)
        if self.type_gen == 'predict':
            return X
        else:
            return X, y

    def get_sampling_frame(self, len_frames):   
        '''
        Sampling n_sequence frame from video file
        Input: 
            len_frames -- number of frames that this video have
        Output: 
            index_sampling -- n_sequence frame indexs from sampling algorithm 
        '''             
        # Define maximum sampling rate
        random_sample_range = 9
        if random_sample_range*self.n_sequence > len_frames:
            random_sample_range = len_frames//self.n_sequence
        # Randomly choose sample interval and start frame
#         print(random_sample_range)
        sample_interval = np.random.randint(1, random_sample_range + 1)
        start_i = np.random.randint(0, len_frames - sample_interval * self.n_sequence + 1)
        
        # Get n_sequence index of frames
        index_sampling = []
        end_i = sample_interval * self.n_sequence + start_i
        for i in range(start_i, end_i, sample_interval):
            if len(index_sampling) < self.n_sequence:
                index_sampling.append(i)
        
        return index_sampling

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
                dict_input['ty'] = np.random.randint(-30, 30)

            elif dictkey_list[i] == 'tx': # height_shift
                dict_input['tx'] = np.random.randint(-15, 15)

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
            sequence[i] = self.aug_gen.apply_transform(sequence[i],dict_input)
                
        return sequence
    
    def __data_generation(self, list_IDs_temp):
        'Generates data containing batch_size samples'
        # Initialization
        X = np.empty((self.batch_size, self.n_sequence, *self.dim, self.n_channels)) # X : (n_samples, *dim, n_channels)
        Y = np.zeros((self.batch_size, self.class_num))
        
        for i, ID in enumerate(list_IDs_temp):  # ID is name of file
            path_file = os.path.join(self.path_dataset,ID)            
            cap = cv2.VideoCapture(path_file)
            length_file = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) # get how many frames this video have
#             print(path_file,length_file)
            index_sampling = self.get_sampling_frame(length_file) # get sampling index
            for j, n_pic in enumerate(index_sampling):
                cap.set(cv2.CAP_PROP_POS_FRAMES, n_pic) # jump to that index
                ret, frame = cap.read()
                if ret is True:
                    new_image = cv2.resize(frame, self.dim)   
                    X[i,j,:,:,:] = new_image
                else:
                    print('read file ', path_file, 'error', length_file, n_pic)

            if self.type_gen =='train':
                X[i,] = self.sequence_augment(X[i,])    # apply the same rule
            else:
                X[i,] = X[i,]
    
            if self.option == 'RGBdiff':
                X[i,] = calculateRGBdiff(X[i,])

            Y[i][self.labels[ID]-1] = 1.0
            cap.release()

        return X, Y
