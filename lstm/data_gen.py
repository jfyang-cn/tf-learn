import numpy as np
from tensorflow import keras
import cv2
import os
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from data_helper import calculateRGBdiff
from tensorflow.python.keras.utils.data_utils import Sequence

class DataGenerator(Sequence):
    'Generates data for Keras'
    def __init__(self, list_IDs, labels, batch_size=32, dim=(32,32), 
                n_channels=1,n_sequence=4, shuffle=True, path_dataset=None,
                type_gen='train', option=None):
        'Initialization'
        self.dim = dim
        self.batch_size = batch_size
        self.labels = labels
        self.list_IDs = list_IDs
        self.n_channels = n_channels
        self.n_sequence = n_sequence+1    # get n_sequence diff image
        self.shuffle = shuffle
        self.path_dataset = path_dataset
        self.type_gen = type_gen
        self.option = option
        self.aug_gen = ImageDataGenerator() 
        print("all:", len(self.list_IDs), " batch per epoch", int(np.floor(len(self.list_IDs) / self.batch_size)) )
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
        D, X, y = self.__data_generation(list_IDs_temp)
        if self.type_gen == 'predict':
            return {'input_1':D, 'input_2':X}
        elif self.type_gen == 'train' or self.type_gen == 'test':
            return ({'input_1':D, 'input_2':X}, y)
        elif self.type_gen == 'quantize':
            return np.concatenate((D, X), axis=3), y

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
        sample_interval = np.random.randint(3, random_sample_range + 1)
        start_i = np.random.randint(0, len_frames - sample_interval * self.n_sequence + 1)
        
        # Get n_sequence index of frames
        index_sampling = []
        end_i = sample_interval * self.n_sequence + start_i
        for i in range(start_i, end_i, sample_interval):
            if len(index_sampling) < self.n_sequence:
                index_sampling.append(i)
        
        return index_sampling

    def sequence_augment(self, sequence, rgb_img=None):
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
                dict_input['ty'] = np.random.randint(-60, 60)

            elif dictkey_list[i] == 'tx': # height_shift
                dict_input['tx'] = np.random.randint(-30, 30)

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
        
        if rgb_img is not None:
            rgb_img = self.aug_gen.apply_transform(rgb_img,dict_input)
        
        return sequence

    def __data_generation2(self, list_IDs_temp):
        'Generates data containing batch_size samples'
        # Initialization
        D = np.empty((self.batch_size, *self.dim, self.n_channels)) # X : (n_samples, *dim, n_channels)
        X = np.empty((self.batch_size, self.n_sequence, *self.dim)) # X : (n_samples, n_sequence, *dim)
        Y = np.empty((self.batch_size), dtype=int)

        for i, ID in enumerate(list_IDs_temp):  # ID is name of file
            path_file = self.path_dataset + ID + '.mp4'
            cap = cv2.VideoCapture(path_file)
            length_file = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) # get how many frames this video have
            index_sampling = self.get_sampling_frame(length_file) # get sampling index
            new_image = None
            for j, n_pic in enumerate(index_sampling):
                cap.set(cv2.CAP_PROP_POS_FRAMES, n_pic) # jump to that index
                ret, frame = cap.read()
                new_image = cv2.resize(frame, self.dim)
                gray = cv2.cvtColor(new_image, cv2.COLOR_BGR2GRAY)
                X[i,j,:,:] = gray
                D[i,:,:,:] = new_image

            if self.type_gen =='train':
#                 X[i,] = self.sequence_augment(X[i,])/255.0
                X[i,] = X[i,]/255.0
                D[i,] = D[i,]/255.0
            else:
                X[i,] = X[i,]/255.0
                D[i,] = D[i,]/255.0

            if self.option == 'RGBdiff':
                X[i,] = calculateRGBdiff(X[i,])
#                 X = np.absolute(X)
#                 X[i,] = np.delete(X[i,], 0, axis=0)    # remove first image

            Y[i] = self.labels[ID]
            cap.release()

        X = np.rollaxis(X, 1, 4)
        return D,X[:,:,:,1:self.n_sequence],Y  # remove first image
    
    
    def __data_generation(self, list_IDs_temp):
        'Generates data containing batch_size samples'
        # Initialization
        X = np.empty((self.batch_size, self.n_sequence, *self.dim, self.n_channels)) # X : (n_samples, *dim, n_channels)
        Y = np.empty((self.batch_size), dtype=int)
        
        D = np.empty((self.batch_size, *self.dim, self.n_channels)) # X : (n_samples, *dim, n_channels)
        S = np.empty((self.batch_size, self.n_sequence, *self.dim)) # X : (n_samples, n_sequence, *dim)

        for i, ID in enumerate(list_IDs_temp):  # ID is name of file
            path_file = os.path.join(self.path_dataset,ID+'.mp4')
            cap = cv2.VideoCapture(path_file)
            length_file = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) # get how many frames this video have
            index_sampling = self.get_sampling_frame(length_file) # get sampling index
            for j, n_pic in enumerate(index_sampling):
                cap.set(cv2.CAP_PROP_POS_FRAMES, n_pic) # jump to that index
                ret, frame = cap.read()
                new_image = cv2.resize(frame, self.dim)   
                X[i,j,:,:,:] = new_image

            if self.type_gen =='train':
                X[i,] = self.sequence_augment(X[i,])    # apply the same rule
            else:
                X[i,] = X[i,]
    
            n = len(X[i,])
            
            # always get the last image
            D[i,:,:,:] = X[i,n-1,:,:,:]
        
            # convert to gray image and normalize to 0~1
            for k in range(n):
                gray = X[i,k,:,:]
                gray = np.array(gray, dtype=np.float32)
                gray = cv2.cvtColor(gray, cv2.COLOR_BGR2GRAY)
                S[i,k,:,:] = gray/255.0
            D[i,:,:,:] = D[i,:,:,:]/255.0
                
            if self.option == 'RGBdiff':
                S[i,] = calculateRGBdiff(S[i,])

            Y[i] = self.labels[ID]
            cap.release()

        S = np.rollaxis(S, 1, 4)
        return D,S[:,:,:,1:self.n_sequence],Y  # remove first image

