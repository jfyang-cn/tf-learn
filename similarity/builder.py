# -*- coding: utf-8 -*-
from tensorflow.keras.applications import inception_v3,mobilenet,vgg16,resnet50 
from model import similarity_model
from data_gen import DataGen

class ModelBuilder():
    
    def __init__(self, config):
        
        self.config = config

        if self.config['model']['backbone'] == 'vgg16':
            self.preprocess_input = vgg16.preprocess_input
        elif self.config['model']['backbone'] == 'resnet50':
            self.preprocess_input = resnet50.preprocess_input
        elif self.config['model']['backbone'] == 'inception_v3':
            self.preprocess_input = inception_v3.preprocess_input
        else:
            self.preprocess_input = None
        
    def build_model(self):
        
        input_width   = self.config['model']['input_width']
        input_height  = self.config['model']['input_height']
        class_num     = self.config['model']['class_num']
        backbone      = self.config['model']['backbone']

        train_base    = self.config['train']['train_base']

        return similarity_model(input_width,input_height,backbone,train_base)
    
    def build_datagen(self, filepath, with_aug=True):
       
        batch_size    = self.config['train']['batch_size']
        data_dir      = self.config['train']['data_dir']
        input_width   = self.config['model']['input_width']
        input_height  = self.config['model']['input_height']
        class_num     = self.config['model']['class_num']
        
        return DataGen(filepath,batch_size,class_num,(input_width,input_height),3,
                       self.preprocess_input,with_aug,True,data_dir,'train',None)