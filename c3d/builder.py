# -*- coding: utf-8 -*-
# from tensorflow.keras.applications import inception_v3,mobilenet,vgg16,resnet50 
from model import c3d
from data_gen import DataGen

class ModelBuilder():
    
    def __init__(self, config):
        
        self.config = config
        self.preprocess_input = None
        
    def build_model(self):
        
        input_width   = self.config['model']['input_width']
        input_height  = self.config['model']['input_height']
        input_depth   = self.config['model']['input_depth']
        class_num     = self.config['model']['class_num']
        backbone      = self.config['model']['backbone']

        train_base    = self.config['train']['train_base']

        return c3d(class_num,input_width,input_height,input_depth,backbone,train_base)
    
    def build_train_datagen(self, filepath, with_aug=True):

        batch_size    = self.config['train']['batch_size']
        data_dir      = self.config['train']['data_dir']
        input_width   = self.config['model']['input_width']
        input_height  = self.config['model']['input_height']
        input_depth   = self.config['model']['input_depth']

        return DataGen(filepath,batch_size,(input_width,input_height),3,input_depth,
                       self.preprocess_input,with_aug,True,data_dir,'train',None)
    
    def build_valid_datagen(self, filepath, with_aug=True):

        batch_size    = self.config['train']['batch_size']
        data_dir      = self.config['valid']['data_dir']
        input_width   = self.config['model']['input_width']
        input_height  = self.config['model']['input_height']
        input_depth   = self.config['model']['input_depth']

        return DataGen(filepath,batch_size,(input_width,input_height),3,input_depth,
                       self.preprocess_input,with_aug,False,data_dir,'valid',None)