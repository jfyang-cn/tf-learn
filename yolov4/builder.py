# -*- coding: utf-8 -*-
# from tensorflow.keras.applications import inception_v3,mobilenet,vgg16,resnet50 
import numpy as np
from tensorflow import keras
import tensorflow.keras.backend as K
from tensorflow.keras.layers import Input, Lambda
from tensorflow.keras.models import Model
from nets.loss import yolo_loss
from nets.yolo4 import yolo_body

import sys
sys.path.append('../')
from datagen.coco2017_gen import DataGen

class ModelBuilder():
    
    def __init__(self, config):
        
        self.config = config
        self.preprocess_input = None
        anchors = [12, 16,  19, 36,  40, 28,  36, 75,  76, 55,  72, 146,  142, 110,  192, 243,  459, 401]
        anchors = np.array(anchors).reshape(-1, 2)
        self.anchors = anchors
        
    def build_model(self):
        
        input_width   = self.config['model']['input_width']
        input_height  = self.config['model']['input_height']
        input_depth   = self.config['model']['input_depth']
        class_num     = self.config['model']['class_num']
        backbone      = self.config['model']['backbone']

        train_base    = self.config['train']['train_base']
        weights_path = self.config['train']['pretrained_weights']

        num_anchors = len(self.anchors)
        label_smoothing = 0
        normalize = False

        K.clear_session()
        #------------------------------------------------------#
        #   创建yolo模型
        #------------------------------------------------------#
        image_input = Input(shape=(input_width, input_height, 3))
        print('Create YOLOv4 model with {} anchors and {} classes.'.format(num_anchors, class_num))
        model_body = yolo_body(image_input, num_anchors//3, class_num)
#         model_body.summary()

        #------------------------------------------------------#
        #   载入预训练权重
        #------------------------------------------------------#
#         print('Load weights {}.'.format(weights_path))
#         model_body.load_weights(weights_path, by_name=True, skip_mismatch=True)

        #------------------------------------------------------#
        #   在这个地方设置损失，将网络的输出结果传入loss函数
        #   把整个模型的输出作为loss
        #------------------------------------------------------#
        y_true = [Input(shape=(input_height//{0:32, 1:16, 2:8}[l], input_width//{0:32, 1:16, 2:8}[l], \
            num_anchors//3, class_num+5)) for l in range(3)]
        loss_input = [*model_body.output, *y_true]
        model_loss = Lambda(yolo_loss, output_shape=(1,), name='yolo_loss',
            arguments={'anchors': self.anchors, 'num_classes': class_num, 'ignore_thresh': 0.5, 
                'label_smoothing': label_smoothing, 'normalize': normalize})(loss_input)

        model = Model([model_body.input, *y_true], model_loss)

        return model
    
    def build_train_datagen(self):

        file_list      = self.config['train']['file_list']
        batch_size    = self.config['train']['batch_size']
        data_dir      = self.config['train']['data_dir']
        input_width   = self.config['model']['input_width']
        input_height  = self.config['model']['input_height']
        input_depth   = self.config['model']['input_depth']
        class_num     = self.config['model']['class_num']

        params = {
          'batch_size':batch_size,
          'dim': (input_width,input_height),
          'batch_size': batch_size,
          'class_num':class_num,
          'n_channels': 3,
          'preprocess_input':self.preprocess_input,
          'path_dataset': data_dir,
          'with_aug':True,
          'option': None,
          'shuffle': True,
          'type_gen': 'train'}
        
        return DataGen(file_list, self.anchors, **params)

    def build_valid_datagen(self):

        file_list      = self.config['valid']['file_list']
        batch_size    = self.config['train']['batch_size']
        data_dir      = self.config['valid']['data_dir']
        input_width   = self.config['model']['input_width']
        input_height  = self.config['model']['input_height']
        input_depth   = self.config['model']['input_depth']
        class_num     = self.config['model']['class_num']
        
        params = {
          'batch_size':batch_size,
          'dim': (input_width,input_height),
          'batch_size': batch_size,
          'class_num':class_num,
          'n_channels': 3,
          'preprocess_input':self.preprocess_input,
          'path_dataset': data_dir,
          'with_aug':False,
          'option': None,
          'shuffle': True,
          'type_gen': 'valid'}

        return DataGen(file_list, self.anchors, **params)