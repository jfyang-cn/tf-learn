# -*- coding: utf-8 -*-
import sys,os,argparse
sys.path.append('./')
sys.path.append('../')
import numpy as np
import json
import cv2

from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array,load_img
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Layer
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.applications import inception_v3,mobilenet,vgg16,resnet50
from tensorflow import keras
from model import classifier
from preprocess import img_pad

import tensorflow as tf
print(tf.__version__)

# tensorflow allocates all gpu memory, set a limit to avoid CUDA ERROR
if tf.__version__ == '1.14.0':
    from tensorflow.compat.v1 import ConfigProto
    from tensorflow.compat.v1 import InteractiveSession

    tf_config = ConfigProto(allow_soft_placement=True)
    tf_config.gpu_options.allow_growth = True
        
    gpus = tf.config.experimental.list_physical_devices('GPU')
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
        
    tf_config.gpu_options.per_process_gpu_memory_fraction = 0.4
elif tf.__version__ == '1.11.0' or tf.__version__ == '1.13.2' or tf.__version__ == '1.12.0':
    from tensorflow import ConfigProto
    from tensorflow import InteractiveSession

    tf_config = ConfigProto(allow_soft_placement=True)
    tf_config.gpu_options.allow_growth = True
        
    tf_config.gpu_options.per_process_gpu_memory_fraction = 0.9

np_load_old = np.load
np.load = lambda *a, **k: np_load_old(*a, allow_pickle=True, **k)

def test_list_file(config, list_file, weights, model_file):
    
    labels_file = config['model']['labels']
    data_dir = config['test']['data_dir']
    input_width = config['model']['input_width']
    input_height = config['model']['input_height']
    
    # load labels
    print('load label file', labels_file)
    label_dict = np.load(labels_file).item()
    class_num = len(label_dict)
    print('class num:', class_num)
    print(label_dict)
        
    image_size = (input_width,input_height)
    
    # train
    train_graph = tf.Graph()
    train_sess = tf.Session(graph=train_graph,config=tf_config)

    keras.backend.set_session(train_sess)
    with train_graph.as_default():
    
        if model_file is not None:
            cls = load_model(model_file, compile=False)
        elif weights:
            cls = classifier(class_num,input_width,input_height)
            cls.load_weights(weights)
        else:
            print('either weights or model file should be specified.')

        with open(os.path.join(data_dir,list_file), 'r') as f:
            filelist = f.readlines()

        conf_thresh = 0.6
        conf_num = 0
        num = 0
        n_correct = 0
        for line in filelist:
            line = line.rstrip('\n')
            img_path = os.path.join(data_dir, line.split(' ')[0])
            y_true = line.split(' ')[1]

            # load dataset
            img = cv2.imread(img_path)
            img = cv2.resize(img, (input_width, input_height))
#             img = img_pad(img, input_width, input_height)
            img = img[:,:,::-1]
            x_pred = np.array([img]).astype('float32')
#             x_pred = np.array([img])/255.0

            preprocess_input = vgg16.preprocess_input

#             x_pred = np.expand_dims(img_to_array(load_img(img_path, target_size=image_size)), axis=0).astype('float32')
            x_pred = preprocess_input(x_pred)
            y_pred = cls.predict(x_pred)
#             print(img_path)
#             print(y_pred)
            
#             if (y_pred[0][0] > 0.99):
#                 y_index = 0
#             else:
#                 y_index = 1

            y_index = np.argmax(y_pred[0])
            confidence = y_pred[0][y_index]

            num = num + 1
            if confidence > conf_thresh:
                conf_num = conf_num + 1
                if y_true == label_dict[y_index]:
                    n_correct = n_correct + 1
                else:
                    print('%s,%f,%s vs %s' % (img_path, confidence, y_true, label_dict[y_index]))
#                     print(y_pred)

    print('confidence:%f' % (conf_thresh))
    print('correct/conf_num/total: %d/%d/%d' % (n_correct,conf_num,num))
    print('precision: %f,%f' % (n_correct/num, n_correct/conf_num))
    
    
def main(args):
    
    config_path = args.conf
    image_file = args.image_file
    list_file = args.list_file
    weights = args.weights
    model_file = args.model

    with open(config_path) as config_buffer:    
        config = json.loads(config_buffer.read())

    if image_file is not None:
        pass
    
    if list_file is not None:
        test_list_file(config, list_file, weights, model_file)


def parse_arguments(argv):
    
    parser = argparse.ArgumentParser()
    
    parser.add_argument(
        '-i',
        '--image_file', 
        type=str,
        help='image file path for predicting', default=None)
    
    parser.add_argument(
        '-l',
        '--list_file', 
        type=str,
        help='image shortname list txt file for testing', default=None)
    
    parser.add_argument(
        '-w',
        '--weights', 
        type=str,
        help='weights file. either weights or model should be specified.', default=None)
    
    parser.add_argument(
        '-m',
        '--model', 
        type=str,
        help='model file. either weights or model should be specified.', default=None)
    
    parser.add_argument(
        '-c',
        '--conf',
        help='path to configuration file')
        
    return parser.parse_args(argv)

if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))