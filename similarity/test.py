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
# from model import classifier
from preprocess import img_pad
from builder import ModelBuilder
from utils.region_proposal import region_search
from utils.metric import euclidean_distance
from utils.region_proposal import region_search

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
        
    tf_config.gpu_options.per_process_gpu_memory_fraction = 0.9
elif tf.__version__ == '1.11.0' or tf.__version__ == '1.13.2' or tf.__version__ == '1.12.0':
    from tensorflow import ConfigProto
    from tensorflow import InteractiveSession

    tf_config = ConfigProto(allow_soft_placement=True)
    tf_config.gpu_options.allow_growth = True
        
    tf_config.gpu_options.per_process_gpu_memory_fraction = 0.9

np_load_old = np.load
np.load = lambda *a, **k: np_load_old(*a, allow_pickle=True, **k)


def test_file(config, position, src_image_file, tgt_image_file, weights, model_file):
    
    labels_file = config['model']['labels']
    data_dir = config['test']['data_dir']
    input_width = config['model']['input_width']
    input_height = config['model']['input_height']

    #
    builder = ModelBuilder(config)
    preprocess_input = builder.preprocess_input

    image_size = (input_width,input_height)
    
    # train
    train_graph = tf.Graph()
    train_sess = tf.Session(graph=train_graph,config=tf_config)

    keras.backend.set_session(train_sess)
    with train_graph.as_default():
    
        if model_file is not None:
            model = load_model(model_file, compile=False)
        elif weights:
            model = builder.build_model()
            model.load_weights(weights)
        else:
            print('using base model')
            model = builder.build_model()
            model.summary()

        x0,y0,w,h = tuple(position)
        img_path = os.path.join(data_dir, src_image_file)
        src = cv2.imread(img_path)
        src = src[:,:,[2,1,0]]    # to RGB
        anchor = src[y0:y0+h, x0:x0+w]
        anchor = cv2.resize(anchor, (input_width, input_height))
        x_pred = np.array([anchor]).astype('float32')
#         x_pred = np.expand_dims(x_pred, axis=0)
#         x_pred = preprocess_input(x_pred)
#         print(x_pred)
#         x_pred = x_pred/255.0
        x_pred = x_pred/127.5
        x_pred = x_pred-1.
        anchor_vector = model.predict(x_pred)[0]
        
        print(img_path,x0,y0,w,h)
        print(anchor_vector)
        for v in anchor_vector:
            print(v)
        
        img_path = os.path.join(data_dir, tgt_image_file)
        tgt = cv2.imread(img_path)
        tgt = tgt[:,:,[2,1,0]]    # to RGB
        x_pred = np.array(tgt).astype('float32')
#         x_pred = preprocess_input(x_pred)
#         x_pred = x_pred/255.0        
        x_pred = x_pred/127.5
        x_pred = x_pred-1.
        
        ratio = 1.0
        max_score = 0.0
        min_dist = 100000000000.0
        x,y = 0,0
        win_size = (w,h)
        i = 0

        try:
            print(anchor.shape, x_pred.shape)
            ww,hh=704,int(np.floor(576*0.8))
            x_pred=x_pred[57:hh+57,0:ww]
            it = region_search(anchor,x_pred,int(70*ratio),0.0)
#             print('1111')
            while True:
#                 print('222')
                x1,y1,mv_size,seg = next(it)
                i = i+1
#                 print('333')
                seg = np.expand_dims(seg, axis=0)
                tgt_vector = model.predict(seg)[0]
                dist = euclidean_distance(anchor_vector, tgt_vector)
                if x0==x1 and y0==y1+57:
                    print(x0,y0,mv_size,dist)
                if dist < min_dist:
                    min_dist = dist
                    x,y,win_size = x1,y1,mv_size
        except StopIteration:
            print('region search done ', i)
            pass
        
        x,y,win_size = int(x/ratio),int(y/ratio)+57,(int(win_size[0]/ratio),int(win_size[1]/ratio))
        print(x,y,win_size,min_dist)


def main(args):
    
    position = args.position
    config_path = args.conf
    src_image_file = args.src_image_file
    tgt_image_file = args.tgt_image_file
    weights = args.weights
    model_file = args.model

    with open(config_path) as config_buffer:    
        config = json.loads(config_buffer.read())

#     print(position)
    test_file(config, position, src_image_file, tgt_image_file, weights, model_file)

def parse_arguments(argv):
    
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '-p',
        '--position', 
        nargs='+', 
        type=int,
        help='the coordinate of anchor, e.g. x,y,w,h', default=None)
    
    parser.add_argument(
        '-s',
        '--src_image_file', 
        type=str,
        help='anchor image file shortname', default=None)
    
    parser.add_argument(
        '-t',
        '--tgt_image_file', 
        type=str,
        help='base image file shortname', default=None)
    
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