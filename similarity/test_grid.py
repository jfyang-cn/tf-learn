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
from utils.region_proposal import slide_window
from utils.metric import euclidean_distance
from sklearn import metrics as mr

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
#             model.summary()
        
        img_path = os.path.join(data_dir, src_image_file)
        src = cv2.imread(img_path)
        src = src[:,:,[2,1,0]]    # to RGB
        src = np.array(src).astype('float32')
        src = src/127.5
        src = src-1.
                
        img_path = os.path.join(data_dir, tgt_image_file)
        tgt = cv2.imread(img_path)
        tgt = tgt[:,:,[2,1,0]]    # to RGB
        tgt = np.array(tgt).astype('float32')
        tgt = tgt/127.5
        tgt = tgt-1.
        
        out_img = cv2.imread(img_path)
        
        ratio=1.0
        i = 0
        try:
            win_size = (112,112)
            tgt_size = (704,int(np.floor(576*0.8)))
            it = slide_window(win_size,tgt_size,int(100*ratio),0.0)
            while True:
                x1,y1,mv_size = next(it)
                i = i+1
                
                seg = src[y1+57:y1+57+win_size[1],x1:x1+win_size[0]]
                seg = np.expand_dims(seg, axis=0)                
                src_vector = model.predict(seg)[0]
                seg = tgt[y1+57:y1+57+win_size[1],x1:x1+win_size[0]]
                seg = np.expand_dims(seg, axis=0)                
                tgt_vector = model.predict(seg)[0]

                dist = euclidean_distance(src_vector, tgt_vector)
                print(i,x1,y1+57,dist)
                if dist < 0.15:
                    cv2.rectangle(out_img, (x1, y1+57), (x1+mv_size[0], y1+57+mv_size[1]), (0, 255, 0), 2)
#                 score = mr.mutual_info_score(src_vector, tgt_vector)
#                 print(i,x1,y1+57,score)                
#                 if score > 4.:
#                     cv2.rectangle(out_img, (x1, y1+57), (x1+mv_size[0], y1+57+mv_size[1]), (0, 255, 0), 2)

        except StopIteration:
            print('region search done ', i)
            pass
        
        cv2.imwrite('img_grid.jpg', out_img)

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