# -*- coding: utf-8 -*-
import os, sys, datetime, argparse
import numpy as np
import cv2
from tensorflow.keras.callbacks import ModelCheckpoint,TensorBoard
from tensorflow.keras import callbacks
import tensorflow.keras.backend as K
from data_gen import DataGen
from model import autoencoder

import tensorflow as tf
print(tf.__version__)

# Try to enable Auto Mixed Precision on TF 2.0
# os.environ['TF_ENABLE_AUTO_MIXED_PRECISION'] = '1'
# os.environ['TF_AUTO_MIXED_PRECISION_GRAPH_REWRITE_IGNORE_PERFORMANCE'] = '1'
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

def optimize_tf_gpu(tf, K):
    if tf.__version__.startswith('2'):
        gpus = tf.config.experimental.list_physical_devices('GPU')
        if gpus:
            try:
                # Currently, memory growth needs to be the same across GPUs
                for gpu in gpus:
                    tf.config.experimental.set_virtual_device_configuration(gpu, [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=6144)])
#                     tf.config.experimental.set_memory_growth(gpu, True)
            except RuntimeError as e:
                # Memory growth must be set before GPUs have been initialized
                print(e)
    else:
        config = tf.ConfigProto()
        config.gpu_options.allow_growth=True   #dynamic alloc GPU resource
        config.gpu_options.per_process_gpu_memory_fraction = 0.9  #GPU memory threshold 0.3
        session = tf.Session(config=config)

        # set session
        K.set_session(session)
        
optimize_tf_gpu(tf, K)


obj_detect, segment_image, colormap = None, None, None

class_num = 2
color_list = [[255,215,0],[0,215,255],[215,0,96],[0,0,255],[16,96,255],[96,255,16],[255,16,96]]

def main(args):

    basedir = args.basedir
    savedir = args.savedir
    filelist = args.filelist
    weights = args.weight_path
    
    if filelist is not None:
        with open(filelist) as f:
            filenames = f.read().splitlines()
    
    if not os.path.exists(savedir):
        os.makedirs(savedir)
    
    model = autoencoder()
    model.load_weights(weights)
    
    for filename in filenames:
        filepath = os.path.join(basedir, 'images/%s.jpg' % (filename))
        print(filepath)
        img = cv2.imread(filepath)
        img = cv2.resize(img, (256,256))
        img = img[:,:,[2,1,0]]    # to RGB
        
        x = np.array(img)
        X = np.expand_dims(x, 0)
        X = X/127.5
        X = X-1.
        
        y_pred = model.predict(X)
        y_pred = np.argmax(y_pred, axis=-1)        
        pred_mask = y_pred.reshape((256,256))
        pred_mask = pred_mask.astype('int')
        h, w = pred_mask.shape
        ori = cv2.resize(img, (w, h))

        for i in range(class_num+1):
            mask = (pred_mask == i).astype(np.bool)
            color_mask = np.array(color_list[i], dtype=np.uint8)
            alpha = 0.55
            ori[mask] = ori[mask] * (1 - alpha) + color_mask * (alpha)

        filesave = os.path.join(savedir, '%s.jpg' % (filename))
        cv2.imwrite(filesave, ori)
                
def parse_arguments(argv):
    
    parser = argparse.ArgumentParser()
    
    parser.add_argument(
        '-d',
        '--savedir',
        help='dir path to save output image files')
    
    parser.add_argument(
        '-b',
        '--basedir',
        help='dir path to image files')
    
    parser.add_argument(
        '-l',
        '--filelist',
        help='filename list for image files')
    
    parser.add_argument(
        '-w',
        '--weight_path',
        help='file path to weight file')
        
    return parser.parse_args(argv)

if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))
