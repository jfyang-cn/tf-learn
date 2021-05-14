import os, datetime, sys, argparse
# sys.path.append("..")
# sys.path.append(".")
# sys.path.append("")
import numpy as np
import json
import colorsys
import cv2

from tensorflow.keras.callbacks import ModelCheckpoint,TensorBoard,EarlyStopping,ReduceLROnPlateau
from tensorflow.keras import callbacks
from builder import ModelBuilder
from nets.yolo4 import yolo_body, yolo_eval, yolo_decodeout

import tensorflow as tf
print(tf.__version__)

physical_devices = tf.config.experimental.list_physical_devices('GPU')
print(physical_devices)
tf.config.experimental.set_visible_devices(physical_devices[0:], 'GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)

# Restrict TensorFlow to only allocate 1GB of memory on the first GPU
try:
    tf.config.experimental.set_virtual_device_configuration(
        physical_devices[0],
        [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=6144)])
    logical_gpus = tf.config.experimental.list_logical_devices('GPU')
    print(len(physical_devices), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
except RuntimeError as e:
    # Virtual devices must be set before GPUs have been initialized
    print(e)
    
np_load_old = np.load
np.load = lambda *a, **k: np_load_old(*a, allow_pickle=True, **k)

def predict(config, input_image, weights):
    
    input_width        = config['model']['input_width']
    input_height       = config['model']['input_height']
    class_num          = config['model']['class_num']   
   
    builder = ModelBuilder(config)
    model = builder.build_model(training=False)
    model.summary()

    if weights is not None:
        model.load_weights(weights)

    img = cv2.imread(input_image)
    image_h, image_w, _ = img.shape
    print(image_h, image_w)
    img = cv2.resize(img, (input_width, input_height))
    print(img.shape)
    img = img[:,:,::-1]
    x_pred = np.array([img]).astype('float32')
    x_pred /= 255.
    print(x_pred.shape)

    y_outs = model(x_pred)
    y_outs = [a[0].numpy() for a in y_outs]
    print(y_outs[0].shape)
    print(y_outs[1].shape)
    print(y_outs[2].shape)


    # 画框设置不同的颜色
    hsv_tuples = [(x / class_num, 1., 1.) for x in range(class_num)]
    colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
    colors = list(
        map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)),
            colors))
    
    boxes, scores, classes, colors = yolo_decodeout(y_outs, image_h, image_w, colors, 
                                                    input_w=input_width, input_h=input_height, 
                                                    nms_thresh=0.25, class_thresh=0.25, obj_thresh=0.5)    
    for box in boxes:
        print(box.xmin,box.ymin,box.xmax,box.ymax)
    print(scores)
    print(classes)
    print(len(classes))
    
def main(args):
    
    config_path = args.conf
    input_image = args.input_image
    weights = args.weights
    
    with open(config_path) as config_buffer:    
        config = json.loads(config_buffer.read())
        print('config loaded')
        print(config)
        predict(config, input_image, weights)

def parse_arguments(argv):
    
    parser = argparse.ArgumentParser()
    
    parser.add_argument(
        '-c',
        '--conf',
        help='path to configuration file')
    
    parser.add_argument(
        '-i',
        '--input_image',
        help='path to image file',
        default = None)
    
    parser.add_argument(
        '-w',
        '--weights',
        help='path to weights file',
        default = None)
        
    return parser.parse_args(argv)

if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))
