import os,sys,argparse,datetime
import numpy as np
import json
import cv2

from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array,load_img
from tensorflow.keras.applications import inception_v3,mobilenet,vgg16,resnet50
from tensorflow import keras
from tensorflow.keras.models import Model
from model import classifier
from matplotlib import pyplot as plt

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
        
    tf_config.gpu_options.per_process_gpu_memory_fraction = 0.3
elif tf.__version__ == '1.11.0' or tf.__version__ == '1.13.2':
    from tensorflow import ConfigProto
    from tensorflow import InteractiveSession

    tf_config = ConfigProto(allow_soft_placement=True)
    tf_config.gpu_options.allow_growth = True
        
    tf_config.gpu_options.per_process_gpu_memory_fraction = 0.9

np_load_old = np.load
np.load = lambda *a, **k: np_load_old(*a, allow_pickle=True, **k)

def build_model(base_model, out_layer):
    
    x = base_model.get_layer(out_layer)
    model = Model(inputs=base_model.input, outputs=x.output)
    return model

def layer_visualize(config, img_file, weights, model_file, layer_name):
    
    labels_file    = config['model']['labels']
    input_width    = config['model']['input_width']
    input_height   = config['model']['input_height']
    
    test_data_dir  = config['test']['data_dir']
    
    # load labels
    print('load label file', labels_file)
    label_dict = np.load(labels_file).item()
    class_num = len(label_dict)
    print('class num:', class_num)
    print(label_dict)
    
    visdir = 'visual'
    if not os.path.exists(visdir):
        os.makedirs(visdir)
        
    laydir = os.path.join(visdir,layer_name)
    if not os.path.exists(laydir):
        os.makedirs(laydir)
    
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

        model = build_model(cls, layer_name)

        img_path = os.path.join(test_data_dir, img_file)
        print(img_path)
        img = cv2.imread(img_path)
        img = cv2.resize(img, (input_width, input_height))
#             img = img_pad(img, input_width, input_height)
        img = img[:,:,::-1]
        x_pred = np.array([img]).astype('float32')
#       x_pred = np.array([img])/255.0

        preprocess_input = vgg16.preprocess_input

#       x_pred = np.expand_dims(img_to_array(load_img(img_path, target_size=image_size)), axis=0).astype('float32')
        x_pred = preprocess_input(x_pred)
        y_pred = model.predict(x_pred)                

        _,_,_,n = y_pred.shape
        for i in range(n):
            filename = './%s/%i.png' % (laydir, i)
            print(filename)
            plt.imsave(filename, y_pred[0][:,:,i],cmap='viridis') #gray

def main(args):
    
    config_path = args.conf
    image_file = args.image_file
    weights = args.weights
    model_file = args.model
    layer_name = args.layer_name
    
    with open(config_path) as config_buffer:    
        config = json.loads(config_buffer.read())    
        layer_visualize(config, image_file, weights, model_file, layer_name)

def parse_arguments(argv):
    
    parser = argparse.ArgumentParser()
    
    parser.add_argument(
        '-c',
        '--conf',
        help='path to configuration file')
    
    parser.add_argument(
        '-i',
        '--image_file', 
        type=str,
        help='image file path for visualize analysis', default=None)

    parser.add_argument(
        '-l',
        '--layer_name', 
        type=str,
        help='specific layer to be visualized', default=None)
    
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

    return parser.parse_args(argv)

if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))
