import os, datetime, sys, argparse
import numpy as np
import json

from data_gen import DataGenerator
from model import create_model
from data_helper import readfile_to_dict

from tensorflow.keras.callbacks import Callback, ModelCheckpoint,TensorBoard
from tensorflow import keras
from google.protobuf import json_format

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
elif tf.__version__ == '1.11.0' or tf.__version__ == '1.13.2':
    from tensorflow import ConfigProto
    from tensorflow import InteractiveSession

    tf_config = ConfigProto(allow_soft_placement=True)
    tf_config.gpu_options.allow_growth = True
        
    tf_config.gpu_options.per_process_gpu_memory_fraction = 0.9
    

def train(config):
    
    ###### Parameters setting
    dim = (config['model']['input_width'],config['model']['input_height']) # for MobileNetV2
    n_sequence = config['model']['sequence'] # for LSTM
    n_channels = config['model']['channels'] # color channel(RGB)
    n_output = config['model']['class_num'] # number of output class
    batch_size = config['train']['batch_size']
    n_mul_train = 1 # To increase sample of train set
    n_mul_test = 4 # To increase sample of test set
    path_dataset = config['train']['data_dir']
    ######    
    
    # Keyword argument
    params = {'dim': dim,
              'batch_size': batch_size, # you can increase for faster training
              'n_sequence': n_sequence,
              'n_channels': n_channels,
              'path_dataset': path_dataset,
              'option': 'RGBdiff',
              'shuffle': True}

    train_txt = config['train']['file_list']
    test_txt = config['valid']['file_list']

    # Read file
    # train_d and test_d is dictionary that contain name of video as key and label as value
    # For example, {'a01\a01_s08_e01': 0, 'a01\a01_s08_e02': 0, .... }
    # It's used for getting label(Y)
    train_d = readfile_to_dict(train_txt)
    test_d = readfile_to_dict(test_txt)

    # Prepare key, name of video(X)
    train_keys = list(train_d.keys()) * n_mul_train
    test_keys = list(test_d.keys()) * n_mul_test

    # Generators
    training_generator = DataGenerator(train_keys, train_d, **params, type_gen='train')
    validation_generator = DataGenerator(test_keys, test_d, **params, type_gen='test')

    # define logs for tensorboard
    tensorboard = TensorBoard(log_dir='logs', histogram_freq=0)

    # train
    train_graph = tf.Graph()
    train_sess = tf.Session(graph=train_graph,config=tf_config)

    keras.backend.set_session(train_sess)
    with train_graph.as_default():

        # Design model
        model = create_model(n_output)    

    #     model.compile(optimizer=tf.train.AdamOptimizer(), loss='sparse_categorical_crossentropy', metrics=['acc'])
        model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['acc'])
        model.summary()

        start_epoch = 0

        # Load weight of unfinish training model(optional)
        pretrained_weights = config['train']['pretrained_weights'] # name of model             
        if pretrained_weights != '':
            start_epoch = config['train']['start_epoch']
            model.load_weights(pretrained_weights)

        # Set callback
        wgtdir = 'save_weight'
        if not os.path.exists(wgtdir):
            os.makedirs(wgtdir)
        validate_freq = 5
        filepath = os.path.join(wgtdir, "weight-{epoch:02d}-{acc:.2f}-{val_acc:.2f}.hdf5")
        checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=False, period=validate_freq)
        callbacks_list = [checkpoint,tensorboard]

        # Train model on dataset
        model.fit_generator(generator=training_generator,
                            validation_data=validation_generator,
                            epochs=config['train']['nb_epochs'],
                            callbacks=callbacks_list,
                            initial_epoch=start_epoch,
                            validation_freq=validate_freq
                            )

def main(args):
    
    config_path = args.conf
    
    with open(config_path) as config_buffer:    
        config = json.loads(config_buffer.read())    
        train(config)
        
def parse_arguments(argv):
    
    parser = argparse.ArgumentParser()
    
    parser.add_argument(
        '-c',
        '--conf',
        help='path to configuration file')
        
    return parser.parse_args(argv)

if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))