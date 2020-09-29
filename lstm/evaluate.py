import os, datetime, sys, argparse
from sklearn.metrics import confusion_matrix
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sn
import numpy as np
import json

from data_gen import DataGenerator
from model import create_model
from data_helper import readfile_to_dict

import tensorflow as tf
from tensorflow import keras

print(tf.__version__)

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
        
#     gpus = tf.config.experimental.list_physical_devices('GPU')
#     for gpu in gpus:
#         tf.config.experimental.set_memory_growth(gpu, True)
        
    tf_config.gpu_options.per_process_gpu_memory_fraction = 0.9

    
def evaluate(config, weights_path):
    
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

    params = {'dim': dim,
              'batch_size': batch_size,
              'n_sequence': n_sequence,
              'n_channels': n_channels,
              'path_dataset': path_dataset,
              'option': 'RGBdiff',
              'shuffle': False}

    test_txt = config['valid']['file_list']
    test_d = readfile_to_dict(test_txt)
    key_list = list(test_d.keys()) * n_mul_test  # IDs

    # validation_generator = DataGeneratorBKB(partition['validation'] , labels, **params, type_gen='test') # for evalutate_generator
    predict_generator = DataGenerator(key_list , test_d, **params, type_gen='predict')

    # evaluate
    eval_graph = tf.Graph()
    eval_sess = tf.Session(graph=eval_graph,config=tf_config)

    keras.backend.set_session(eval_sess)
    with eval_graph.as_default():
        keras.backend.set_learning_phase(0)

        model = create_model(n_output)

        model.load_weights(weights_path)

        # Example for evaluate generator
        # If you want to use, just uncomment it
        # loss, acc = model.evaluate_generator(validation_generator, verbose=0)
        # print(loss,acc)

        # #### Confusion Matrix
        y_pred_prob = model.predict_generator(predict_generator, workers=0)


    test_y = np.array(list(test_d.values()) * n_mul_test)
    print("-----------")
    print(y_pred_prob.shape)
    print(len(test_y))

    y_pred = np.argmax(y_pred_prob, axis=1)
    normalize = True

    all_y = len(test_y)
    sum = all_y
    for i in range(len(y_pred)):
        if test_y[i] != y_pred[i]:
            sum -= 1
            print(key_list[i],' actual:',test_y[i],'predict:',y_pred[i])

    cm = confusion_matrix(test_y, y_pred)
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    accuracy = sum / all_y
    print("accuracy:",accuracy)

    classes = [*range(1,n_output+1)] # [1,2,3,...,18]

    df_cm = pd.DataFrame(cm, columns=classes, index=classes)
    df_cm.index.name = 'Actual'
    df_cm.columns.name = 'Predicted'
    fig, ax = plt.subplots(figsize=(5,5))
    sn.set(font_scale=0.6)#for label size
    sn.heatmap(df_cm, cmap="Blues", annot=True,fmt=".2f", annot_kws={"size": 8})# font size
    # ax.set_ylim(5, 0)
    # plt.show()
    plt.savefig('eval_model.png')


def main(args):
    
    config_path = args.conf
    weights_path = args.weights
    
    with open(config_path) as config_buffer:    
        config = json.loads(config_buffer.read())    
        evaluate(config, weights_path)
        
def parse_arguments(argv):
    
    parser = argparse.ArgumentParser()
    
    parser.add_argument(
        '-c',
        '--conf',
        help='path to configuration file')
        
    parser.add_argument(
        '-w',
        '--weights', 
        type=str,
        help='weights file. either weights or model should be specified.', default=None)

    return parser.parse_args(argv)

if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))