import tensorflow as tf
from tensorflow import keras

import os, datetime, sys, argparse
import numpy as np
import json
import matplotlib.pyplot as plt
from model import create_model_pretrain
from google.protobuf import json_format

print(tf.__version__)
if tf.__version__ == '1.14.0':
    from tensorflow.compat.v1 import ConfigProto
    from tensorflow.compat.v1 import InteractiveSession

    tf_config = ConfigProto()
    tf_config.gpu_options.allow_growth = True
elif tf.__version__ == '1.11.0' or tf.__version__ == '1.13.2':
    from tensorflow import ConfigProto
    from tensorflow import InteractiveSession

    tf_config = ConfigProto(allow_soft_placement=True)
    tf_config.gpu_options.allow_growth = True
        
    tf_config.gpu_options.per_process_gpu_memory_fraction = 0.9

    
def freeze(config):

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

    # with open('all_tensor_names.txt') as f:
    #     t_names = f.readlines()

    # tensor_names = []
    # for name in t_names:
    #     if name.split(':  ')[0] == 'tensor_name':
    #         tensor_names.append(name.split(':  ')[1])

    # print('tensor_names: ',len(tensor_names), tensor_names)

    # eval
    # tf.reset_default_graph()
    eval_graph = tf.Graph()
    eval_sess = tf.Session(graph=eval_graph,config=tf_config)

    keras.backend.set_session(eval_sess)

    with eval_graph.as_default():
        keras.backend.set_learning_phase(0)     # 

        # Design model
        eval_model = create_model_pretrain(n_output, is_training=True)
    #     eval_model.summary()
    #     eval_sess.run(tf.global_variables_initializer())

        tf.contrib.quantize.create_eval_graph(input_graph=eval_graph)
    #     tf.contrib.quantize.create_training_graph(input_graph=eval_graph)
        eval_graph_def = eval_graph.as_graph_def()

        json_string = json_format.MessageToJson(eval_graph_def)
        with open('eval_graph.json', 'w') as fw:
            json.dump(json_string, fw)

        print('model output:',eval_model.output)

    #     variables_to_restore = tf.contrib.slim.get_variables_to_restore(include=tensor_names)

        saver = tf.train.Saver()
        saver.restore(eval_sess, 'checkpoints')
        frozen_graph_def = tf.graph_util.convert_variables_to_constants(
            eval_sess,
            eval_graph_def,
            ['s_out/Softmax']
    #         [eval_model.output[0].op.name,
    #         eval_model.output[1].op.name,
    #         eval_model.output[2].op.name]
        )

        with open('lstm_model.pb', 'wb') as f:
            f.write(frozen_graph_def.SerializeToString())
        
def main(args):
    
    config_path = args.conf
    
    with open(config_path) as config_buffer:    
        config = json.loads(config_buffer.read())    
        freeze(config)
        
def parse_arguments(argv):
    
    parser = argparse.ArgumentParser()
    
    parser.add_argument(
        '-c',
        '--conf',
        help='path to configuration file')
        
    return parser.parse_args(argv)

if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))
