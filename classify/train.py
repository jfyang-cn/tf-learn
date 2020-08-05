import os, datetime
import numpy as np
from tensorflow.keras.callbacks import ModelCheckpoint,TensorBoard
from tensorflow.keras import callbacks
from tensorflow import keras
from data_gen import DataGen
from model import classifier

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
    
    input_width        = config['model']['input_width']
    input_height       = config['model']['input_height']
    
    data_dir           = config['train']['data_dir']
    file_list          = config['train']['file_list']
    pretrained_weights = config['train']['pretrained_weights']
    batch_size         = config['train']['batch_size']
    learning_rate      = config['train']['learning_rate']
    nb_epochs          = config['train']['nb_epochs']
    start_epoch        = config['train']['start_epoch']
    
    filepath = os.path.join(data_dir, file_list)
    gen = DataGen(filepath, batch_size, target_size=(input_width,input_height)):
    gen.save_labels()
    dataGen, steps_per_epoch = gen.from_frame(directory=data_dir)

    # define checkpoint
    dataset_name = gen4cls.name()
    dirname = 'ckpt-' + dataset_name
    if not os.path.exists(dirname):
        os.makedirs(dirname)

    timestr = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    filepath = os.path.join(dirname, 'weights-%s-{epoch:02d}-{loss:.2f}.hdf5' %(timestr))
    checkpoint = ModelCheckpoint(filepath=filepath, 
                             monitor='loss',    # acc outperforms loss
                             verbose=1, 
                             save_best_only=True, 
                             save_weights_only=True, 
                             period=1)

    # define logs for tensorboard
    tensorboard = TensorBoard(log_dir='logs', histogram_freq=0)

    wgtdir = 'weights'
    if not os.path.exists(wgtdir):
        os.makedirs(wgtdir)

    # train
    train_graph = tf.Graph()
    train_sess = tf.Session(graph=train_graph,config=tf_config)

    keras.backend.set_session(train_sess)
    with train_graph.as_default():
        cls = classifier()
        cls.compile(optimizer=tf.keras.optimizers.Adam(lr=learning_rate), loss='categorical_crossentropy')
        cls.summary()

        # Load weight of unfinish training model(optional)
        if pretrained_weights != '':
            cls.load_weights(pretrained_weights)

        cls.fit_generator(dataGen, 
                          initial_epoch=start_epoch, 
                          epochs=nb_epochs, 
                          callbacks=[checkpoint,tensorboard], 
                          use_multiprocessing=False, 
                          workers=16)
        cls.save_weights('cls_weights.h5')

def main(args):
    
    config_path = args.conf
    
    with open(config_path) as config_buffer:    
        config = json.loads(config_buffer.read())    
        train(config)

def parse_arguments(argv):
    
    parser = argparse.ArgumentParser()
    
    argparser.add_argument(
        '-c',
        '--conf',
        help='path to configuration file')
        
    return parser.parse_args(argv)

if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))
