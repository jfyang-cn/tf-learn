import os, datetime
import numpy as np
from tensorflow.keras.callbacks import ModelCheckpoint,TensorBoard
from tensorflow.keras import callbacks
from tensorflow import keras
from data_gen import DataGen
from model import autoencoder,autoencoder1

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

save_dir = './results'
filepath = '/home/philyang/git/dataset/helmet/train_mask.txt'
dataGen = DataGen(filepath)


# define checkpoint
dataset_name = dataGen.name()
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
                         period=10)

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
    autoencoder, encoder = autoencoder1()
    autoencoder.compile(optimizer=tf.keras.optimizers.Adadelta(learning_rate=0.0001), loss='binary_crossentropy')
    encoder.summary()
    autoencoder.summary()
    
    start_epoch = 0
    
    # Load weight of unfinish training model(optional)
    load_model = False
    if load_model:
        weights_path = 'ckpt-human/weights-20200724-183044-1220-0.24.hdf5' # name of model 
        start_epoch = 1220
        autoencoder.load_weights(weights_path)
    
#     autoencoder.fit(x, x, batch_size=batch_size, epochs=pretrain_epochs) #, callbacks=cb)
    autoencoder.fit_generator(dataGen, initial_epoch=start_epoch, epochs=100000, callbacks=[checkpoint,tensorboard], use_multiprocessing=False, workers=16)
    autoencoder.save_weights(save_dir + '/myae_weights.h5')
