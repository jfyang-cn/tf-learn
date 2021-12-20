import os, datetime
import numpy as np
from tensorflow.keras.callbacks import ModelCheckpoint,TensorBoard
from tensorflow.keras import callbacks
from tensorflow import keras
from data_gen import DataGen
from model import autoencoder

import tensorflow as tf
print(tf.__version__)

# Try to enable Auto Mixed Precision on TF 2.0
os.environ['TF_ENABLE_AUTO_MIXED_PRECISION'] = '1'
os.environ['TF_AUTO_MIXED_PRECISION_GRAPH_REWRITE_IGNORE_PERFORMANCE'] = '1'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

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

def main(args):
    
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

    autoencoder, encoder = autoencoder()
    autoencoder.compile(optimizer=tf.keras.optimizers.Adadelta(learning_rate=0.0001), loss='binary_crossentropy')
    encoder.summary()
    autoencoder.summary()
    
    start_epoch = 0
    
    # Load weight of unfinish training model(optional)
    load_model = False
    if load_model:
        weights_path = 'ckpt-drone/weights-20200724-183044-1220-0.24.hdf5' # name of model 
        start_epoch = 1220
        autoencoder.load_weights(weights_path)
    
#     autoencoder.fit(x, x, batch_size=batch_size, epochs=pretrain_epochs) #, callbacks=cb)
    autoencoder.fit_generator(
        dataGen, 
        initial_epoch=start_epoch, 
        epochs=100000, 
        callbacks=[checkpoint,tensorboard], 
        use_multiprocessing=False, 
        workers=16)
    
    autoencoder.save_weights(save_dir + '/myae_weights.h5')
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    main(args)
