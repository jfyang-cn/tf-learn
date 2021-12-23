import os, sys, datetime, argparse
import numpy as np
from tensorflow.keras.callbacks import ModelCheckpoint,TensorBoard
from tensorflow.keras import callbacks
import tensorflow.keras.backend as K
from data_gen import DataGen
from cifar10_gen import DataGenCifar10
from model import autoencoder

import tensorflow as tf
print(tf.__version__)

physical_devices = tf.config.experimental.list_physical_devices('GPU')
print(physical_devices)
tf.config.experimental.set_visible_devices(physical_devices[0:], 'GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)

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


def sparse_crossentropy_ignoring_first_label(y_true, y_pred):
    num_classes = K.shape(y_pred)[-1]
    y_true = K.one_hot(tf.cast(y_true[..., 0], tf.int32), num_classes)[..., 1:num_classes]
    y_pred = y_pred[..., 1:num_classes]
    return K.categorical_crossentropy(y_true, y_pred)

def sparse_crossentropy(y_true, y_pred):
    num_classes = K.shape(y_pred)[-1]
    y_true = K.one_hot(tf.cast(y_true[..., 0], tf.int32), num_classes)
    return K.categorical_crossentropy(y_true, y_pred)

def main(args):
    
    path_dataset = args.dataset # '/home/philyang/drone/data/data512'
    traintxt = args.traintxt # '/home/philyang/drone/data/data512/train.txt'
    trainGen = DataGen(filepath=traintxt, path_dataset=path_dataset)
#     trainGen = DataGenCifar10(batch_size=4, class_num=10, dim=(256,256), n_channels=3)
    
    valtxt = args.valtxt    
    valGen = DataGen(filepath=valtxt, path_dataset=path_dataset)
#     valGen = DataGenCifar10(batch_size=4, class_num=10, dim=(256,256), n_channels=3)
    
    # define checkpoint
    dataset_name = trainGen.name()
    dirname = 'ckpt-' + dataset_name
    if not os.path.exists(dirname):
        os.makedirs(dirname)

    timestr = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    filepath = os.path.join(dirname, 'weights-%s-{epoch:02d}-{loss:.2f}.hdf5' %(timestr))
    checkpoint = ModelCheckpoint(filepath=filepath, 
                             monitor='val_loss',    # acc outperforms loss
                             verbose=1, 
                             save_best_only=False, 
                             save_weights_only=True, 
                             period=5)

    # define logs for tensorboard
    tensorboard = TensorBoard(log_dir='logs', histogram_freq=0)

    wgtdir = 'weights'
    if not os.path.exists(wgtdir):
        os.makedirs(wgtdir)    

    strategy = tf.distribute.MirroredStrategy()
    print("Number of devices: {}".format(strategy.num_replicas_in_sync))
        
    # Open a strategy scope.
#     with strategy.scope():
    model = autoencoder()
    model.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=0.001), loss=sparse_crossentropy)
    model.summary()

    start_epoch = 0

    # Load weight of unfinish training model(optional)
    if args.weights is not None and args.start_epoch is not None:
        weights_path = args.weights
        start_epoch = int(args.start_epoch)
        model.load_weights(weights_path)

    model.fit_generator(
        generator=trainGen,
        steps_per_epoch=len(trainGen),
        validation_data=valGen,
        validation_steps=len(valGen),
        initial_epoch=start_epoch, 
        epochs=1000, 
        callbacks=[checkpoint,tensorboard], 
        use_multiprocessing=False, 
        verbose=1,
        workers=1,
        max_queue_size=10)

    model.save('./model.h5')

def parse_arguments(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('-t','--traintxt', type=str, required=True, default='train.txt',
        help='filename list for training data without dir path')
    parser.add_argument('-v','--valtxt', type=str, required=True, default='val.txt',
        help='filename list for validating data without dir path')
    parser.add_argument('-d','--dataset', type=str, required=True, default='./',
        help='dataset path for images and lables')
    parser.add_argument('-w','--weights', type=str, required=False, default=None,
        help='weights file path for pre-train model')
    parser.add_argument('-s','--start_epoch', type=int, required=False, default=None,
        help='epoch num to start with')
    return parser.parse_args(argv)
    
if __name__ == "__main__":
    main(parse_arguments(sys.argv[1:]))