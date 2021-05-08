import os, datetime, sys, argparse
# sys.path.append("..")
# sys.path.append(".")
# sys.path.append("")
import numpy as np
import json
from tensorflow.keras.callbacks import ModelCheckpoint,TensorBoard,EarlyStopping,ReduceLROnPlateau
from tensorflow.keras import callbacks
from builder import ModelBuilder

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

is_debug = True

def train(config):
    
    input_width        = config['model']['input_width']
    input_height       = config['model']['input_height']
    input_depth        = config['model']['input_depth']
    label_file         = config['model']['labels']
    model_name         = config['model']['name']
    class_num          = config['model']['class_num']

    pretrained_weights = config['train']['pretrained_weights']
    batch_size         = config['train']['batch_size']
    learning_rate      = config['train']['learning_rate']
    nb_epochs          = config['train']['nb_epochs']
    start_epoch        = config['train']['start_epoch']
    train_base         = config['train']['train_base']
    

    builder = ModelBuilder(config)
    
    train_gen = builder.build_train_datagen()
    train_gen.save_labels(label_file)
    trainDataGen = train_gen
    train_steps_per_epoch = train_gen.steps_per_epoch
#     trainDs = tf.data.Dataset.from_generator(
#         lambda: trainDataGen, 
#         output_types=(tf.float32, tf.float32), 
#         output_shapes=([batch_size,input_width,input_height,3], [batch_size,class_num])
#     )
#     options = tf.data.Options()
#     options.experimental_distribute.auto_shard_policy = tf.data.experimental.AutoShardPolicy.DATA
#     trainDs = trainDs.with_options(options)
    
    valid_gen = builder.build_valid_datagen()
    validDataGen = valid_gen
    valid_steps_per_epoch = valid_gen.steps_per_epoch
#     validDs = tf.data.Dataset.from_generator(
#         lambda: validDataGen, 
#         output_types=(tf.float32, tf.float32), 
#         output_shapes=([batch_size,input_width,input_height,3], [batch_size,class_num])
#     )    
#     options = tf.data.Options()
#     options.experimental_distribute.auto_shard_policy = tf.data.experimental.AutoShardPolicy.DATA
#     validDs = validDs.with_options(options)    
    
    if is_debug:
        train_steps_per_epoch = 30
        valid_steps_per_epoch = 5
    
    # define checkpoint
    dataset_name = model_name
    dirname = 'ckpt-' + dataset_name
    if not os.path.exists(dirname):
        os.makedirs(dirname)

    timestr = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    filepath = os.path.join(dirname, 'weights-%s-%s-{epoch:02d}-{val_loss:.2f}.hdf5' %(model_name, timestr))
    checkpoint = ModelCheckpoint(filepath=filepath, 
                             monitor='val_loss',    # acc outperforms loss
                             verbose=1, 
                             save_best_only=True, 
                             save_weights_only=True, 
                             period=1)
    
    early_stopping = EarlyStopping(monitor='val_loss', min_delta=0, patience=10, verbose=1)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, verbose=1)
    
    # define logs for tensorboard
    tensorboard = TensorBoard(log_dir='logs', histogram_freq=0)

    wgtdir = 'weights'
    if not os.path.exists(wgtdir):
        os.makedirs(wgtdir)

    # train
    # tf2.5
    strategy = tf.distribute.MirroredStrategy()
    print("Number of devices: {}".format(strategy.num_replicas_in_sync))

    try:    
        # Open a strategy scope.
        with strategy.scope():
            model = builder.build_model()
            model.compile(optimizer=tf.optimizers.Adam(learning_rate), loss={'yolo_loss': lambda y_true, y_pred: y_pred})
            model.summary()

            # Load weight of unfinish training model(optional)
            if pretrained_weights != '':
                model.load_weights(pretrained_weights)

            model.fit(train_gen,
    #                   batch_size = batch_size,
                      steps_per_epoch=train_steps_per_epoch,
                      validation_data = valid_gen,
                      validation_steps=valid_steps_per_epoch,
                      initial_epoch=start_epoch, 
                      epochs=nb_epochs, 
                      callbacks=[checkpoint,tensorboard,reduce_lr,early_stopping], 
                      use_multiprocessing=False, 
                      workers=4)
            model_file = '%s_%s.h5' % (model_name,timestr)
            model.save(model_file)
            print('save model to ', model_file)
    except IndexError:
        pass  # There was a "pop from empty list" error in "tensorflow/python/distribute/distribution_strategy_context.py" that I'm ignoring

def main(args):
    
    config_path = args.conf
    
    with open(config_path) as config_buffer:    
        config = json.loads(config_buffer.read())
        print('config loaded')
        print(config)
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