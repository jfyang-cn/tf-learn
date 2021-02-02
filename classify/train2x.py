import os, datetime, sys, argparse
import numpy as np
import json
from tensorflow.keras.callbacks import ModelCheckpoint,TensorBoard
from tensorflow.keras import callbacks
from builder import ModelBuilder

import tensorflow as tf
print(tf.__version__)

physical_devices = tf.config.experimental.list_physical_devices('GPU')
print(physical_devices)
tf.config.experimental.set_visible_devices(physical_devices[2:], 'GPU')
#tf.config.experimental.set_memory_growth(physical_devices[0], True)
    
def train(config):
    
    input_width        = config['model']['input_width']
    input_height       = config['model']['input_height']
    label_file         = config['model']['labels']
    model_name         = config['model']['name']
    class_num          = config['model']['class_num']
    
    train_data_dir     = config['train']['data_dir']
    train_file_list    = config['train']['file_list']
    pretrained_weights = config['train']['pretrained_weights']
    batch_size         = config['train']['batch_size']
    learning_rate      = config['train']['learning_rate']
    nb_epochs          = config['train']['nb_epochs']
    start_epoch        = config['train']['start_epoch']
    train_base         = config['train']['train_base']
    
    valid_data_dir     = config['valid']['data_dir']
    valid_file_list    = config['valid']['file_list']
    
    builder = ModelBuilder(config)

    filepath = train_file_list
    train_gen = builder.build_datagen(filepath)
    train_gen.save_labels(label_file)
    trainDataGen, train_steps_per_epoch = train_gen.from_frame(directory=train_data_dir)
    trainDs = tf.data.Dataset.from_generator(
        lambda: trainDataGen, 
        output_types=(tf.float32, tf.float32), 
        output_shapes=([batch_size,input_width,input_height,3], [batch_size,class_num])
    )
    options = tf.data.Options()
    options.experimental_distribute.auto_shard_policy = tf.data.experimental.AutoShardPolicy.DATA
    trainDs = trainDs.with_options(options)

    
    filepath = valid_file_list
    valid_gen = builder.build_datagen(filepath, with_aug=False)
    validDataGen, valid_steps_per_epoch = valid_gen.from_frame(directory=valid_data_dir)
    validDs = tf.data.Dataset.from_generator(
        lambda: validDataGen, 
        output_types=(tf.float32, tf.float32), 
        output_shapes=([batch_size,input_width,input_height,3], [batch_size,class_num])
    )    
    options = tf.data.Options()
    options.experimental_distribute.auto_shard_policy = tf.data.experimental.AutoShardPolicy.DATA
    validDs = validDs.with_options(options)    
    
    # define checkpoint
    dataset_name = model_name
    dirname = 'ckpt-' + dataset_name
    if not os.path.exists(dirname):
        os.makedirs(dirname)

    timestr = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    filepath = os.path.join(dirname, 'weights-%s-%s-{epoch:02d}-{val_accuracy:.2f}.hdf5' %(model_name, timestr))
    checkpoint = ModelCheckpoint(filepath=filepath, 
                             monitor='val_accuracy',    # acc outperforms loss
                             verbose=1, 
                             save_best_only=True, 
                             save_weights_only=True, 
                             period=5)

    # define logs for tensorboard
    tensorboard = TensorBoard(log_dir='logs', histogram_freq=0)

    wgtdir = 'weights'
    if not os.path.exists(wgtdir):
        os.makedirs(wgtdir)

    # train
    # tf2.5
    strategy = tf.distribute.MirroredStrategy()
    print("Number of devices: {}".format(strategy.num_replicas_in_sync))

    # Open a strategy scope.
    with strategy.scope():
        model = builder.build_model()

        # tf2.5
        if class_num == 2:
            model.compile(optimizer=tf.optimizers.Adam(learning_rate=learning_rate), 
                          loss='categorical_crossentropy',metrics=['accuracy'])
        else:
            model.compile(optimizer=tf.optimizers.Adam(learning_rate=learning_rate), 
                          loss='sparse_categorical_crossentropy',metrics=['accuracy'])
        model.summary()

        # Load weight of unfinish training model(optional)
        if pretrained_weights != '':
            model.load_weights(pretrained_weights)

        model.fit(trainDs,
                  batch_size = batch_size,
                  steps_per_epoch=train_steps_per_epoch,
                          validation_data = validDs,
                  validation_steps=valid_steps_per_epoch,
                          initial_epoch=start_epoch, 
                          epochs=nb_epochs, 
                          callbacks=[checkpoint,tensorboard], 
                          use_multiprocessing=True, 
                          workers=16)
        model_file = '%s_%s.h5' % (model_name,timestr)
        model.save(model_file)
        print('save model to %s' % (model_file))

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
