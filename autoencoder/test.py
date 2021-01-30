import os, datetime
import numpy as np
from model import autoencoder1
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from tensorflow import keras

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


filepath = '/home/philyang/git/dataset/helmet/test_a.txt'
target_size = (48,48)
data_dir = './test'
fname = '4.png'

# train
train_graph = tf.Graph()
train_sess = tf.Session(graph=train_graph,config=tf_config)

keras.backend.set_session(train_sess)
with train_graph.as_default():
    autoencoder, encoder = autoencoder1()
#     autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy')
    encoder.summary()
    weights_path = 'ckpt-human/weights-20200724-184507-3209-0.21.hdf5' # name of model 
    encoder.load_weights(weights_path, by_name=True)
        
    sources = []
    # srclist = ['1.png', '2.png', '3.png', '4.png', '5.jpg', '13678.jpg', '13827.jpg', '13361.jpg']
    # srclist = ['5.png','6.png','7.png','8.png', '9.png', '10.png']
    srclist = os.listdir(data_dir)
    for fname in srclist:
        x = np.array([img_to_array(load_img(os.path.join(data_dir, fname), target_size=target_size))
                              ]).astype('float32')/255.0        
        en = encoder.predict(x)
#         print(en.shape)
        out = en[0].reshape((72))
    #     print(np.array(out))
        sources.append(out)

    np.save('sources.npy', np.array(sources))
    np.save('srclist.npy',np.array(srclist))
    print(srclist)
    
#     results.append(out)
#     namelist.append(fname)
    
#     with open(filepath, 'r') as f:
#         lines = f.readlines()

#     data_dir = os.path.dirname(os.path.abspath(filepath))
#     fnames = [line.replace('\n', '') for line in lines]
    
    results = []
    namelist = []
    data_dir = '/home/philyang/git/dataset/helmet/test'    
    for fname in os.listdir(data_dir):
        x = np.array([img_to_array(load_img(os.path.join(data_dir, fname), target_size=target_size))
                      ]).astype('float32')/255.0
        en = encoder.predict(x)
        out = en[0].reshape((72))
        results.append(out)
        namelist.append(fname)
    
    np.save('results.npy', np.array(results))
    np.save('namelist.npy',np.array(namelist))
