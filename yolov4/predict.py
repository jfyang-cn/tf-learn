'''
predict.py有几个注意点
1、无法进行批量预测，如果想要批量预测，可以利用os.listdir()遍历文件夹，利用Image.open打开图片文件进行预测。
2、如果想要保存，利用r_image.save("img.jpg")即可保存。
3、如果想要获得框的坐标，可以进入detect_image函数，读取top,left,bottom,right这四个值。
4、如果想要截取下目标，可以利用获取到的top,left,bottom,right这四个值在原图上利用矩阵的方式进行截取。
'''
from tensorflow import keras
from tensorflow.keras.layers import Input
from PIL import Image
import os

from nets.yolo4 import yolo_body
from yolo import YOLO
import numpy as np

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

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="0"
    
np_load_old = np.load
np.load = lambda *a, **k: np_load_old(*a, allow_pickle=True, **k)

# while True:
#     img = input('Input image filename:')
img1 = 'img/1.jpg'
img2 = 'img/2.jpg'
try:
    image1 = Image.open(img1)
    image2 = Image.open(img2)
    images = []
    images.append(image1)
    images.append(image2)
    
    images.append(image1)
    images.append(image2)
    
    images.append(image1)
    images.append(image2)
    
    images.append(image1)
    images.append(image2)
    
    images.append(image1)
    images.append(image2)
    
    images.append(image1)
    images.append(image2)
    
    images.append(image1)
    images.append(image2)
    
    images.append(image1)
    images.append(image2)
    
#     //////////////////////
    
    images.append(image1)
    images.append(image2)
    
    images.append(image1)
    images.append(image2)
    
    images.append(image1)
    images.append(image2)
    
    images.append(image1)
    images.append(image2)
    
    images.append(image1)
    images.append(image2)
    
    images.append(image1)
    images.append(image2)
    
    images.append(image1)
    images.append(image2)
    
    images.append(image1)
    images.append(image2)
    
except:
    print('Open Error! Try again!')
else:
        # train
    train_graph = tf.Graph()
    train_sess = tf.Session(graph=train_graph,config=tf_config)

    keras.backend.set_session(train_sess)
    with train_graph.as_default():
        yolo = YOLO()
        
        for i in range(10):
            r_images = yolo.detect_on_batch(images)
#         for i,r_image in enumerate(r_images):
#             r_image.save('out%d.jpg' % (i))
#         r_image.show()

yolo.close_session()
