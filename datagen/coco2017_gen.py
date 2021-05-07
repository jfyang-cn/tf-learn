import numpy as np
import cv2
import os
import tensorflow as tf
from pycocotools.coco import COCO
# from matplotlib.colors import hsv_to_rgb, rgb_to_hsv
from PIL import Image
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.python.keras.utils.data_utils import Sequence

import colorsys
import random

def rand(a=0, b=1):
    return np.random.rand()*(b-a) + a

def get_random_data(filepath, input_shape, box, max_boxes=100, jitter=.3, hue=.1, sat=1.5, val=1.5, random=True):
    '''random preprocessing for real-time data augmentation'''

    image = Image.open(filepath)
    iw, ih = image.size
    h, w = input_shape

    if not random:
        # resize image
        scale = min(w/iw, h/ih)
        nw = int(iw*scale)
        nh = int(ih*scale)
        dx = (w-nw)//2
        dy = (h-nh)//2

        image = image.resize((nw,nh), Image.BICUBIC)
        new_image = Image.new('RGB', (w,h), (128,128,128))
        new_image.paste(image, (dx, dy))
        image_data = np.array(new_image, np.float32)/255

        # correct boxes
        box_data = np.zeros((max_boxes,5))
        if len(box)>0:
            np.random.shuffle(box)
            box[:, [0,2]] = box[:, [0,2]]*nw/iw + dx
            box[:, [1,3]] = box[:, [1,3]]*nh/ih + dy
            box[:, 0:2][box[:, 0:2]<0] = 0
            box[:, 2][box[:, 2]>w] = w
            box[:, 3][box[:, 3]>h] = h
            box_w = box[:, 2] - box[:, 0]
            box_h = box[:, 3] - box[:, 1]
            box = box[np.logical_and(box_w>1, box_h>1)] # discard invalid box
            if len(box)>max_boxes: box = box[:max_boxes]
            box_data[:len(box)] = box

        return image_data, box_data
        
    # 对图像进行缩放并且进行长和宽的扭曲
    new_ar = w/h * rand(1-jitter,1+jitter)/rand(1-jitter,1+jitter)
    scale = rand(.25, 2)
    if new_ar < 1:
        nh = int(scale*h)
        nw = int(nh*new_ar)
    else:
        nw = int(scale*w)
        nh = int(nw/new_ar)
    image = image.resize((nw,nh), Image.BICUBIC)

    # 将图像多余的部分加上灰条
    dx = int(rand(0, w-nw))
    dy = int(rand(0, h-nh))
    new_image = Image.new('RGB', (w,h), (128,128,128))
    new_image.paste(image, (dx, dy))
    image = new_image

    # 翻转图像
    flip = rand()<.5
    if flip: image = image.transpose(Image.FLIP_LEFT_RIGHT)

    # 色域扭曲
    hue = rand(-hue, hue)
    sat = rand(1, sat) if rand()<.5 else 1/rand(1, sat)
    val = rand(1, val) if rand()<.5 else 1/rand(1, val)
    x = cv2.cvtColor(np.array(image,np.float32)/255, cv2.COLOR_RGB2HSV)
    x[..., 0] += hue*360
    x[..., 0][x[..., 0]>1] -= 1
    x[..., 0][x[..., 0]<0] += 1
    x[..., 1] *= sat
    x[..., 2] *= val
    x[x[:,:, 0]>360, 0] = 360
    x[:, :, 1:][x[:, :, 1:]>1] = 1
    x[x<0] = 0
    image_data = cv2.cvtColor(x, cv2.COLOR_HSV2RGB) # numpy array, 0 to 1

    # 将box进行调整
    box_data = np.zeros((max_boxes,5))
    if len(box)>0:
        np.random.shuffle(box)
        box[:, [0,2]] = box[:, [0,2]]*nw/iw + dx
        box[:, [1,3]] = box[:, [1,3]]*nh/ih + dy
        if flip: box[:, [0,2]] = w - box[:, [2,0]]
        box[:, 0:2][box[:, 0:2]<0] = 0
        box[:, 2][box[:, 2]>w] = w
        box[:, 3][box[:, 3]>h] = h
        box_w = box[:, 2] - box[:, 0]
        box_h = box[:, 3] - box[:, 1]
        box = box[np.logical_and(box_w>1, box_h>1)] # discard invalid box
        if len(box)>max_boxes: box = box[:max_boxes]
        box_data[:len(box)] = box
    
    return image_data, box_data

#---------------------------------------------------#
#   读入xml文件，并输出y_true
#---------------------------------------------------#
def preprocess_true_boxes(true_boxes, input_shape, anchors, num_classes):
    assert (true_boxes[..., 4]<num_classes).all(), 'class id must be less than num_classes'
    # 一共有三个特征层数
    num_layers = len(anchors)//3
    #-----------------------------------------------------------#
    #   13x13的特征层对应的anchor是[142, 110], [192, 243], [459, 401]
    #   26x26的特征层对应的anchor是[36, 75], [76, 55], [72, 146]
    #   52x52的特征层对应的anchor是[12, 16], [19, 36], [40, 28]
    #-----------------------------------------------------------#
    anchor_mask = [[6,7,8], [3,4,5], [0,1,2]]

    #-----------------------------------------------------------#
    #   获得框的坐标和图片的大小
    #-----------------------------------------------------------#
    true_boxes = np.array(true_boxes, dtype='float32')
    input_shape = np.array(input_shape, dtype='int32')
    #-----------------------------------------------------------#
    #   通过计算获得真实框的中心和宽高
    #   中心点(m,n,2) 宽高(m,n,2)
    #-----------------------------------------------------------#
    boxes_xy = (true_boxes[..., 0:2] + true_boxes[..., 2:4]) // 2
    boxes_wh = true_boxes[..., 2:4] - true_boxes[..., 0:2]
    #-----------------------------------------------------------#
    #   将真实框归一化到小数形式
    #-----------------------------------------------------------#
    true_boxes[..., 0:2] = boxes_xy/input_shape[::-1]
    true_boxes[..., 2:4] = boxes_wh/input_shape[::-1]

    # m为图片数量，grid_shapes为网格的shape
    m = true_boxes.shape[0]
    grid_shapes = [input_shape//{0:32, 1:16, 2:8}[l] for l in range(num_layers)]
    #-----------------------------------------------------------#
    #   y_true的格式为(m,13,13,3,85)(m,26,26,3,85)(m,52,52,3,85)
    #-----------------------------------------------------------#
    y_true = [np.zeros((m,grid_shapes[l][0],grid_shapes[l][1],len(anchor_mask[l]),5+num_classes),
        dtype='float32') for l in range(num_layers)]

    #-----------------------------------------------------------#
    #   [9,2] -> [1,9,2]
    #-----------------------------------------------------------#
    anchors = np.expand_dims(anchors, 0)
    anchor_maxes = anchors / 2.
    anchor_mins = -anchor_maxes

    #-----------------------------------------------------------#
    #   长宽要大于0才有效
    #-----------------------------------------------------------#
    valid_mask = boxes_wh[..., 0]>0

    for b in range(m):
        # 对每一张图进行处理
        wh = boxes_wh[b, valid_mask[b]]
        if len(wh)==0: continue
        #-----------------------------------------------------------#
        #   [n,2] -> [n,1,2]
        #-----------------------------------------------------------#
        wh = np.expand_dims(wh, -2)
        box_maxes = wh / 2.
        box_mins = -box_maxes

        #-----------------------------------------------------------#
        #   计算所有真实框和先验框的交并比
        #   intersect_area  [n,9]
        #   box_area        [n,1]
        #   anchor_area     [1,9]
        #   iou             [n,9]
        #-----------------------------------------------------------#
        intersect_mins = np.maximum(box_mins, anchor_mins)
        intersect_maxes = np.minimum(box_maxes, anchor_maxes)
        intersect_wh = np.maximum(intersect_maxes - intersect_mins, 0.)
        intersect_area = intersect_wh[..., 0] * intersect_wh[..., 1]

        box_area = wh[..., 0] * wh[..., 1]
        anchor_area = anchors[..., 0] * anchors[..., 1]

        iou = intersect_area / (box_area + anchor_area - intersect_area)
        #-----------------------------------------------------------#
        #   维度是[n,] 感谢 消尽不死鸟 的提醒
        #-----------------------------------------------------------#
        best_anchor = np.argmax(iou, axis=-1)

        for t, n in enumerate(best_anchor):
            #-----------------------------------------------------------#
            #   找到每个真实框所属的特征层
            #-----------------------------------------------------------#
            for l in range(num_layers):
                if n in anchor_mask[l]:
                    #-----------------------------------------------------------#
                    #   floor用于向下取整，找到真实框所属的特征层对应的x、y轴坐标
                    #-----------------------------------------------------------#
                    i = np.floor(true_boxes[b,t,0] * grid_shapes[l][1]).astype('int32')
                    j = np.floor(true_boxes[b,t,1] * grid_shapes[l][0]).astype('int32')
                    #-----------------------------------------------------------#
                    #   k指的的当前这个特征点的第k个先验框
                    #-----------------------------------------------------------#
                    k = anchor_mask[l].index(n)
                    #-----------------------------------------------------------#
                    #   c指的是当前这个真实框的种类
                    #-----------------------------------------------------------#
                    c = true_boxes[b, t, 4].astype('int32')
                    #-----------------------------------------------------------#
                    #   y_true的shape为(m,13,13,3,85)(m,26,26,3,85)(m,52,52,3,85)
                    #   最后的85可以拆分成4+1+80，4代表的是框的中心与宽高、
                    #   1代表的是置信度、80代表的是种类
                    #-----------------------------------------------------------#
                    y_true[l][b, j, i, k, 0:4] = true_boxes[b, t, 0:4]
                    y_true[l][b, j, i, k, 4] = 1
                    y_true[l][b, j, i, k, 5+c] = 1

    return y_true

class DataGen(Sequence):
    'Generates data for Keras'
    def __init__(self, filepath, anchors, batch_size=32, class_num=2, dim=(32,32), 
                n_channels=1, preprocess_input=None, with_aug=True, shuffle=True, path_dataset=None,
                type_gen='train', option=None):
        
        coco=COCO(filepath)
        print(f'Annotation file: {filepath}')

        label_dict = {}
        for i,class_id in enumerate(coco.cats.keys()):
            label_dict[class_id] = i
        
        print(len(label_dict.keys()))
        
        self.anchors = anchors
        self.coco = coco
        self.dim = dim
        self.batch_size = batch_size
        self.class_num = class_num
        self.labels = label_dict
        self.list_IDs = list(coco.imgs.keys())
        self.n_channels = n_channels
        self.shuffle = shuffle
        self.path_dataset = path_dataset
        self.type_gen = type_gen
        self.aug_gen = ImageDataGenerator() 
        self.steps_per_epoch = len(self.list_IDs) // self.batch_size
        print("all:", len(self.list_IDs), " batch per epoch", self.steps_per_epoch)
        self.on_epoch_end()
        
    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.list_IDs) / self.batch_size))
    
    def on_epoch_end(self):
        'Updates indexes after each epoch'        
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]
        # Find list of IDs
        list_IDs_temp = [self.list_IDs[k] for k in indexes]
        # Generate data
        X, y = self.__data_generation(list_IDs_temp)
        if self.type_gen == 'predict':
            return X
        else:
            return X, y

    def __data_generation(self, list_IDs_temp):
        'Generates data containing batch_size samples'

        image_data = []
        box_data = []
        for i, ID in enumerate(list_IDs_temp):  # ID is name of file
            
            annIds = self.coco.getAnnIds(imgIds=ID)
            anns = self.coco.loadAnns(annIds)
            boxes = []
            for ann in anns:
                bbox = ann['bbox']
                class_id = ann['category_id']
                boxes.append([bbox[0],bbox[1],bbox[0]+bbox[2],bbox[1]+bbox[3],self.labels[class_id]])

#             print(np.array(boxes))
            
            imgInfo = self.coco.loadImgs(ID)[0]
            path_file = os.path.join(self.path_dataset, imgInfo['file_name']) 
            if self.type_gen == 'train':
                image, box = get_random_data(path_file, self.dim, np.array(boxes), random=True)
            else:
                image, box = get_random_data(path_file, self.dim, np.array(boxes), random=False)
            image_data.append(image)
            box_data.append(box)

        image_data = np.array(image_data)
        box_data = np.array(box_data)
        y_true = preprocess_true_boxes(box_data, self.dim, self.anchors, self.class_num)

        return [image_data, *y_true], np.zeros(self.batch_size)
    
    def save_labels(self, filepath=None):

        if filepath is None:
            filepath = f'coco2017_{self.class_num}_labels.npy'
            
        np.save(filepath, self.labels)
        print(filepath, ' saved.')