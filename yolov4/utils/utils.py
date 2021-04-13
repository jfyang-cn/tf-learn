from functools import reduce

import cv2
from tensorflow import keras
import tensorflow.keras.backend as K
import numpy as np
from matplotlib.colors import hsv_to_rgb, rgb_to_hsv
from PIL import Image

# Generate colors in bounding boxes
import colorsys
import random

# Draw bounding boxes
class BoundBox:
    def __init__(self, xmin, ymin, xmax, ymax, objness = None, classes = None):
        self.xmin = xmin
        self.ymin = ymin
        self.xmax = xmax
        self.ymax = ymax
        self.objness = objness
        self.classes = classes
        self.label = -1
        self.score = -1
 
    def get_label(self):
        if self.label == -1:
            self.label = np.argmax(self.classes)
 
        return self.label
 
    def get_score(self):
        if self.score == -1:
            self.score = self.classes[self.get_label()]
 
        return self.score

def _sigmoid(x):
    return 1. / (1. + np.exp(-x))
 
def decode_netout(netout, anchors, obj_thresh, net_h, net_w, anchors_nb, scales_x_y):
#     print(netout.shape)
    grid_h, grid_w = netout.shape[:2]  
    nb_box = anchors_nb
    netout = netout.reshape((grid_h, grid_w, nb_box, -1))
    nb_class = netout.shape[-1] - 5 # 5 = bx,by,bh,bw,pc

#     print("grid_h,grid_w: ",grid_h,grid_w)   
#     print("nb class: ",nb_class)   
    
    boxes = []
    netout[..., :2] = _sigmoid(netout[..., :2]) # x, y
    netout[..., :2] = netout[..., :2]*scales_x_y - 0.5*(scales_x_y - 1.0) # scale x, y

    netout[..., 4:] = _sigmoid(netout[..., 4:]) # objectness + classes probabilities

    for i in range(grid_h*grid_w):

        row = i / grid_w
        col = i % grid_w
        
        
        for b in range(nb_box):
            # 4th element is objectness
            objectness = netout[int(row)][int(col)][b][4]

            if(objectness > obj_thresh):
#                 print("objectness: ",objectness)                
            
                # first 4 elements are x, y, w, and h
                x, y, w, h = netout[int(row)][int(col)][b][:4]
                x = (col + x) / grid_w # center position, unit: image width
                y = (row + y) / grid_h # center position, unit: image height
                w = anchors[2 * b + 0] * np.exp(w) / net_w # unit: image width
                h = anchors[2 * b + 1] * np.exp(h) / net_h # unit: image height            
            
                # last elements are class probabilities
                classes = objectness*netout[int(row)][col][b][5:]
                classes *= classes > obj_thresh
                box = BoundBox(x-w/2, y-h/2, x+w/2, y+h/2, objectness, classes)           
                boxes.append(box)
    return boxes


# Correct the boxes according the inital size of the image
def correct_yolo_boxes(boxes, image_h, image_w, net_h, net_w):
#     new_w, new_h = net_w, net_h
    
    if (float(net_w)/image_w) < (float(net_h)/image_h):
        new_w = net_w
        new_h = (image_h*net_w)/image_w
    else:
        new_h = net_w
        new_w = (image_w*net_h)/image_h

    for i in range(len(boxes)):
        x_offset, x_scale = (net_w - new_w)/2./net_w, float(new_w)/net_w
        y_offset, y_scale = (net_h - new_h)/2./net_h, float(new_h)/net_h
        boxes[i].xmin = int((boxes[i].xmin - x_offset) / x_scale * image_w)
        boxes[i].xmax = int((boxes[i].xmax - x_offset) / x_scale * image_w)
        boxes[i].ymin = int((boxes[i].ymin - y_offset) / y_scale * image_h)
        boxes[i].ymax = int((boxes[i].ymax - y_offset) / y_scale * image_h)

def _interval_overlap(interval_a, interval_b):
    x1, x2 = interval_a
    x3, x4 = interval_b
    if x3 < x1:
        if x4 < x1:
            return 0
        else:
            return min(x2,x4) - x1
    else:
        if x2 < x3:
            return 0
        else:
            return min(x2,x4) - x3

def compute_overlap(a, b):
    """
    Code originally from https://github.com/rbgirshick/py-faster-rcnn.
    Parameters
    ----------
    a: (N, 4) ndarray of float
    b: (K, 4) ndarray of float
    Returns
    -------
    overlaps: (N, K) ndarray of overlap between boxes and query_boxes
    """
    area = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])

    iw = np.minimum(np.expand_dims(a[:, 2], axis=1), b[:, 2]) - np.maximum(np.expand_dims(a[:, 0], 1), b[:, 0])
    ih = np.minimum(np.expand_dims(a[:, 3], axis=1), b[:, 3]) - np.maximum(np.expand_dims(a[:, 1], 1), b[:, 1])

    iw = np.maximum(iw, 0)
    ih = np.maximum(ih, 0)

    ua = np.expand_dims((a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1]), axis=1) + area - iw * ih

    ua = np.maximum(ua, np.finfo(float).eps)

    intersection = iw * ih

    return intersection / ua  
        
def bbox_iou(box1, box2):
    intersect_w = _interval_overlap([box1.xmin, box1.xmax], [box2.xmin, box2.xmax])
    intersect_h = _interval_overlap([box1.ymin, box1.ymax], [box2.ymin, box2.ymax])
    intersect = intersect_w * intersect_h
    w1, h1 = box1.xmax-box1.xmin, box1.ymax-box1.ymin
    w2, h2 = box2.xmax-box2.xmin, box2.ymax-box2.ymin
    union = w1*h1 + w2*h2 - intersect
    return float(intersect) / union

def do_nms(boxes, nms_thresh):
    if len(boxes) > 0:
        nb_class = len(boxes[0].classes)
    else:
        return
    for c in range(nb_class):
        sorted_indices = np.argsort([-box.classes[c] for box in boxes])
        for i in range(len(sorted_indices)):
            index_i = sorted_indices[i]
            if boxes[index_i].classes[c] == 0: continue
            for j in range(i+1, len(sorted_indices)):
                index_j = sorted_indices[j]
                if bbox_iou(boxes[index_i], boxes[index_j]) >= nms_thresh:
                    boxes[index_j].classes[c] = 0

def generate_colors(class_names):
    hsv_tuples = [(x / len(class_names), 1., 1.) for x in range(len(class_names))]
    colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
    colors = list(map(lambda x: (int(x[0]*255), int(x[1]*255), int(x[2]*255)), colors))
    random.seed(10101)  # Fixed seed for consistent colors across runs.
    random.shuffle(colors)  # Shuffle colors to decorrelate adjacent classes.
    random.seed(None)  # Reset seed to default.
    return colors

# get all of the results above a threshold
def get_boxes(boxes, labels, thresh, colors):
    v_boxes, v_labels, v_scores, v_colors = list(), list(), list(), list()
    # enumerate all boxes
    for box in boxes:
        # enumerate all possible labels
        for i in range(len(labels)):

            # check if the threshold for this label is high enough
            if box.classes[i] > thresh:
                v_boxes.append(box)
                v_labels.append(labels[i])
                v_scores.append(box.classes[i]*100)
                v_colors.append(colors[i])
                # don't break, many labels may trigger for one box
    return v_boxes, v_labels, v_scores, v_colors



def compose(*funcs):
    if funcs:
        return reduce(lambda f, g: lambda *a, **kw: g(f(*a, **kw)), funcs)
    else:
        raise ValueError('Composition of empty sequence not supported.')

def letterbox_image(image, size):
    iw, ih = image.size
    w, h = size
    scale = min(w/iw, h/ih)
    nw = int(iw*scale)
    nh = int(ih*scale)

    image = image.resize((nw,nh), Image.BICUBIC)
    new_image = Image.new('RGB', size, (128,128,128))
    new_image.paste(image, ((w-nw)//2, (h-nh)//2))
    return new_image

def rand(a=0, b=1):
    return np.random.rand()*(b-a) + a

def merge_bboxes(bboxes, cutx, cuty):
    merge_bbox = []
    for i in range(len(bboxes)):
        for box in bboxes[i]:
            tmp_box = []
            x1,y1,x2,y2 = box[0], box[1], box[2], box[3]

            if i == 0:
                if y1 > cuty or x1 > cutx:
                    continue
                if y2 >= cuty and y1 <= cuty:
                    y2 = cuty
                    if y2-y1 < 5:
                        continue
                if x2 >= cutx and x1 <= cutx:
                    x2 = cutx
                    if x2-x1 < 5:
                        continue
                
            if i == 1:
                if y2 < cuty or x1 > cutx:
                    continue

                if y2 >= cuty and y1 <= cuty:
                    y1 = cuty
                    if y2-y1 < 5:
                        continue
                
                if x2 >= cutx and x1 <= cutx:
                    x2 = cutx
                    if x2-x1 < 5:
                        continue

            if i == 2:
                if y2 < cuty or x2 < cutx:
                    continue

                if y2 >= cuty and y1 <= cuty:
                    y1 = cuty
                    if y2-y1 < 5:
                        continue

                if x2 >= cutx and x1 <= cutx:
                    x1 = cutx
                    if x2-x1 < 5:
                        continue

            if i == 3:
                if y1 > cuty or x2 < cutx:
                    continue

                if y2 >= cuty and y1 <= cuty:
                    y2 = cuty
                    if y2-y1 < 5:
                        continue

                if x2 >= cutx and x1 <= cutx:
                    x1 = cutx
                    if x2-x1 < 5:
                        continue

            tmp_box.append(x1)
            tmp_box.append(y1)
            tmp_box.append(x2)
            tmp_box.append(y2)
            tmp_box.append(box[-1])
            merge_bbox.append(tmp_box)
    return merge_bbox

def get_random_data_with_Mosaic(annotation_line, input_shape, max_boxes=100, hue=.1, sat=1.5, val=1.5):
    '''random preprocessing for real-time data augmentation'''
    h, w = input_shape
    min_offset_x = 0.3
    min_offset_y = 0.3
    scale_low = 1-min(min_offset_x,min_offset_y)
    scale_high = scale_low+0.2

    image_datas = [] 
    box_datas = []
    index = 0

    place_x = [0,0,int(w*min_offset_x),int(w*min_offset_x)]
    place_y = [0,int(h*min_offset_y),int(h*min_offset_y),0]

    for line in annotation_line:
        # 每一行进行分割
        line_content = line.split()
        # 打开图片
        image = Image.open(line_content[0])
        image = image.convert("RGB") 
        # 图片的大小
        iw, ih = image.size
        # 保存框的位置
        box = np.array([np.array(list(map(int,box.split(',')))) for box in line_content[1:]])
        
        # 是否翻转图片
        flip = rand()<.5
        if flip and len(box)>0:
            image = image.transpose(Image.FLIP_LEFT_RIGHT)
            box[:, [0,2]] = iw - box[:, [2,0]]

        # 对输入进来的图片进行缩放
        new_ar = w/h
        scale = rand(scale_low, scale_high)
        if new_ar < 1:
            nh = int(scale*h)
            nw = int(nh*new_ar)
        else:
            nw = int(scale*w)
            nh = int(nw/new_ar)
        image = image.resize((nw,nh), Image.BICUBIC)

        # 进行色域变换
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
        image = cv2.cvtColor(x, cv2.COLOR_HSV2RGB) # numpy array, 0 to 1

        image = Image.fromarray((image*255).astype(np.uint8))
        # 将图片进行放置，分别对应四张分割图片的位置
        dx = place_x[index]
        dy = place_y[index]
        new_image = Image.new('RGB', (w,h), (128,128,128))
        new_image.paste(image, (dx, dy))
        image_data = np.array(new_image)/255

        index = index + 1
        box_data = []
        # 对box进行重新处理
        if len(box)>0:
            np.random.shuffle(box)
            box[:, [0,2]] = box[:, [0,2]]*nw/iw + dx
            box[:, [1,3]] = box[:, [1,3]]*nh/ih + dy
            box[:, 0:2][box[:, 0:2]<0] = 0
            box[:, 2][box[:, 2]>w] = w
            box[:, 3][box[:, 3]>h] = h
            box_w = box[:, 2] - box[:, 0]
            box_h = box[:, 3] - box[:, 1]
            box = box[np.logical_and(box_w>1, box_h>1)]
            box_data = np.zeros((len(box),5))
            box_data[:len(box)] = box
        
        image_datas.append(image_data)
        box_datas.append(box_data)

    # 将图片分割，放在一起
    cutx = np.random.randint(int(w*min_offset_x), int(w*(1 - min_offset_x)))
    cuty = np.random.randint(int(h*min_offset_y), int(h*(1 - min_offset_y)))

    new_image = np.zeros([h,w,3])
    new_image[:cuty, :cutx, :] = image_datas[0][:cuty, :cutx, :]
    new_image[cuty:, :cutx, :] = image_datas[1][cuty:, :cutx, :]
    new_image[cuty:, cutx:, :] = image_datas[2][cuty:, cutx:, :]
    new_image[:cuty, cutx:, :] = image_datas[3][:cuty, cutx:, :]

    # 对框进行进一步的处理
    new_boxes = merge_bboxes(box_datas, cutx, cuty)

    # 将box进行调整
    box_data = np.zeros((max_boxes,5))
    if len(new_boxes)>0:
        if len(new_boxes)>max_boxes: new_boxes = new_boxes[:max_boxes]
        box_data[:len(new_boxes)] = new_boxes
    return new_image, box_data


def get_random_data(annotation_line, input_shape, max_boxes=100, jitter=.3, hue=.1, sat=1.5, val=1.5, random=True):
    '''random preprocessing for real-time data augmentation'''
    line = annotation_line.split()
    image = Image.open(line[0])
    iw, ih = image.size
    h, w = input_shape
    box = np.array([np.array(list(map(int,box.split(',')))) for box in line[1:]])

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


def cosine_decay_with_warmup(global_step,
                             learning_rate_base,
                             total_steps,
                             warmup_learning_rate=0.0,
                             warmup_steps=0,
                             hold_base_rate_steps=0,
                             min_learn_rate=0,
                             ):
    """
    参数：
            global_step: 上面定义的Tcur，记录当前执行的步数。
            learning_rate_base：预先设置的学习率，当warm_up阶段学习率增加到learning_rate_base，就开始学习率下降。
            total_steps: 是总的训练的步数，等于epoch*sample_count/batch_size,(sample_count是样本总数，epoch是总的循环次数)
            warmup_learning_rate: 这是warm up阶段线性增长的初始值
            warmup_steps: warm_up总的需要持续的步数
            hold_base_rate_steps: 这是可选的参数，即当warm up阶段结束后保持学习率不变，知道hold_base_rate_steps结束后才开始学习率下降
    """
    if total_steps < warmup_steps:
        raise ValueError('total_steps must be larger or equal to '
                            'warmup_steps.')
    #这里实现了余弦退火的原理，设置学习率的最小值为0，所以简化了表达式
    learning_rate = 0.5 * learning_rate_base * (1 + np.cos(np.pi *
        (global_step - warmup_steps - hold_base_rate_steps) / float(total_steps - warmup_steps - hold_base_rate_steps)))
    #如果hold_base_rate_steps大于0，表明在warm up结束后学习率在一定步数内保持不变
    if hold_base_rate_steps > 0:
        learning_rate = np.where(global_step > warmup_steps + hold_base_rate_steps,
                                    learning_rate, learning_rate_base)
    if warmup_steps > 0:
        if learning_rate_base < warmup_learning_rate:
            raise ValueError('learning_rate_base must be larger or equal to '
                                'warmup_learning_rate.')
        #线性增长的实现
        slope = (learning_rate_base - warmup_learning_rate) / warmup_steps
        warmup_rate = slope * global_step + warmup_learning_rate
        #只有当global_step 仍然处于warm up阶段才会使用线性增长的学习率warmup_rate，否则使用余弦退火的学习率learning_rate
        learning_rate = np.where(global_step < warmup_steps, warmup_rate,
                                    learning_rate)

    learning_rate = max(learning_rate,min_learn_rate)
    return learning_rate


class WarmUpCosineDecayScheduler(keras.callbacks.Callback):
    """
    继承Callback，实现对学习率的调度
    """
    def __init__(self,
                 learning_rate_base,
                 total_steps,
                 global_step_init=0,
                 warmup_learning_rate=0.0,
                 warmup_steps=0,
                 hold_base_rate_steps=0,
                 min_learn_rate=0,
                 # interval_epoch代表余弦退火之间的最低点
                 interval_epoch=[0.05, 0.15, 0.30, 0.50],
                 verbose=0):
        super(WarmUpCosineDecayScheduler, self).__init__()
        # 基础的学习率
        self.learning_rate_base = learning_rate_base
        # 热调整参数
        self.warmup_learning_rate = warmup_learning_rate
        # 参数显示  
        self.verbose = verbose
        # learning_rates用于记录每次更新后的学习率，方便图形化观察
        self.min_learn_rate = min_learn_rate
        self.learning_rates = []

        self.interval_epoch = interval_epoch
        # 贯穿全局的步长
        self.global_step_for_interval = global_step_init
        # 用于上升的总步长
        self.warmup_steps_for_interval = warmup_steps
        # 保持最高峰的总步长
        self.hold_steps_for_interval = hold_base_rate_steps
        # 整个训练的总步长
        self.total_steps_for_interval = total_steps

        self.interval_index = 0
        # 计算出来两个最低点的间隔
        self.interval_reset = [self.interval_epoch[0]]
        for i in range(len(self.interval_epoch)-1):
            self.interval_reset.append(self.interval_epoch[i+1]-self.interval_epoch[i])
        self.interval_reset.append(1-self.interval_epoch[-1])

	#更新global_step，并记录当前学习率
    def on_batch_end(self, batch, logs=None):
        self.global_step = self.global_step + 1
        self.global_step_for_interval = self.global_step_for_interval + 1
        lr = K.get_value(self.model.optimizer.lr)
        self.learning_rates.append(lr)

	#更新学习率
    def on_batch_begin(self, batch, logs=None):
        # 每到一次最低点就重新更新参数
        if self.global_step_for_interval in [0]+[int(i*self.total_steps_for_interval) for i in self.interval_epoch]:
            self.total_steps = self.total_steps_for_interval * self.interval_reset[self.interval_index]
            self.warmup_steps = self.warmup_steps_for_interval * self.interval_reset[self.interval_index]
            self.hold_base_rate_steps = self.hold_steps_for_interval * self.interval_reset[self.interval_index]
            self.global_step = 0
            self.interval_index += 1

        lr = cosine_decay_with_warmup(global_step=self.global_step,
                                      learning_rate_base=self.learning_rate_base,
                                      total_steps=self.total_steps,
                                      warmup_learning_rate=self.warmup_learning_rate,
                                      warmup_steps=self.warmup_steps,
                                      hold_base_rate_steps=self.hold_base_rate_steps,
                                      min_learn_rate = self.min_learn_rate)
        K.set_value(self.model.optimizer.lr, lr)
        if self.verbose > 0:
            print('\nBatch %05d: setting learning '
                  'rate to %s.' % (self.global_step + 1, lr))
