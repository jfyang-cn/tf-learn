import colorsys
import copy
import os
import time
from timeit import default_timer as timer

import numpy as np
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Input
from tensorflow.keras.models import load_model
from PIL import Image, ImageDraw, ImageFont

from nets.yolo4 import yolo_body, yolo_eval, yolo_decodeout
from utils.utils import letterbox_image
import threading
from multiprocessing import Pool

def timestampe():
    millis = int(round(time.time() * 1000))
    return millis


def multi_thread_post(y_out, class_colors, image_h, image_w, i, result_boxes, result_scores, result_classes, result_colors):
    boxes, scores, classes, colors = yolo_decodeout(y_out, image_h, image_w, class_colors)
    result_boxes[i] = boxes
    result_scores[i] = scores
    result_classes[i] = classes
    result_colors[i] = colors
    print(i, classes)
    return boxes, scores, classes, colors

def test_func(i):
    return i

#--------------------------------------------#
#   使用自己训练好的模型预测需要修改2个参数
#   model_path和classes_path都需要修改！
#   如果出现shape不匹配，一定要注意
#   训练时的model_path和classes_path参数的修改
#--------------------------------------------#
class YOLO(object):
    _defaults = {
        "model_path"        : 'model_data/yolo4_weight.h5',
        "anchors_path"      : 'model_data/yolo_anchors.txt',
        "classes_path"      : 'model_data/coco_classes.txt',
        "score"             : 0.5,
        "iou"               : 0.3,
        "max_boxes"         : 100,
        # 显存比较小可以使用416x416
        # 显存比较大可以使用608x608
        "model_image_size"  : (608, 608),
        #---------------------------------------------------------------------#
        #   该变量用于控制是否使用letterbox_image对输入图像进行不失真的resize，
        #   在多次测试后，发现关闭letterbox_image直接resize的效果更好
        #---------------------------------------------------------------------#
        "letterbox_image"   : False,
    }

    @classmethod
    def get_defaults(cls, n):
        if n in cls._defaults:
            return cls._defaults[n]
        else:
            return "Unrecognized attribute name '" + n + "'"

    #---------------------------------------------------#
    #   初始化yolo
    #---------------------------------------------------#
    def __init__(self, **kwargs):
        self.__dict__.update(self._defaults)
        self.class_names = self._get_class()
        self.anchors = self._get_anchors()
        self.generate()
        self.pool = Pool(16)


    #---------------------------------------------------#
    #   获得所有的分类
    #---------------------------------------------------#
    def _get_class(self):
        classes_path = os.path.expanduser(self.classes_path)
        with open(classes_path) as f:
            class_names = f.readlines()
        class_names = [c.strip() for c in class_names]
        return class_names

    #---------------------------------------------------#
    #   获得所有的先验框
    #---------------------------------------------------#
    def _get_anchors(self):
        anchors_path = os.path.expanduser(self.anchors_path)
        with open(anchors_path) as f:
            anchors = f.readline()
        anchors = [float(x) for x in anchors.split(',')]
        return np.array(anchors).reshape(-1, 2)

    #---------------------------------------------------#
    #   载入模型
    #---------------------------------------------------#    
    def generate(self):
        model_path = os.path.expanduser(self.model_path)
        assert model_path.endswith('.h5'), 'Keras model or weights must be a .h5 file.'
        
        #---------------------------------------------------#
        #   计算先验框的数量和种类的数量
        #---------------------------------------------------#
        num_anchors = len(self.anchors)
        num_classes = len(self.class_names)

        #---------------------------------------------------------#
        #   载入模型，如果原来的模型里已经包括了模型结构则直接载入。
        #   否则先构建模型再载入
        #---------------------------------------------------------#
        try:
            print("load model")
            self.yolo_model = load_model(model_path, compile=False)
        except:
            print("build model")
            self.yolo_model = yolo_body(Input(shape=(self.model_image_size[0],self.model_image_size[1],3)), 
                                        num_anchors//3, num_classes)
            self.yolo_model.load_weights(self.model_path)
        else:
            assert self.yolo_model.layers[-1].output_shape[-1] == \
                num_anchors/len(self.yolo_model.output) * (num_classes + 5), \
                'Mismatch between model and given anchor and class sizes'

        print('{} model, anchors, and classes loaded.'.format(model_path))
            
        self.yolo_model.summary()

        # 画框设置不同的颜色
        hsv_tuples = [(x / len(self.class_names), 1., 1.)
                      for x in range(len(self.class_names))]
        self.colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
        self.colors = list(
            map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)),
                self.colors))

        # 打乱颜色
        np.random.seed(10101)
        np.random.shuffle(self.colors)
        np.random.seed(None)
        
        return 

    #---------------------------------------------------#
    #   检测图片
    #---------------------------------------------------#
    def detect_image(self, image):
        #---------------------------------------------------------#
        #   给图像增加灰条，实现不失真的resize
        #   也可以直接resize进行识别
        #---------------------------------------------------------#
        if self.letterbox_image:
            boxed_image = letterbox_image(image, (self.model_image_size[1],self.model_image_size[0]))
        else:
            boxed_image = image.convert('RGB')
            boxed_image = boxed_image.resize((self.model_image_size[1],self.model_image_size[0]), Image.BICUBIC)
        image_data = np.array(boxed_image, dtype='float32')
        image_data /= 255.
        #---------------------------------------------------------#
        #   添加上batch_size维度
        #---------------------------------------------------------#
#         image_data = np.expand_dims(image_data, 0)
#         print(image_data.shape)

        #---------------------------------------------------------#
        #   将图像输入网络当中进行预测！
        #---------------------------------------------------------#
#         out_boxes, out_scores, out_classes = self.sess.run(
#             [self.boxes, self.scores, self.classes],
#             feed_dict={
#                 self.yolo_model.input: image_data,
#                 self.input_image_shape: [image.size[1], image.size[0]]})

#         print('Found {} boxes for {}'.format(len(out_boxes), 'img'))

        image_datas = [image_data,image_data]        
        yret = self.yolo_model.predict_on_batch(np.array(image_datas))
        print(np.array(yret[0]).shape, np.array(yret[1]).shape, np.array(yret[2]).shape)

        #---------------------------------------------------------#
        #   设置字体
        #---------------------------------------------------------#
        font = ImageFont.truetype(font='font/simhei.ttf',
                    size=np.floor(3e-2 * image.size[1] + 0.5).astype('int32'))

        thickness = max((image.size[0] + image.size[1]) // 300, 1)

        for i, c in list(enumerate(out_classes)):
            predicted_class = self.class_names[c]
            box = out_boxes[i]
            score = out_scores[i]

            top, left, bottom, right = box
            top = top - 5
            left = left - 5
            bottom = bottom + 5
            right = right + 5

            top = max(0, np.floor(top + 0.5).astype('int32'))
            left = max(0, np.floor(left + 0.5).astype('int32'))
            bottom = min(image.size[1], np.floor(bottom + 0.5).astype('int32'))
            right = min(image.size[0], np.floor(right + 0.5).astype('int32'))

            # 画框框
            label = '{} {:.2f}'.format(predicted_class, score)
            draw = ImageDraw.Draw(image)
            label_size = draw.textsize(label, font)
            label = label.encode('utf-8')
            print(label, top, left, bottom, right)
            
            if top - label_size[1] >= 0:
                text_origin = np.array([left, top - label_size[1]])
            else:
                text_origin = np.array([left, top + 1])

            for i in range(thickness):
                draw.rectangle(
                    [left + i, top + i, right - i, bottom - i],
                    outline=self.colors[c])
            draw.rectangle(
                [tuple(text_origin), tuple(text_origin + label_size)],
                fill=self.colors[c])
            draw.text(text_origin, str(label,'UTF-8'), fill=(0, 0, 0), font=font)
            del draw

        return image
    
    def detect_on_batch(self, images):
        
        tick0 = timestampe()
        print("tick0:", tick0)
        #---------------------------------------------------------#
        #   给图像增加灰条，实现不失真的resize
        #   也可以直接resize进行识别
        #---------------------------------------------------------#
        image_datas = []
        for image in images:
            if self.letterbox_image:
                boxed_image = letterbox_image(image, (self.model_image_size[1],self.model_image_size[0]))
            else:
                boxed_image = image.convert('RGB')
                boxed_image = boxed_image.resize((self.model_image_size[1],self.model_image_size[0]), Image.BICUBIC)
            image_data = np.array(boxed_image, dtype='float32')
            image_data /= 255.
            image_datas.append(image_data)


        tick1 = timestampe()
        print("tick1:", tick1, tick1-tick0)
#         y_outs = self.yolo_model.predict(np.array(image_datas), batch_size=16, max_queue_size=64, workers=16, use_multiprocessing=True)
        y_outs = self.yolo_model(np.array(image_datas))
        y_outs = [a.numpy() for a in y_outs]
        tick2 = timestampe()
        print("tick2:", tick2, tick2-tick1)
        
#         print("output:",[a.shape for a in y_outs])
        
        # 多线程
        batch_size = len(images)
        print('batch_size', batch_size)
        boxes_list, scores_list, classes_list, colors_list = [None] * batch_size, [None] * batch_size, [None] * batch_size, [None] * batch_size
#         threads = []
#         for i in range(len(images)):
#             t = threading.Thread(target=self.multi_thread_post, args=([a[i] for a in y_outs],
#                 images[i].size[0], images[i].size[1], i, boxes_list, scores_list, classes_list, colors_list))
#             threads.append(t)
#             t.start()
#         # 等待所有线程任务结束。
#         for t in threads:
#             t.join()

        # 多进程
#         res = [pool.apply_async(target=job, (i,)) for i in range(3)]
#         with self.pool:
        res = [self.pool.apply_async(multi_thread_post, ([a[i] for a in y_outs], self.colors, images[i].size[1], images[i].size[0], i, boxes_list, scores_list, classes_list, colors_list)) for i in range(batch_size)]

#         res = [self.pool.apply_async(test_func, (i,)) for i in range(batch_size)]
#         print([r.get() for r in res])
        for i,r in enumerate(res):
            boxes_list[i], scores_list[i], classes_list[i], colors_list[i] = r.get()
        
        print(boxes_list, scores_list, classes_list, colors_list)

#         boxes_list = []
#         scores_list = []
#         classes_list = []
#         colors_list = []
#         num_classes = len(self.class_names)
#         for i in range(len(images)):
#             y_out = [a[i] for a in y_outs]
# #             print("output:",[a.shape for a in y_out])
# #             boxes, scores, classes = yolo_eval(y_out, self.anchors,
# #                 num_classes, self.input_image_shape, max_boxes = self.max_boxes,
# #                 score_threshold = self.score, iou_threshold = self.iou, letterbox_image = self.letterbox_image)
#             boxes, scores, classes, colors = yolo_decodeout(y_out, images[i].size[0], images[i].size[1], self.colors)
# #             print(boxes, scores, classes, colors)
#             boxes_list.append(boxes)
#             scores_list.append(scores)
#             classes_list.append(classes)
#             colors_list.append(colors)
        tick3 = timestampe()
        print("tick3:", tick3, tick3-tick2)
        
        for j,image in enumerate(images):
            
            #---------------------------------------------------------#
            #   设置字体
            #---------------------------------------------------------#
            font = ImageFont.truetype(font='font/simhei.ttf',
                    size=np.floor(3e-2 * image.size[1] + 0.5).astype('int32'))

            thickness = max((image.size[0] + image.size[1]) // 300, 1)
            
            out_boxes = boxes_list[j]
            out_scores = scores_list[j]
            out_classes = classes_list[j]
            out_color = colors_list[j]
            print(out_classes)
            for k, c in enumerate(out_classes):
                predicted_class = c
                box = out_boxes[k]
                score = out_scores[k]

                top, left, bottom, right = box.ymin, box.xmin, box.ymax, box.xmax
                top = top - 5
                left = left - 5
                bottom = bottom + 5
                right = right + 5

                top = max(0, np.floor(top + 0.5).astype('int32'))
                left = max(0, np.floor(left + 0.5).astype('int32'))
                bottom = min(image.size[1], np.floor(bottom + 0.5).astype('int32'))
                right = min(image.size[0], np.floor(right + 0.5).astype('int32'))

                # 画框框
                label = '{} {:.2f}'.format(predicted_class, score)
                draw = ImageDraw.Draw(image)
                label_size = draw.textsize(label, font)
                label = label.encode('utf-8')
#                 print(label, top, left, bottom, right)

                if top - label_size[1] >= 0:
                    text_origin = np.array([left, top - label_size[1]])
                else:
                    text_origin = np.array([left, top + 1])

                for i in range(thickness):
                    draw.rectangle(
                        [left + i, top + i, right - i, bottom - i],
                        outline=out_color[k])
                draw.rectangle(
                    [tuple(text_origin), tuple(text_origin + label_size)],
                    fill=out_color[k])
                draw.text(text_origin, str(label,'UTF-8'), fill=(0, 0, 0), font=font)
                del draw

        return images

    def close_session(self):
#         self.sess.close()
        pass
