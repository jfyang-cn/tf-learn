#! /usr/bin/env python
# -*- coding: utf-8 -*-
import sys,os,argparse
import cv2
import numpy as np
import xml.etree.ElementTree as ET
import xmltodict
import dicttoxml
import xmltodict
import json
from xml.dom.minidom import parseString

# convert VOC annotations to specific size image with padding to remain original w/h ratio


###################################################
# arguments
# 

tgt_width = 416     # target width
tgt_height = 416    # target height

# target dir
tgt_dir = ''

# source dir
img_dir = ''
ann_dir = ''
###################################################

tgt_img_dir = os.path.join(tgt_dir,'val%d' % (tgt_width))
if not os.path.exists(tgt_img_dir):
    os.makedirs(tgt_img_dir)

tgt_ann_dir = os.path.join(tgt_img_dir,'xmls')
if not os.path.exists(tgt_ann_dir):
    os.makedirs(tgt_ann_dir)

def parse_ann(ann_path):
    with open(ann_path) as f:
        ann = xmltodict.parse(f.read(), force_list=('object'))
    return ann

def transform_ann(ann, tgt_width, tgt_height):
    src_width, src_height = int(ann['annotation']['size']['width']), int(ann['annotation']['size']['height'])
    ratio = float(tgt_width/src_width)
#     print(ratio)
    if src_width < src_height:
        ratio = float(tgt_height/src_height)

    ann['annotation']['size']['width'] = tgt_width
    ann['annotation']['size']['height'] = tgt_height

    for obj in ann['annotation']['object']:
        obj['bndbox']['xmin'] = int(round(float(ratio*int(obj['bndbox']['xmin']))))
        obj['bndbox']['ymin'] = int(round(float(ratio*int(obj['bndbox']['ymin']))))
        obj['bndbox']['xmax'] = int(round(float(ratio*int(obj['bndbox']['xmax']))))
        obj['bndbox']['ymax'] = int(round(float(ratio*int(obj['bndbox']['ymax']))))
            
    return ann

def save_ann(ann, ann_path):
#     xml = dicttoxml.dicttoxml(ann,attr_type=False,custom_root='').decode("utf-8") # 默认是byte 类型，转成str。
#     dom = parseString(xml)
    xml_str = xmltodict.unparse(ann,pretty=True,encoding='utf-8')
    f = open(ann_path,'w')
    f.writelines(xml_str)
#     f.writelines(dom.toprettyxml())
    f.close()
    
def img_prep(src_path, imgname, tgt_width, tgt_height, img_dir):
    img = cv2.imread(os.path.join(src_path, imgname))
    img_path = os.path.join(img_dir, imgname)
    
    src_height, src_width, _ = img.shape

    top, bottom, left, right = 0,0,0,0
    if src_width > src_height:
        ratio = float(tgt_width/src_width)
        img = cv2.resize(img, (tgt_width, int(round(float(ratio*src_height)))))
        h, w, _ = img.shape
        bottom = tgt_height-h
    else:
        ratio = float(tgt_height/src_height)
        img = cv2.resize(img, (int(round(float(ratio*src_width))), tgt_width))
        h, w, _ = img.shape
        right = tgt_width-w

#     print(img_path)
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=[0,0,0])
    cv2.imwrite(img_path, img)

def img_process(img_dir, imgname, tgt_width, tgt_height, tgt_img_dir):
    ann_list = os.listdir(ann_dir)
    for annname in ann_list:
        if os.path.isfile(os.path.join(ann_dir, annname)):
            print(os.path.join(ann_dir, annname))
            ann = parse_ann(os.path.join(ann_dir, annname))
            ann = transform_ann(ann, tgt_width, tgt_height)
            save_ann(ann, os.path.join(tgt_ann_dir, annname))
            imgname = ann['annotation']['filename']
            img_prep(img_dir, imgname, tgt_width, tgt_height, tgt_img_dir)
        
def main(args):
    
    tgt_width  = args.tgt_width
    tgt_height = args.tgt_height
    tgt_dir    = args.tgt_dir

    # source dir
    img_dir = args.img_dir
    ann_dir = args.ann_dir

    img_process(img_dir, imgname, tgt_width, tgt_height, tgt_img_dir)


def parse_arguments(argv):
    
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--image_file', type=str,
                        help='image file path for predicting', default=None)
    
    parser.add_argument('--list_file', type=str,
                        help='image shortname list txt file for testing', default=None)
    
    parser.add_argument('--weights', type=str,
                        help='weights file', default=None)
    
    parser.add_argument(
        '-c',
        '--conf',
        help='path to configuration file')
        
    return parser.parse_args(argv)

if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))