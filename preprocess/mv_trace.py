#! /usr/bin/env python
# -*- coding: utf-8 -*-
import sys,os,argparse
import cv2
import numpy as np

def gen_image(video_path, out_dir, thresh, img_size=(224,224)):
    cap = cv2.VideoCapture(video_path)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) # get how many frames this video have
#     print(frame_count)

    start_i = 0
    end_i = min(frame_count, 120000)
#     print(start_i, end_i)
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_i) # jump to that index
    ret, frame0 = cap.read()
    # img_size = (frame0.shape[1],frame0.shape[0])
#     img_size = (224,224)
    frame0 = cv2.resize(frame0,img_size)
    gray0 = cv2.cvtColor(frame0, cv2.COLOR_BGR2GRAY)

    shortname = os.path.split(video_path)[1]
    shortname = os.path.splitext(shortname)[0]
    print(shortname)

    frame_out = np.zeros([img_size[1], img_size[0], 3],np.uint8)
#     print(frame_out.shape)

    for i in range(start_i+1,end_i):
        cap.set(cv2.CAP_PROP_POS_FRAMES, i) # jump to that index
        ret, frame1 = cap.read()

        if ret == False:
            continue

    #     out = cv2.subtract(frame1, frame0)
    #         frame0 = frame1

        frame1 = cv2.resize(frame1, img_size)
        gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
        out = cv2.subtract(gray1, gray0)
        gray0 = gray1

    #     dst = cv2.GaussianBlur(out,(3,3),0.9)
    #     #高斯双边滤波
    #     out=cv2.bilateralFilter(dst,4,7,7)

        area_mask = (out > thresh).astype(np.uint8)

        # draw the masked image
        area_mask = area_mask.astype(np.bool)
        n = frame1[area_mask].shape[0]
    #     color_mask = np.full((n), [0,0,255], dtype=np.uint8)
        color_mask = np.array([0, 0, 255], dtype=np.uint8)
        frame_out[area_mask] = color_mask

#         video_writer.write(np.uint8(frame_out))
    #     frame_out = np.zeros([img_size[1], img_size[0], 3],np.uint8)

    cap.release()
#     video_writer.release()
    cv2.imwrite('%s/%s.jpg' % (out_dir,shortname), frame_out)

def mv_trace(list_file, out_dir, thresh):
    
    with open(list_file, 'r') as f:
        lines = f.readlines()

    for line in lines:
        line = line.rstrip('\n')
        video_path = line.split(' ')[0]
#         y_true = line.split(' ')[1]

        gen_image(video_path, out_dir, thresh)
            
def main(args):
    
    list_file  = args.list_file
    image_file = args.image_file
    out_dir    = args.out_dir
    thresh     = args.thresh

    mv_trace(list_file, out_dir, thresh)


def parse_arguments(argv):
    
    parser = argparse.ArgumentParser()
    
    parser.add_argument('-i', '--image_file', type=str,
                        help='image file path for predicting', default=None)
    
    parser.add_argument('-l', '--list_file', type=str,
                        help='image file path list', default=None)
    
    parser.add_argument('-o', '--out_dir', type=str,
                        help='image output dir', default='./')
    
    parser.add_argument('-t', '--thresh', type=int,
                        help='threshold for movement detection', default=20)
        
    return parser.parse_args(argv)

if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))