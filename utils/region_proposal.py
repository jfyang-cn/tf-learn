import cv2
import numpy as np

def slide_window(win_size, tgt_size, win_step=5, scale=1.1):
    
    win_w,win_h = win_size[0],win_size[1]
    win_size0 = (win_w,win_h)
    
    max_score = 0.0
    x,y = 0,0
    win_size = (win_w,win_h)
    
    x1,y1 = 0,0
    mv_size = (win_w,win_h)
    
    tgt_w = tgt_size[0]
    tgt_h = tgt_size[1]
    
    i = 0
    while mv_size[0] <=tgt_w and mv_size[1]<tgt_h:
        x1 = 0
        while x1+mv_size[0]<=tgt_w:
            y1 = 0
            while y1+mv_size[1]<=tgt_h:
                x2 = x1+mv_size[0]
                y2 = y1+mv_size[1]
                
                yield x1,y1,mv_size
                
                i = i+1
                y1 = y1+win_step
            x1 = x1+win_step

        if scale > 0.00001:
            mv_size = int(mv_size[0]*scale),int(mv_size[1]*scale)
        else:
            break

def region_search(src, tgt, win_step=5, scale=1.1):
    
    win_h,win_w = src.shape[0],src.shape[1]
    win_size0 = (win_w,win_h)
    
    max_score = 0.0
    x,y = 0,0
    win_size = (win_w,win_h)
    
    x1,y1 = 0,0
    mv_size = (win_w,win_h)
    
    tgt_w = tgt.shape[1]
    tgt_h = tgt.shape[0]
    
    i = 0
    while mv_size[0] <=tgt_w and mv_size[1]<tgt_h:
        x1 = 0
        while x1+mv_size[0]<=tgt_w:
            y1 = 0
            while y1+mv_size[1]<=tgt_h:
                x2 = x1+mv_size[0]
                y2 = y1+mv_size[1]
                
                dst = tgt[y1:y2,x1:x2]
                dst = cv2.resize(dst,win_size0)
#                 score = mr.mutual_info_score(np.reshape(src, -1), np.reshape(dst, -1))
                yield x1,y1,mv_size,dst
                i = i+1
#                 cv2.imwrite('segs2/%d_%d_%d.jpg' %(i,x1,y1), dst)
#                 if score > max_score:
#                     max_score = score
#                     x,y,win_size = x1,y1,mv_size
                y1 = y1+win_step
            x1 = x1+win_step

        if scale > 0.00001:
            mv_size = int(mv_size[0]*scale),int(mv_size[1]*scale)
        else:
            break

#     print("total segments:", i)