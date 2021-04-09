import os

dirpath = '/home/philyang/git/dataset/dizhi/17/day'
filelist = os.listdir(dirpath)
for fname in filelist:
    print('%s 1' % (fname))
