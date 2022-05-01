'''
================================================================
   Editor      : Pycharm
   File name   : inference
   Author      : LLLLLuuusa(HuangDaxu)
   Created date: 2022/4/30
   Email       : 1095663821@qq.com
   QQ          : 1095663821
   Description :

   (/≧▽≦)/ long mine the sun shine!!!
================================================================
'''
from util.commonUtils import setSeed
from util.dataSet import Dataloader
import numpy as np
import tensorflow as tf
import cv2

def trainer(ops):
    #设置种子
    setSeed(ops.seed)
    dataloader=Dataloader(ops,flag_agu=ops.flag_agu)
    db = dataloader()


    # for img,pts,ages,gender in db:
    #     print(np.shape(img),np.shape(pts),np.shape(gender),np.shape(ages))
    img,pts,ages,gender =next(iter(db))
    print(np.shape(img),np.shape(pts),np.shape(gender),np.shape(ages))
    demo_img=img[0]
    demo_img=tf.cast((demo_img*256)+128,tf.uint8).numpy()
    for pt_ in pts[0]:
        cv2.circle(demo_img, (int(pt_[0]*224), int(pt_[1]*224)), 2, (0, 255, 0), -1)
    while True:
        cv2.imshow('crop',demo_img)
        cv2.waitKey(1)