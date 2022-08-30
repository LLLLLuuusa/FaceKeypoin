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
from model.resNet import *
import numpy as np
import tensorflow as tf
import cv2

def inference():
    img = cv2.imread("data/value_face/808_1899-08-13_1955.jpg")

    model = resnet50(landmarks_num=196, dropout_factor=0.5)
    model.build(input_shape=(None, 224, 224, 3))
    model.summary()
    model.load_weights("weight/faceKeypoint")
    # for img,pts,ages,gender in db:
    #     print(np.shape(img),np.shape(pts),np.shape(gender),np.shape(ages))
    # img,pts,ages,gender =next(iter(db))
    # print(np.shape(img),np.shape(pts),np.shape(gender),np.shape(ages))
    # demo_img=img[0]
    # demo_img=tf.cast((demo_img*256)+128,tf.uint8).numpy()

    img_ = tf.cast(img - 128., tf.float32) / 256.
    img_ = tf.expand_dims(img_,axis=0)
    print(np.shape(img))
    pts, genders, ages=model(img_, training=False)
    print(np.shape(pts),np.shape(genders))
    for i in range(int(np.shape(pts)[1]/2)):
        print(pts[0, i * 2 + 0], pts[0, i * 2 + 1])
        x = int(pts[0,i*2+0]*224)
        y = int(pts[0,i*2+1]*224)

        print(x,y)
        cv2.circle(img,(x,y), 2, (0, 255, 0), -1)
        # cv2.circle(img, (int(pt_[0]*224), int(pt_[1]*224)), 2, (0, 255, 0), -1)
    while True:
        cv2.imshow('crop',img)
        cv2.waitKey(1)

if __name__ == '__main__':
    inference()