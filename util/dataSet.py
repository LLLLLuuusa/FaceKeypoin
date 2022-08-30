'''
================================================================
   Editor      : Pycharm
   File name   : dataset
   Author      : LLLLLuuusa(HuangDaxu)
   Created date: 2022/4/29
   Email       : 1095663821@qq.com
   QQ          : 1095663821
   Description :

   (/≧▽≦)/ long mine the sun shine!!!
================================================================
'''
# 创建db文件
import json
import os
import random

import cv2
import numpy as np
import tensorflow as tf
from util.dataAug import *


class Dataloader():
    def __init__(self, ops ,img_size=(224,224), flag_agu = False,fix_res = True,vis = False):
        self.ops = ops
        self.max_age=0
        self.min_age=65535.
        self.img_size=img_size
        self.flag_agu=flag_agu
        self.fix_res=fix_res


    def __call__(self):
        db = tf.data.Dataset.from_generator(self.gen,output_signature=(
            tf.TensorSpec(shape=[224,224,3], dtype=tf.float32),
            # tf.RaggedTensorSpec(shape=[98,2], dtype=tf.float32),
            tf.TensorSpec(shape=(196,), dtype=tf.float32),
            tf.TensorSpec(shape=(), dtype=tf.int8),
            tf.TensorSpec(shape=(), dtype=tf.float32)
        ))
        db = db.batch(self.ops.batch_size,drop_remainder=True)
        return db

    def __len__(self):
        return int(len(os.listdir(self.ops.train_path))/self.ops.batch_size)

    def gen(self):
        #file_list = []
        # landmarks_list = []
        # age_list = []
        # gender_list = []
        train_path=self.ops.train_path
        idx = 0
        #train_path_len = len(os.listdir(train_path))
        for f_ in os.listdir(train_path):
            f = open(train_path + f_, encoding='utf-8')  # 读取 json文件
            dict = json.load(f)
            f.close()

            if dict["age"]>100. or dict["age"]<1.:
                continue
            idx += 1
            #-------------------------------------------------------------------
            #根据json路径找到图片路径,并读取图片
            img_path = (train_path + f_).replace("label_new","image").replace(".json",".jpg")
            #读取图片
            #file_list.append(img_path)
            #打印josn里的maker信息、age信息和当前读取到第几个信息
            #------> maker : 作者 ,age:年龄, <x/总文件数>
            #print("------> maker : {} ,age:{:.3f}, <{}/{}>".format(dict["maker"],dict["age"],idx,train_path_len))

            # 获取josn的关键点数据
            pts = []
            for pt_ in dict["landmarks"]:
                x, y = pt_
                pts.append([x, y])

            # 获取json的年龄数据,并获取最小和最大年龄,(便于后期训练时,年龄统一在数据集的范围内)

            # pts=tf.ragged.constant(pts)

            #数据处理
            img,pts,gender,age=self.process(img_path,pts,dict['gender'],dict['age'])

            yield img,pts,gender,age

    def process(self,img_path,pts,gender,age):
        #img_path 图片路径
        #pts 关键点 (98, 2)
        #gender 性别 ()
        #age 年龄 ()
        # --------------------------gender数据处理----------------------------------------
        # 获取json的性别数据,并格式化,男性为1,女性为0
        if gender == "male":
            # gender_list.append(1.0)
            gender = 1
        else:
            # gender_list.append(0.0)
            gender= 0
        # --------------------------图片读取----------------------------------------
        img = cv2.imread(img_path)  # BGR
        # --------------------------数据增强,关键点归一化----------------------------------------
        # 关键点归一化,65%数据增强--仿射变化
        # if self.flag_agu == True and random.random() > 0.35:
        if self.flag_agu == True and random.random() > 0.35:
            # 获取左右眼的中心点
            left_eye = tf.reduce_mean(pts[60:68],axis=0)
            right_eye = tf.reduce_mean(pts[68:76],axis=0)
            # 随机生成一个-33到33的旋转角度
            angle_random = random.randint(-33, 33)
            # 返回 crop 图 和 归一化 landmarks
            img_, landmarks_ = face_random_rotate(img, pts, angle_random, left_eye, right_eye,
                                                   img_size=self.img_size)
        else:
            x_max = -65535
            y_max = -65535
            x_min = 65535
            y_min = 65535

            for pt_ in pts:
                x_, y_ = int(pt_[0]), int(pt_[1])
                # cv2.circle(img, (int(pt_[0]), int(pt_[1])), 2, (0, 255, 0), -1)
                x_min = x_ if x_min > x_ else x_min
                y_min = y_ if y_min > y_ else y_min
                x_max = x_ if x_max < x_ else x_max
                y_max = y_ if y_max < y_ else y_max
            # ----------------------------------------
            face_w = x_max - x_min
            face_h = y_max - y_min
            x_min = int(x_min - random.randint(-6, int(face_w / 10)))
            y_min = int(y_min - random.randint(-6, int(face_h / 10)))
            x_max = int(x_max + random.randint(-6, int(face_w / 10)))
            y_max = int(y_max + random.randint(-6, int(face_h / 10)))

            x_min = np.clip(x_min, 0, img.shape[1] - 1)
            x_max = np.clip(x_max, 0, img.shape[1] - 1)
            y_min = np.clip(y_min, 0, img.shape[0] - 1)
            y_max = np.clip(y_max, 0, img.shape[0] - 1)

            face_w = x_max - x_min
            face_h = y_max - y_min

            face_crop = img[y_min:y_max, x_min:x_max, :]
            landmarks_ = []
            for pt_ in pts:
                x_, y_ = int(pt_[0]) - x_min, int(pt_[1]) - y_min

                landmarks_.append([float(x_) / float(face_w), float(y_) / float(face_h)])
            img_ = cv2.resize(face_crop, self.img_size, interpolation=random.randint(0, 4))
        # 0.5%数据增强--亮度增强,噪点增加
        if self.flag_agu == True and random.random() > 0.5:
            c = float(random.randint(80,120))/100.
            b = random.randint(-10,10)
            img_ = contrast_img(img_, c, b)

        # 0.3概率随机HSV
        if self.flag_agu == True and random.random() > 0.7:
            # print('agu hue ')
            img_hsv=cv2.cvtColor(img_,cv2.COLOR_BGR2HSV)
            hue_x = random.randint(-10,10)
            # print(cc)
            img_hsv[:,:,0]=(img_hsv[:,:,0]+hue_x)
            img_hsv[:,:,0] =np.maximum(img_hsv[:,:,0],0)
            img_hsv[:,:,0] =np.minimum(img_hsv[:,:,0],180)#范围 0 ~180
            img_=cv2.cvtColor(img_hsv,cv2.COLOR_HSV2BGR)

        # 0.1概率图像变暗
        if self.flag_agu == True and random.random() > 0.9:
            img_ = img_agu_channel_same(img_)

        # --------------------------图片数据归一化,年龄归一化----------------------------------------
        img_ = tf.cast(img_-128., tf.float32) / 256.
        age = (age-50.)/100.
        landmarks_ = np.array(landmarks_).ravel()
        return img_,landmarks_,gender,age



