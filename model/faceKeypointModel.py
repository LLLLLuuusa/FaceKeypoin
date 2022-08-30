'''
================================================================
   Editor      : Pycharm
   File name   : faceKeypointModel
   Author      : LLLLLuuusa(HuangDaxu)
   Created date: 2022/4/30
   Email       : 1095663821@qq.com
   QQ          : 1095663821
   Description :

   (/≧▽≦)/ long mine the sun shine!!!
================================================================
'''
from model.resNet import *
from tensorflow.keras import Sequential,layers

class FaceKeypointModel(tf.keras.Model):
    def __init__(self,landmarks_num=1000):
        self.resnet = resnet50()

        self.dropout1 = layers.Dropout(self.dropout_factor)
        self.dropout2 = layers.Dropout(0.8)
        self.dropout3 = layers.Dropout(0.65)

        self.fc_landmarks_1 = layers.Dense(1024)
        self.fc_landmarks_2 = layers.Dense(landmarks_num)

        self.fc_gender_1 = layers.Dense(64)
        self.fc_gender_2 = layers.Dense(2)

        self.fc_age_1 = layers.Dense(64)
        self.fc_age_2 = layers.Dense(1)

    def call(self,inputs):
        landmarks = self.fc_landmarks_1(inputs)
        landmarks = self.dropout1(landmarks)
        landmarks = self.fc_landmarks_2(landmarks)

        gender = self.fc_gender_1(inputs)
        gender = self.dropout2(gender)
        gender = self.fc_gender_2(gender)

        age = self.fc_age_1(inputs)
        age = self.dropout3(age)
        age = self.fc_age_2(age)

        return landmarks, gender, age
