import tensorflow as tf
from tensorflow.keras import Sequential,layers
import numpy as np
#tensorflow2.6版本以后,kears和tf已经完全分离,这里会报错,但是不好影响使用,可以不管


# [2*conv+identity],两个卷积层完成后加入短接线
class BasicBlock(layers.Layer):
    def __init__(self, filter_num, stride=1):
        super(BasicBlock, self).__init__()
        self.filter=filter_num
        # 第一层卷积层
        self.conv1 = layers.Conv2D(filter_num,(3, 3), strides=stride, padding='same')
        self.bn1 = layers.BatchNormalization()
        self.relu = layers.Activation('relu')

        # 第二层卷积层
        self.conv2 = layers.Conv2D(filter_num,(3, 3), strides=1, padding='same')
        self.bn2 = layers.BatchNormalization()

        # 判断,如果上面卷积层出现下采样,将短接层进行同样操作
        if stride != 1:
            #self.downsample = Sequential()
            #self.downsample.add(layers.Conv2D(filter_num,(1, 1), strides=stride))
            self.downsample=layers.Conv2D(filter_num, kernel_size=(1, 1), strides=stride, padding='same')
        else:
            self.downsample = lambda x: x

    def call(self, inputs, training=None):
        # 建立第一层卷积层
        out = self.conv1(inputs)
        out = self.bn1(out, training=training)
        out = self.relu(out)

        # 建立第二层卷积层
        out = self.conv2(out)
        out = self.bn2(out, training=training)

        # 建立短接层
        identity = self.downsample(inputs)
        # 将短接层和最后的卷积层连接
        output = layers.add([out, identity])
        output = tf.nn.relu(output)

        return output


class ResNet(tf.keras.Model):
    def __init__(self, layer_dims,landmarks_num=1000,dropout_factor = 0.5):
        super(ResNet, self).__init__()

        self.dropout_factor = dropout_factor

        # 预处理层(卷积层)
        self.stem = Sequential([
            layers.Conv2D(64, kernel_size=(3, 3), strides=(1, 1), padding='same'),
            layers.BatchNormalization(),
            layers.Activation('relu'),
            layers.MaxPool2D(pool_size=(2, 2), strides=(1, 1), padding='same')
        ])

        # 第1个组合层,(1+layer_dims[])*2层
        self.layers1 = self.build_resblock(64,layer_dims[0], stride=1)

        # 第2个组合层,(1+layer_dims[])*2层
        self.layers2 = self.build_resblock(128,layer_dims[1], stride=2)
        # 第3个组合层,(1+layer_dims[])*2层
        self.layers3 = self.build_resblock(256,layer_dims[2], stride=2)
        # 第4个组合层,(1+layer_dims[])*2层
        self.layers4 = self.build_resblock(512,layer_dims[3], stride=2)

        #[b,w,h,c]--->[b,c]
        self.avgpool=layers.GlobalAveragePooling2D()

        self.dropout1 = layers.Dropout(self.dropout_factor)
        self.dropout2 = layers.Dropout(0.8)
        self.dropout3 = layers.Dropout(0.65)

        self.fc_landmarks_1 = layers.Dense(1024)
        self.fc_landmarks_2 = layers.Dense(landmarks_num)

        self.fc_gender_1 = layers.Dense(64)
        self.fc_gender_2 = layers.Dense(2)


        self.fc_age_1 = layers.Dense(64)
        self.fc_age_2 = layers.Dense(1)

    def call(self, inputs, training=None):
        x = self.stem(inputs, training=training)

        x = self.layers1(x, training=training)
        x = self.layers2(x, training=training)
        x = self.layers3(x, training=training)
        x = self.layers4(x, training=training)

        x = self.avgpool(x)

        landmarks = self.fc_landmarks_1(x)
        landmarks = self.dropout1(landmarks)
        landmarks = self.fc_landmarks_2(landmarks)

        gender = self.fc_gender_1(x)
        gender = self.dropout2(gender)
        gender = self.fc_gender_2(gender)

        age = self.fc_age_1(x)
        age = self.dropout3(age)
        age = self.fc_age_2(age)
        return landmarks, gender, age

    # 建立组合层,1+layer_dims
    def build_resblock(self, filter_num, block,stride=1):
        res_blocks = Sequential()

        # 组合层第一层---由下采样为1的卷积层
        res_blocks.add(BasicBlock(filter_num, stride))

        # 组合层第n层
        for _ in range(1,block):
            res_blocks.add(BasicBlock(filter_num,stride=1))

        return res_blocks

def resnet18(landmarks_num=1000,dropout_factor=0.5):
    return ResNet([2, 2, 2, 2],landmarks_num=landmarks_num,dropout_factor=dropout_factor)

def resnet50(landmarks_num=1000,dropout_factor=0.5):
    return ResNet([3, 4, 6, 3],landmarks_num=landmarks_num,dropout_factor=dropout_factor)
