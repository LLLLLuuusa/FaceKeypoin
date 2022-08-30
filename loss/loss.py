'''
================================================================
   Editor      : Pycharm
   File name   : loss
   Author      : LLLLLuuusa(HuangDaxu)
   Created date: 2022/4/30
   Email       : 1095663821@qq.com
   QQ          : 1095663821
   Description :

   (/≧▽≦)/ long mine the sun shine!!!
================================================================
'''
import tensorflow as tf
import math
import numpy as np


def loss_wing(pred, true, w=0.06, epsilon=0.01):
    # pred:[b,21,2]
    # true:[b,21,2]
    # print(pred.shape)
    # print(true.shape)

    x = pred - true
    c = w * (1.0 - math.log(1.0 + w / epsilon))
    absolute_x = tf.abs(x)

    losses = tf.where( \
        (w > absolute_x), \
        w * tf.math.log(1.0 + absolute_x / epsilon), \
        absolute_x - c)

    losses = tf.reduce_mean(losses, axis=1)
    loss = tf.reduce_mean(losses)

    return loss


def faceKeypoinLoss(pre_pts, pre_genders, pre_ages, pts, genders, ages):
    pts_loss = loss_wing(pre_pts, pts)
    gender_loss = tf.losses.sparse_categorical_crossentropy(genders, pre_genders, from_logits=True)
    gender_loss = tf.reduce_mean(gender_loss)

    ages = tf.expand_dims(ages,axis=-1)
    age_loss = loss_wing(pre_ages, ages)

    loss = pts_loss + 0.3 * age_loss + 0.25 * gender_loss

    return loss
