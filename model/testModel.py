'''
================================================================
   Editor      : Pycharm
   File name   : testModel
   Author      : LLLLLuuusa(HuangDaxu)
   Created date: 2022/4/30
   Email       : 1095663821@qq.com
   QQ          : 1095663821
   Description :

   (/≧▽≦)/ long mine the sun shine!!!
================================================================
'''
from tensorflow.keras.utils import plot_model
from model.resNet import *
import tensorflow as tf

print(tf.__version__)
print(tf.test.is_gpu_available())
print(tf.test.gpu_device_name())

if __name__ == '__main__':
    x = tf.random.normal([1,224,224,3])
    model = resnet50(landmarks_num=196)

    model.build(input_shape=(None,224,224,3))
    model.summary()
    plot_model(model,
               to_file='resNet模型图.png',
               show_shapes=True)

    landmarks, gender, age = model(x, training=False)
    print(landmarks,gender,age)



