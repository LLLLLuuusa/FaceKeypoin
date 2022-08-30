'''
================================================================
   Editor      : Pycharm
   File name   : commonUtils
   Author      : LLLLLuuusa(HuangDaxu)
   Created date: 2022/4/29
   Email       : 1095663821@qq.com
   QQ          : 1095663821
   Description :

   (/≧▽≦)/ long mine the sun shine!!!
================================================================
'''
import argparse
import json
import os
import time
import tensorflow as tf
from tensorflow.keras import optimizers
from util.commonUtils import *
from util.dataSet import Dataloader
from model.resNet import *
from loss.loss import *
from tqdm import tqdm, trange
print(tf.__version__)
print(tf.test.is_gpu_available())


def trainer(ops):
    # 设置种子
    setSeed(ops.seed)
    # dataloader=Dataloader(ops,flag_agu=ops.flag_agu)

    # 数据加载
    dataloader = Dataloader(ops, flag_agu=True)
    db = dataloader()

    # 读取模型
    model = resnet50(landmarks_num=ops.num_classes,dropout_factor=ops.dropout)
    model.build(input_shape=(None,224,224,3))
    model.summary()


    #优化器设计
    optimizer = optimizers.Adam(learning_rate=ops.init_lr, beta_1=0.9,beta_2=0.999, epsilon=ops.weight_decay)

    # 生成一个 TensorBoard 的输出
    # writer = tf.summary.create_file_writer("log")

    # 加载 finetune 模型
    # 读取别的模型,进行迁移学习
    if ops.fintune_model != 'None':  # checkpoint
        model.load_weights(ops.fintune_model)
        print(f'读取权重文件 : {ops.fintune_model}')
    print('/**********************************************/')

    # 变量初始化,学习率优化
    best_loss = np.inf
    loss_mean = 0.  # 损失均值
    loss_idx = 0.  # 损失计算计数器
    flag_change_lr_cnt = 0  # 学习率更新计数器
    init_lr = ops.init_lr  # 学习率
    epochs = ops.epochs
    for epoch in range(epochs):
        #print(f'epoch {epoch} ------>>>')
        if loss_mean != 0.:
            if best_loss > (loss_mean / loss_idx):
                flag_change_lr_cnt = 0
                best_loss = (loss_mean / loss_idx)
            else:
                flag_change_lr_cnt += 1

                if flag_change_lr_cnt > 10:
                    init_lr = init_lr * ops.lr_decay
                    optimizer.learninig_rate= init_lr
                    flag_change_lr_cnt = 0

        loss_mean = 0.  # 损失均值
        loss_idx = 0.  # 损失计算计数器

        # 迭代训练
        with trange(len(dataloader)) as t:
            for step in t:
                img, pts, genders, ages = next(iter(db))
                # loss计算
                with tf.GradientTape() as tape:
                    tape.watch(model.trainable_variables)
                    pre_pts, pre_genders, pre_ages = model(img, training=True)
                    loss = faceKeypoinLoss(pre_pts, pre_genders, pre_ages,pts,genders,ages)

                loss_mean += loss.numpy()
                loss_idx += 1.
                # 梯度计算
                grads = tape.gradient(loss, model.trainable_variables)
                # grads, global_norm = tf.clip_by_global_norm(grads, 2)#梯度裁剪,有时候会梯度爆炸不知道为啥,直接给剪了一劳永逸
                optimizer.apply_gradients(zip(grads, model.trainable_variables))
                # 设置进度条右边显示的信息
                des = time.strftime('%m-%d %H:%M:%S', time.localtime(time.time())) + f" Epoch {epoch}"
                #post = f"lr: {init_lr:.5f} total_loss: {loss:.2f}|"
                post = f"loss: {loss:.2f}|"
                t.set_description(des)
                t.set_postfix_str(post)

    model.save_weights(ops.model_exp)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=' 关键点训练 ')
    parser.add_argument('--seed', type=int, default = 15678,
        help = 'seed') # 设置随机种子
    parser.add_argument('--model_exp', type=str, default = 'weight/faceKeypoint',
        help = 'model_exp') # 模型输出文件夹
    parser.add_argument('--model', type=str, default = 'resnet_50',
        help = 'model : resnet_50') # 模型类型
    parser.add_argument('--num_classes', type=int , default = 196,
        help = 'num_classes') #  landmarks 个数*2
    parser.add_argument('--GPUS', type=str, default = '0',
        help = 'GPUS') # GPU选择
    parser.add_argument('--train_path', type=str,
        default = './data/wiki_crop_face/label_new/',
        help = 'train_path')# 训练集标注信息
    parser.add_argument('--pretrained', type=bool, default = True,
        help = 'imageNet_Pretrain') # 初始化学习率
    parser.add_argument('--fintune_model', type=str, default = 'None',
        help = 'fintune_model') # fintune model
    parser.add_argument('--loss_define', type=str, default = 'wing_loss',
        help = 'define_loss') # 损失函数定义
    parser.add_argument('--init_lr', type=float, default = 2e-4,
        help = 'init_learningRate') # 初始化学习率
    parser.add_argument('--lr_decay', type=float, default = 0.5,
        help = 'learningRate_decay') # 学习率权重衰减率
    parser.add_argument('--weight_decay', type=float, default = 5e-4,
        help = 'weight_decay') # 优化器正则损失权重
    parser.add_argument('--momentum', type=float, default = 0.9,
        help = 'momentum') # 优化器动量
    parser.add_argument('--batch_size', type=int, default = 72,
        help = 'batch_size') # 训练每批次图像数量
    parser.add_argument('--dropout', type=float, default = 0.5,
        help = 'dropout') # dropout
    parser.add_argument('--epochs', type=int, default = 50,
        help = 'epochs') # 训练周期
    parser.add_argument('--num_workers', type=int, default = 8,
        help = 'num_workers') # 训练数据生成器线程数
    parser.add_argument('--img_size', type=tuple , default = (256,256),
        help = 'img_size') # 输入模型图片尺寸
    parser.add_argument('--flag_agu', type=bool , default = True,
        help = 'data_augmentation') # 训练数据生成器是否进行数据扩增
    parser.add_argument('--clear_model_exp', type=bool, default = False,
        help = 'clear_model_exp') # 模型输出文件夹是否进行清除

    # --------------------------------------------------------------------------
    args = parser.parse_args()  # 解析添加参数
    # --------------------------------------------------------------------------
    print(f'\n/******************* {parser.description} ******************/\n')

    unparsed = vars(args)  # parse_args()方法的返回值为namespace，用vars()内建函数化为字典
    for key in unparsed.keys():
        print(f'{key} : {unparsed[key]}')

    # 将配置生成对应的json文件
    fs = open(args.model_exp + 'train_ops.json', "w", encoding='utf-8')
    json.dump(unparsed, fs, ensure_ascii=False, indent=1)
    fs.close()

    #/*******************  开始训练  ******************/
    trainer(args)

    print(f'训练结束 : {time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())}')