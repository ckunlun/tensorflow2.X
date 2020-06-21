# reference：
# He, Kaiming, et al. "Deep residual learning for image recognition."
# arXiv preprint arXiv:1512.03385 (2015)
# Author:ckunlun
# Time:2020.6.20

import tensorflow as tf
import tensorflow.keras as keras
import tensorflow.keras.layers as layers


class BasicResBlock(keras.Model):
    """定义基本ResNet类
    输入：filter_num, kernel_size, 
        strides
    输入和输出保持一致，即x和f(x)大小维度都一样，
    从而实现简单相加。y=f(x)+x
    参数默认值：filter_num=[64, 64]
    kernel_size=[3, 3], strides=[1, 1]
    """

    def __init__(self, filter_num=[64, 64], kernel_size=[3, 3], strides=[1, 1]):
        super(BasicResBlock, self).__init__()

        # 初始化滤波器个数、滤波器大小、和步长
        filter_num1, filter_num2 = filter_num
        kernel_size1, kernel_size2 = kernel_size
        strides1, strides2 = strides

        # 定义卷积层
        self.cnn1 = layers.Conv2D(filters=filter_num1, kernel_size=kernel_size1,
                                  strides=strides1, padding='same', activation='relu')
        self.bn1 = layers.BatchNormalization()
        self.cnn2 = layers.Conv2D(filters=filter_num2, kernel_size=kernel_size2,
                                  strides=strides2, padding='same', activation=None)
        self.bn2 = layers.BatchNormalization()

    def call(self, input_tensor, training=False):
        x = self.cnn1(input_tensor)
        x = self.bn1(x)
        x = self.cnn2(x)
        x = self.bn2(x)
        out = tf.nn.relu(layers.Add()([x, input_tensor]))  # 实现relu(f(x)+x)操作

        return out


class BasicResBlockUpdate(keras.Model):
    """ResNet升级版，输入输出大小不一致。
    y=f(x)+Wx W为映射矩阵，保证维度一致。
    输入参数：
    filter_num, kernel_size, strides, input_channels
    参数默认值：filter_num=[64, 64]
    kernel_size=[3, 3], strides=[1, 1]
    """

    def __init__(self, filter_num=[64, 64], kernel_size=[3, 3], strides=[1, 1], input_channels=64):
        super(BasicResBlockUpdate, self).__init__()

        # 初始化滤波器个数、滤波器大小、和步长
        filter_num1, filter_num2 = filter_num
        kernel_size1, kernel_size2 = kernel_size
        strides1, strides2 = strides

        # 定义卷积层
        self.cnn1 = layers.Conv2D(filters=filter_num1, kernel_size=kernel_size1,
                                  strides=strides1, padding='same', activation='relu')
        self.bn1 = layers.BatchNormalization()
        self.cnn2 = layers.Conv2D(filters=filter_num2, kernel_size=kernel_size2,
                                  strides=strides2, padding='same', activation=None)
        self.bn2 = layers.BatchNormalization()

        if strides1 * strides2 == 1 and input_channels == filter_num2:
            self.shortcut = layers.Lambda(lambda x: x)
        else:
            # 维度不一样
            # 两种解决方法

            # 方法1：先将输入补零增加维度，后利用池化降低维度
            # self.shortcut = keras.Sequential([
            #     layers.Lambda(lambda x: tf.pad(x, [[0,  0], [0, 0], [0, 0], [0, filter_num2-x.shape[3]]],
            #     mode='CONSTANT', constant_values=0, name=None)),
            #     layers.MaxPool2D(pool_size=2, strides=strides1 * strides2, padding='same')
            # ])

            # 方法2：利用1*1卷积增加维度，再池化
            self.shortcut = keras.Sequential([
                layers.Conv2D(filter_num2, kernel_size=1,
                              strides=1, padding='same'),
                layers.MaxPool2D(pool_size=2, strides=strides1 *
                                 strides2, padding='same'),
                layers.BatchNormalization()
            ])

    def call(self, inputs, training=False):
        x = self.cnn1(inputs)
        x = self.bn1(x)
        x = self.cnn2(x)
        x = self.bn2(x)
        input_new = self.shortcut(inputs)
        out = tf.nn.relu(input_new + x)

        return out


class BottleneckResBlock(keras.Model):
    """bottleneck building block
    输入参数：
    filter_num, kernel_size, strides, input_channels
    默认参数：filter_num=[64, 64, 256]
    kernel_size=[1, 3, 1], strides=[1, 1, 1]
    input_channels=64
    """

    def __init__(self, filter_num=[64, 64, 256], kernel_size=[1, 3, 1], strides=[1, 1, 1], input_channels=64):
        super(BottleneckResBlock, self).__init__()

        # 初始化滤波器个数、滤波器大小、和步长
        filter_num1, filter_num2, filter_num3 = filter_num
        kernel_size1, kernel_size2, kernel_size3 = kernel_size
        strides1, strides2, strides3 = strides

        # 定义卷积层
        self.cnn1 = layers.Conv2D(filters=filter_num1, kernel_size=kernel_size1,
                                  strides=strides1, padding='same', activation='relu')
        self.bn1 = layers.BatchNormalization()
        self.cnn2 = layers.Conv2D(filters=filter_num2, kernel_size=kernel_size2,
                                  strides=strides2, padding='same', activation='relu')
        self.bn2 = layers.BatchNormalization()
        self.cnn3 = layers.Conv2D(filters=filter_num3, kernel_size=kernel_size3,
                                  strides=strides3, padding='same', activation=None)
        self.bn3 = layers.BatchNormalization()

        if strides1 * strides2 == 1 and input_channels == filter_num3:
            self.shortcut = layers.Lambda(lambda x: x)
        else:
            # 维度不一样
            # 利用1*1卷积增加维度，再池化
            self.shortcut = keras.Sequential([
                layers.Conv2D(filter_num3, kernel_size=1,
                              strides=1, padding='same'),
                layers.MaxPool2D(pool_size=2, strides=strides1 *
                                 strides2 * strides3, padding='same'),
                layers.BatchNormalization()
            ])

    def call(self, inputs, training=False):
        x = self.cnn1(inputs)
        x = self.bn1(x)
        x = self.cnn2(x)
        x = self.bn2(x)
        x = self.cnn3(x)
        x = self.bn3(x)
        input_new = self.shortcut(inputs)
        out = tf.nn.relu(input_new + x)

        return out


class ResNet(keras.Model):
    """
    定义ResNet类，层数和结构支持自定义
    输入参数：（括号类为默认值）
        block_type='basic', block_num=[3, 4, 6, 3], filter_num=[64, 128, 256, 512], 
        kernel_size=[3, 3， 3， 3],  strides=[[1, 1], [2, 1], [2, 1], [2, 1]], input_channels=64,
        class_num=3
    参数说明：
        block_type:ResBlock类型，可选项：basic, bottleneck
        block_num:ResBlock大块的数目，一个ResNet可有4个ResBlock大块，每个ResBlock大块包含了许多小ResBlock
                  此参数指定每个block大块中的小块个数。
        filte_num:数组个数与block_num一致，每个值对应一个ResBlock大块的单个小块的滤波器数量
        kernel_size:与filter_num对应，设置每个大块中的卷积核大小。
        strides:维数为len(block_num)*2，第一列对应大块中第一层的步长，第二列为其他层的步长。
        input_channels:输入图像的通道数
        class_num:待分类的数目
        备注:默认每个大块里面的网络层除第一层步长可以与其他层不一致外，其他参数完全一致。
    """

    def __init__(self, block_type='basic', block_num=[3, 4, 6, 3], filter_num=[64, 128, 256, 512],
                 kernel_size=[3, 3, 3, 3],  strides=[[1, 1], [2, 1], [2, 1], [2, 1]], input_channels=64, class_num=3):
        super(ResNet, self).__init__()

        # 构建初始的layer
        self.cnn1 = layers.Conv2D(
            input_channels, kernel_size=7, strides=2, padding='same')
        self.pool1 = layers.MaxPooling2D(
            pool_size=2, strides=2, padding='same')
        self.bn1 = layers.BatchNormalization()

        # 构建ResBlock大块
        self.big_block1 = self._make_block(
            block_type, block_num[0], filter_num[0], kernel_size[0], strides[0][:], input_channels)
        self.big_block2 = self._make_block(
            block_type, block_num[1], filter_num[1], kernel_size[1], strides[1][:], block_num[0])
        self.big_block3 = self._make_block(
            block_type, block_num[2], filter_num[2], kernel_size[2], strides[2][:], block_num[1])
        self.big_block4 = self._make_block(
            block_type, block_num[3], filter_num[3], kernel_size[3], strides[3][:], block_num[2])

        # 收尾层
        self.pool2 = layers.AveragePooling2D(pool_size=4)
        self.fl = layers.Flatten()
        self.fc = layers.Dense(class_num, activation='softmax')

    def _make_block(self, btype, block, filters, kernel, stride, pre_block):
        """
        创建ResBlock小块
        """

        layers = []
        # 创建小块。
        if btype == 'basic':
            for i in range(block):
                if i == 0:
                    layers.append(BasicResBlockUpdate(filter_num=[filters, filters], kernel_size=[
                                  kernel, kernel], strides=stride, input_channels=pre_block))
                else:
                    layers.append(BasicResBlockUpdate(filter_num=[filters, filters], kernel_size=[
                                  kernel, kernel], strides=[stride[1], stride[1]], input_channels=pre_block))

        elif btype == 'bottleneck':
            for i in range(block):
                if i == 0:
                    layers.append(BottleneckResBlock(filter_num=[filters, filters], kernel_size=[
                                  kernel, kernel], strides=stride, input_channels=pre_block))
                else:
                    layers.append(BottleneckResBlock(filter_num=[filters, filters], kernel_size=[
                                  kernel, kernel], strides=[stride[1], stride[1]], input_channels=pre_block))
        else:
            print('层类型错误！')
            return

        return keras.Sequential(layers)

    def call(self, inputs, training=False):
        x = self.cnn1(inputs, training=training)
        x = self.pool1(x)
        x = self.bn1(x)
        x = self.big_block1(x, training=training)
        x = self.big_block2(x, training=training)
        x = self.big_block3(x, training=training)
        x = self.big_block4(x, training=training)
        x = self.pool2(x)
        x = self.fl(x)
        out = self.fc(x)

        return out
