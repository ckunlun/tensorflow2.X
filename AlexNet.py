# reference：
# Krizhevsky, Alex, Ilya Sutskever, and Geoffrey E. Hinton. "Imagenet classification
# with deep convolutional neural networks."
# Advances in neural information processing systems. 2012.
# Author:ckunlun
# Time:2020.6.14

# import module
import tensorflow as tf
import tensorflow.keras as keras
import tensorflow.keras.layers as layers

# 创建自定义网络层类时需要继承layers.Layer基类
# 创建自定义的网络类时，需要继承自keras.Model基类


# class MyFCLayers(layers.Layer):
#     """自定义网络层"""

#     def __init__(self, input_dim, output_dim):
#         super(MyFCLayers, self).__init__()   # 继承父类的构造方法

#         # 创建权值张量并添加到类管理列表中，设置为需要优化
#         self.kernel = self.add_variable('w', [input_dim, output_dim], trainable=True)
#         self.bias = self.add_variable('b', [output_dim], trainable=True)

#     def call(self, inputs, training=None):
#         # 实现自定义类的前向计算逻辑
#         # X@W + b
#         out = inputs @ self.kernel + self.bias
#         # 执行激活函数运输
#         out = tf.nn.relu(out)

#         return out


class MyModel(keras.Model):
    """定义ImageNet类"""

    def __init__(self):
        super(MyModel, self).__init__()

        # 完成网络内需要的网络层的创建工作
        # 第一层
        self.cnn1 = layers.Conv2D(
            96, kernel_size=11, strides=4, activation='relu', padding='valid')
        self.maxpool1 = layers.MaxPooling2D(pool_size=3, strides=2)
        self.batchNorm1 = layers.BatchNormalization()
        # 第二层
        self.cnn2 = layers.Conv2D(
            256, kernel_size=5, strides=1, activation='relu', padding='same')
        self.maxpool2 = layers.MaxPooling2D(
            pool_size=3, strides=2, padding='same')
        self.batchNorm2 = layers.BatchNormalization()
        # 第三层
        self.cnn3 = layers.Conv2D(
            384, kernel_size=3, strides=1, activation='relu', padding='same')
        # 第四层
        self.cnn4 = layers.Conv2D(
            384, kernel_size=3, strides=1, activation='relu', padding='same')
        # 第五层
        self.cnn5 = layers.Conv2D(
            256, kernel_size=3, strides=1, activation='relu', padding='same')
        self.maxpool3 = layers.MaxPooling2D(
            pool_size=3, strides=2, padding='same')
        self.batchNorm3 = layers.BatchNormalization()
        # 第六层
        self.flatten = layers.Flatten()
        self.fc1 = layers.Dense(4096, activation='relu')
        self.dropout1 = layers.Dropout(0.5)
        # 第七层
        self.fc2 = layers.Dense(4096, activation='relu')
        self.dropout2 = layers.Dropout(0.5)
        # 第八层
        self.fc3 = layers.Dense(3, activation='softmax')
        # 测试的数据集只有三个分类，此处改完3，原版ImgNet数据集由1000个分类，此处为1000

    def call(self, inputs, training=False):
        # 自定义前向运输逻辑
        x = self.cnn1(inputs)
        x = self.maxpool1(x)
        x = self.batchNorm1(x)
        x = self.cnn2(x)
        x = self.maxpool2(x)
        x = self.batchNorm2(x)
        x = self.cnn3(x)
        x = self.cnn4(x)
        x = self.cnn5(x)
        x = self.maxpool3(x)
        x = self.batchNorm3(x)
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.dropout1(x)
        x = self.fc2(x)
        x = self.dropout2(x)
        x = self.fc3(x)
        return x
