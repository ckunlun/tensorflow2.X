# 测试ResNet网络

# import module
import os
import zipfile
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow.keras as keras
import tensorflow.keras.layers as layers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from ResNet import *
# import cv2

# 导入数据集
# 解压数据集
# local_zip = './rps.zip'
# zip_ref = zipfile.ZipFile(local_zip, 'r')
# zip_ref.extractall('./')
# zip_ref.close()

# local_zip = './rps-test-set.zip'
# zip_ref = zipfile.ZipFile(local_zip, 'r')
# zip_ref.extractall('./')
# zip_ref.close()

# 定义文件夹
train_dir = './rps/'

# 使用图像生成器修改数据
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

validation_dir = './rps-test-set/'
validation_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(227, 227),
    batch_size=20,
    class_mode='categorical'
)
validation_generator = validation_datagen.flow_from_directory(
    validation_dir,
    target_size=(227, 227),
    batch_size=20,
    class_mode='categorical'
)

# filters = [128, 256]
# kernel_size = [5, 3]
# stride = [2, 1]
# # base_model = MyModelUpdate()

# model = keras.Sequential([
#     layers.Conv2D(64, 11, 3, activation='relu'),
#     MyModelUpdate(filters, kernel_size, stride),
#     layers.Flatten(),
#     layers.Dense(200, activation='relu'),
#     layers.Dense(3, activation='softmax')
# ])
# model.summary()
# 添加全连接层,存在问题，如何向自定义类中加入层
model = ResNet()
model.build(input_shape=(None, 227, 227, 3))
model.summary()

# 实现迭代过程中达到目标后终止迭代


class MyCallback(keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs={}):
        if(logs.get('accuracy') > 0.99):  # 在文见中查看准确率
            print('the accuracy is good! So stop!')
            self.model.stop_training = True


callback = MyCallback()


model.compile(optimizer='rmsprop',
              loss='categorical_crossentropy', metrics=['accuracy'])
history = model.fit_generator(train_generator, epochs=100, steps_per_epoch=42, validation_data=validation_generator,
                              validation_steps=6, verbose=2, callbacks=[callback])
