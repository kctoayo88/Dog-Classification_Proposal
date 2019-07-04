from keras.models import Sequential
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Dense, Activation, Flatten, Dropout, BatchNormalization
from keras.layers import Conv2D, MaxPooling2D
from keras import regularizers, optimizers
from keras.applications import mobilenetv2
from keras.models import Model
from keras.callbacks import TensorBoard
import numpy as np

tbCallBack = TensorBoard(log_dir='C:\TB-logfile',  # log 目录
                 histogram_freq=0,  # 按照何等频率（epoch）来计算直方图，0为不计算
                 batch_size=32,     # 用多大量的数据计算直方图
                 write_graph=True,  # 是否存储网络结构图
                 write_grads=False,  # 是否可视化梯度直方图
                 write_images=True) # 是否可视化参数


img_width, img_height = 224, 224
n_batch_size = 4
n_epochs = 30
n_steps_per_epoch = 1500

train_datagen = ImageDataGenerator(rescale=1./255.)
test_datagen = ImageDataGenerator(rescale=1./255.)

train_generator=train_datagen.flow_from_directory('./picture_train', 
                                            target_size=(img_height, img_width), 
                                            batch_size=n_batch_size,
                                            shuffle=True)

test_generator = test_datagen.flow_from_directory('./picture_test',
                                                  target_size=(img_height, img_width),
                                                  batch_size=n_batch_size)

# 以訓練好的 MobileNetV2 為基礎來建立模型
net = mobilenetv2.MobileNetV2(input_shape=(img_height, img_width, 3), 
                              alpha=1.0, 
                              include_top=False, 
                              weights='imagenet', 
                              pooling='max')

x = net.output

# 增加 Dense layer，以 softmax 產生個類別的機率值
output_layer = Dense(3, activation='softmax', name='softmax')(x)

# 設定要進行訓練的網路層
model = Model(inputs=net.input, outputs=output_layer)

# 取ImageNet中的起始Weight，不使他隨機產生，故凍結最底層
FREEZE_LAYERS = 1
for layer in model.layers[:FREEZE_LAYERS]:
    layer.trainable = False
for layer in model.layers[FREEZE_LAYERS:]:
    layer.trainable = True

model.compile(optimizers.Adam(lr=1e-3), 
              loss='categorical_crossentropy', 
              metrics=['accuracy'])

# 輸出整個網路結構
print(model.summary())



model.fit_generator(train_generator,
                    steps_per_epoch = n_steps_per_epoch,
                    epochs = n_epochs,
                    callbacks=[tbCallBack])


model.save('test_model.h5')

prediction = model.evaluate_generator(test_generator, steps=n_steps_per_epoch, verbose=1)
print(prediction)
print(model.metrics_names)

input('')
