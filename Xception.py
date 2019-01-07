from keras.models import Sequential
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Dense, Activation, Flatten, Dropout, BatchNormalization
from keras.layers import Conv2D, MaxPooling2D
from keras import regularizers, optimizers
from keras.applications import Xception
from keras.callbacks import CSVLogger
from keras.models import Model
import numpy as np

csv_logger = CSVLogger('MLP_training.csv')


img_width, img_height = 224, 224
n_batch_size = 4
n_epochs = 5
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

# 以訓練好的 Xception 為基礎來建立模型
net = Xception(input_shape=(img_height, img_width, 3),
                              include_top=False, 
                              weights='imagenet', 
                              pooling='max')

# 增加 Dense layer 與 softmax 產生個類別的機率值
x = net.output
x = Dense(4096,  activation='relu')(x)
output_layer = Dense(3, activation='softmax', name='softmax')(x)

# 設定要進行訓練的網路層
model = Model(inputs=net.input, outputs=output_layer)

# 取ImageNet中的起始Weight，不使他隨機產生，故凍結最底層
FREEZE_LAYERS = 2
for layer in model.layers[:FREEZE_LAYERS]:
    layer.trainable = False
for layer in model.layers[FREEZE_LAYERS:]:
    layer.trainable = True

model.compile(optimizers.RMSprop(), 
              loss='categorical_crossentropy', 
              metrics=['accuracy'])

# 輸出整個網路結構
model.summary()



model.fit_generator(train_generator,
                    steps_per_epoch = n_steps_per_epoch,
                    epochs = n_epochs,
                    callbacks=[csv_logger])


model.save('Xception_model.h5')

prediction = model.evaluate_generator(test_generator, steps=n_steps_per_epoch, verbose=1)
print(prediction)
print(model.metrics_names)

input('')
