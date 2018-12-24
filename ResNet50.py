from keras.models import Sequential
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Dense, Activation, Flatten, Dropout, BatchNormalization
from keras.layers import Conv2D, MaxPooling2D
from keras import regularizers, optimizers
from keras.applications import ResNet50
from keras.models import Model
import numpy as np


img_width, img_height = 224, 224
n_batch_size = 16
n_epochs = 5

train_datagen = ImageDataGenerator(rescale=1./255.)
test_datagen = ImageDataGenerator(rescale=1./255.)

train_generator=train_datagen.flow_from_directory('./picture_train', 
                                            target_size=(img_height, img_width), 
                                            batch_size=n_batch_size,
                                            shuffle=True)

test_generator = test_datagen.flow_from_directory('./picture_test',
                                                  target_size=(img_height, img_width),
                                                  batch_size=n_batch_size)

# 以訓練好的 ResNet50 為基礎來建立模型，
# 捨棄 ResNet50 頂層的 fully connected layers
net = ResNet50(include_top=False, weights='imagenet', input_tensor=None,
               input_shape=(img_height,img_width,3))
x = net.output
x = Flatten()(x)

# 增加 DropOut layer
x = Dropout(0.5)(x)

# 增加 Dense layer，以 softmax 產生個類別的機率值
output_layer = Dense(3, activation='softmax', name='softmax')(x)

# 設定凍結與要進行訓練的網路層
model = Model(inputs=net.input, outputs=output_layer)
FREEZE_LAYERS = 2
for layer in model.layers[:FREEZE_LAYERS]:
    layer.trainable = False
for layer in model.layers[FREEZE_LAYERS:]:
    layer.trainable = True

model.compile(optimizers.Adam(lr=1e-5), 
              loss='categorical_crossentropy', 
              metrics=['accuracy'])

# 輸出整個網路結構
print(model.summary())



model.fit_generator(train_generator,
                    epochs=n_epochs)


model.save('3_model.h5')

prediction = model.evaluate_generator(test_generator, steps=1500, verbose=1)
print(prediction)
print(model.metrics_names)

input('')
