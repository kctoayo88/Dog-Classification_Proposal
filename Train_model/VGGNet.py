from keras.models import Sequential
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Dense, Activation, Flatten, Dropout, BatchNormalization
from keras.layers import Conv2D, MaxPooling2D
from keras import regularizers, optimizers
import numpy as np

img_width, img_height = 120, 120
n_batch_size = 32
nb_train_samples = 2000
n_epochs = 50
nb_validation_samples = 800

train_datagen = ImageDataGenerator(rescale=1./255.)
test_datagen = ImageDataGenerator(rescale=1./255.)


## Build VGGNet
model = Sequential()
model.add(Conv2D(32, (3, 3), padding='same',input_shape=(120,120,3)))
model.add(Activation('relu'))
model.add(Conv2D(32, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Conv2D(64, (3, 3), padding='same'))
model.add(Activation('relu'))
model.add(Conv2D(64, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(512))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(3, activation='softmax'))

model.compile(optimizers.rmsprop(), 
              loss="categorical_crossentropy", 
              metrics=["accuracy"])

train_generator=train_datagen.flow_from_directory('./picture_train', 
                                            target_size=(img_height, img_width), 
                                            batch_size=n_batch_size)

test_generator = test_datagen.flow_from_directory('./picture_test',
                                                  target_size=(img_height, img_width),
                                                  batch_size=n_batch_size)

model.fit_generator(train_generator,
                    samples_per_epoch=nb_train_samples,
                    epochs=n_epochs,
                    validation_data=test_generator,
                    nb_val_samples=nb_validation_samples)


model.save('3_model.h5')

x=model.evaluate_generator(test_generator, val_samples=200)
print(x)

input('')