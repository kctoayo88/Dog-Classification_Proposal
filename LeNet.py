import numpy as np
import keras
import csv
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Conv2D, MaxPooling2D, Flatten
from keras.models import Model
from keras.optimizers import Adam
from keras.callbacks import CSVLogger
from keras.preprocessing.image import ImageDataGenerator

img_size = 120
n_epochs = 10
batch_sizes = 12
n_steps_per_epoch = 500
n_validation_steps = 100
csv_logger = CSVLogger('Lenet.csv')
model_file_name = 'Lenet.h5'


train_datagen = ImageDataGenerator(rescale=1./255.)
test_datagen = ImageDataGenerator(rescale=1./255.)

train_generator=train_datagen.flow_from_directory('./picture_train', 
                                                  target_size=(img_size, img_size), 
                                                  batch_size=batch_sizes)

test_generator = test_datagen.flow_from_directory('./picture_test',
                                                  target_size=(img_size, img_size),
                                                  batch_size=batch_sizes)

model = Sequential()

model.add(Conv2D(filters = 16, kernel_size = (5, 5), 
                 padding = 'same', input_shape = (img_size, img_size, 3), 
                 activation = 'relu'))
model.add(MaxPooling2D(pool_size = (2, 2)))
model.add(Conv2D(filters = 36, kernel_size = (5, 5), 
                 padding = 'same', activation = 'relu'))
model.add(MaxPooling2D(pool_size = (2, 2)))
model.add(Flatten())
model.add(Dense(1024, activation = 'relu'))
model.add(Dropout(0.5))
model.add(Dense(3, activation = 'softmax'))

model.compile(optimizer = Adam(lr=1e-4), 
              loss='categorical_crossentropy', 
              metrics=['accuracy'])

model.summary()

model.fit_generator(train_generator,
                    epochs=n_epochs,
                    validation_data=test_generator,
                    steps_per_epoch = n_steps_per_epoch,
                    validation_steps = n_validation_steps,
                    callbacks=[csv_logger],
                    verbose=1)

model.save(model_file_name)

prediction = model.evaluate_generator(test_generator, steps=n_steps_per_epoch, verbose=1)
print(prediction)
print(model.metrics_names)

input('')
