
# Run this cell if you are using google colab
from google.colab import drive
drive.mount('/content/drive')


# Dataset folder: https://drive.google.com/drive/folders/1fof_IoTHYQxJkYZsPWeXyYvervA20y-A



# Un zip the dataset folder
import zipfile
with zipfile.ZipFile("/content/drive/MyDrive/DL/image_classification/data/Train and Validation.zip", "r") as f:
    f.extractall("/content/drive/MyDrive/DL/image_classification/data/face_expression_data/")

# Libraries importation

import tensorflow
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Dropout, Conv2D, MaxPooling2D, BatchNormalization
from tensorflow.keras.optimizers import Adam, RMSprop
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from keras.utils.vis_utils import plot_model
import os
import glob


# Set Hyperparameters and paths

img_rows, img_cols, = 48,48
num_channel = 1
input_shape = (img_rows, img_cols, num_channel)
batch_size= 8
epochs = 25

train_data_dir = "/content/drive/MyDrive/DL/image_classification/data/face_expression_data/train"
validation_data_dir = "/content/drive/MyDrive/DL/image_classification/data/face_expression_data/validation"



num_classes = 0
for i in glob.glob(train_data_dir+"/*"):
    num_classes += 1
print(num_classes)


# Data augmentation

train_data_generator = ImageDataGenerator(rotation_range=30,
                                          width_shift_range=0.3,
                                          height_shift_range=0.3,
                                          horizontal_flip=True,
                                          shear_range=25,
                                          zoom_range=0.35,
                                          vertical_flip=True,
                                          rescale=1./255)

validation_data_generator = ImageDataGenerator(rescale=1./255)

train_generator = train_data_generator.flow_from_directory(train_data_dir,
                                                           color_mode="grayscale",
                                                           target_size=(img_rows, img_cols),
                                                           batch_size=batch_size,
                                                           class_mode="categorical",
                                                           shuffle=True)

validation_generator = validation_data_generator.flow_from_directory(validation_data_dir,
                                                           color_mode="grayscale",
                                                           target_size=(img_rows, img_cols),
                                                           batch_size=batch_size,
                                                           class_mode="categorical",
                                                           shuffle=True)


# Build the model


model = tensorflow.keras.Sequential()

model.add(Conv2D(32,(3,3),padding='same',kernel_initializer='he_normal', activation='elu',
                 input_shape=(img_rows,img_cols,1)))
model.add(BatchNormalization())
model.add(Conv2D(32,(3,3),padding='same',kernel_initializer='he_normal', activation='elu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.2))

model.add(Conv2D(64,(3,3),padding='same',activation='elu',kernel_initializer='he_normal'))
model.add(BatchNormalization())
model.add(Conv2D(64,(3,3),padding='same',activation='elu',kernel_initializer='he_normal'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.2))

model.add(Conv2D(128,(3,3),padding='same',activation='elu',kernel_initializer='he_normal'))
model.add(BatchNormalization())
model.add(Conv2D(128,(3,3),padding='same',activation='elu',kernel_initializer='he_normal'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.2))

model.add(Conv2D(256,(3,3),padding='same',activation='elu',kernel_initializer='he_normal'))
model.add(BatchNormalization())
model.add(Conv2D(256,(3,3),padding='same',activation='elu',kernel_initializer='he_normal'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.2))

model.add(Flatten())
model.add(Dense(64,kernel_initializer='he_normal',activation='elu',))
model.add(BatchNormalization())
model.add(Dropout(0.5))

model.add(Dense(64,activation='elu',kernel_initializer='he_normal'))
model.add(BatchNormalization())
model.add(Dropout(0.5))

model.add(Dense(num_classes,activation='softmax',kernel_initializer='he_normal'))

print(model.summary())


# Save the model as image


from keras.utils.vis_utils import plot_model
plot_model(model, to_file='/content/drive/MyDrive/DL/image_classification/model_plot.png', show_shapes=True, show_layer_names=True)




checkpoint = ModelCheckpoint(r'/content/drive/MyDrive/DL/image_classification/trained_models/facial_emotion_model.h5',
                             monitor='val_loss',
                             mode='min',
                             save_best_only=True,
                             verbose=1)

earlystop = EarlyStopping(monitor='val_loss',
                          min_delta=0,
                          patience=3,
                          verbose=1,
                          restore_best_weights=True
                          )

reduce_lr = ReduceLROnPlateau(monitor='val_loss',
                              factor=0.2,
                              patience=3,
                              verbose=1,
                              min_delta=0.0001)

callbacks = [earlystop,checkpoint,reduce_lr]



train_sample = 0
for i in glob.glob(train_data_dir+"/*"):
  #print(i)
  for j in glob.glob(i+"/*"):
    train_sample += 1
print(f"train_sample: {train_sample}")



validation_sample = 0
for i in glob.glob(validation_data_dir+"/*"):
  #print(i)
  for j in glob.glob(i+"/*"):
    validation_sample += 1
print(f"validation_sample: {validation_sample}")


# Start Training:


history=model.fit(
                train_generator,
                steps_per_epoch=train_sample//batch_size,
                epochs=epochs,
                callbacks=callbacks,
                validation_data=validation_generator,
                validation_steps=validation_sample//batch_size)



import matplotlib.pyplot as plt

# plotting the accuracy
plt.plot(history.history['accuracy'], label='train acc')
plt.plot(history.history['val_accuracy'], label='val acc')
plt.legend()
plt.savefig('acc_plot.png')
plt.show()


# plotting the loss
plt.plot(history.history['loss'], label='train loss')
plt.plot(history.history['val_loss'], label='val loss')
plt.legend()
plt.savefig('loss_plot.png')
plt.show()





