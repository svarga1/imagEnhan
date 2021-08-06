#Code for autoencoder that enhances images from https://www.analyticsvidhya.com/blog/2020/02/what-is-autoencoder-enhance-image-resolution/

#Modules
import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

from tensorflow.keras import Model, Input, regularizers
from tensorflow.keras.layers import Dense, Conv2D, MaxPool2D, UpSampling2D
from tensorflow.keras.callbacks import EarlyStopping
from keras.preprocessing import image

import glob
from tqdm import tqdm
import warnings;
import matplotlib

matplotlib.use('agg')
warnings.filterwarnings('ignore')

#Dataset

face_images = glob.glob('/work/noaa/da/svarga/imagEnhan/lfw/**/*.jpg')

#Load and preprocess images
#all_images=[]
#for i in tqdm(face_images):
#  img=image.load_img(i, target_size=(80,80,3))
#  img = image.img_to_array(img)
#  img = img/255.
#  all_images.append(img)
  
#all_images = np.array(all_images)

# split data into train and validation data
#train_x, val_x = train_test_split(all_images, random_state=32, test_size=0.1)


# function to reduce image resolution while keeping the image size constant

#def pixalate_image(image, scale_percent = 40):
#  width = int(image.shape[1] * scale_percent / 100)
#  height = int(image.shape[0] * scale_percent / 100)
#  dim = (width, height)

#  small_image = cv2.resize(image, dim, interpolation = cv2.INTER_AREA)
  
  # scale back to original size
#  width = int(small_image.shape[1] * 100 / scale_percent)
#  height = int(small_image.shape[0] * 100 / scale_percent)
#  dim = (width, height)

#  low_res_image = cv2.resize(small_image, dim, interpolation = cv2.INTER_AREA)

#  return low_res_image

#train_x_px = []

##for i in range(train_x.shape[0]):
 # temp = pixalate_image(train_x[i,:,:,:])
 # train_x_px.append(temp)

#train_x_px = np.array(train_x_px)


# get low resolution images for the validation set
#val_x_px = []

#for i in range(val_x.shape[0]):
#  temp = pixalate_image(val_x[i,:,:,:])
# val_x_px.append(temp)

#val_x_px = np.array(val_x_px)

Input_img = Input(shape=(80, 80, 3))  
    
#encoding architecture
x1 = Conv2D(256, (3, 3), activation='relu', padding='same')(Input_img)
x2 = Conv2D(128, (3, 3), activation='relu', padding='same')(x1)
x2 = MaxPool2D( (2, 2))(x2)
encoded = Conv2D(64, (3, 3), activation='relu', padding='same')(x2)

# decoding architecture
x3 = Conv2D(64, (3, 3), activation='relu', padding='same')(encoded)
x3 = UpSampling2D((2, 2))(x3)
x2 = Conv2D(128, (3, 3), activation='relu', padding='same')(x3)
x1 = Conv2D(256, (3, 3), activation='relu', padding='same')(x2)
decoded = Conv2D(3, (3, 3), padding='same')(x1)

autoencoder = Model(Input_img, decoded)
autoencoder.compile(optimizer='adam', loss='mae')

#Train model
#early_stopper = EarlyStopping(monitor='val_loss', min_delta=0.0001, patience=4, verbose=1, mode='auto')

#a_e = autoencoder.fit(train_x_px, train_x,
#            epochs=15, #25,
#            batch_size=256,
#            shuffle=True,
#            validation_data=(val_x_px, val_x),
#            callbacks=[early_stopper])
#Enhance images
#predictions = autoencoder.predict(val_x_px)

#n = 5
#plt.figure(figsize= (20,10))

#for i in range(n):
#  ax = plt.subplot(2, n, i+1)
#  plt.imshow(val_x_px[i+20])
#  ax.get_xaxis().set_visible(False)
#  ax.get_yaxis().set_visible(False)
#
#  ax = plt.subplot(2, n, i+1+n)
#  plt.imshow(predictions[i+20])
#  ax.get_xaxis().set_visible(False)
#  ax.get_yaxis().set_visible(False)
#
#plt.savefig('output50ep')
