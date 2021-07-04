#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd 
import numpy as np
from tqdm import tqdm
import os, sys
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import cv2 as cv
from datetime import datetime

# creating the Deep neural network
import tensorflow as tf
import tensorflow_hub as hub
from tensorflow import keras
print("TF version:", tf.__version__)
print("Hub version:", hub.__version__)
print("GPU is", "available" if tf.config.list_physical_devices('GPU') else "NOT AVAILABLE")


# In[2]:


data = pd.read_csv("blazar_test/labels.csv") 


# In[3]:


data.head(10)


# In[4]:


# reading images
image_names=data['name']
image_labels=data['label']

image_labels_pos=[indx for indx,label in enumerate(image_labels) if label==1]
image_labels_neg_org=[indx for indx,label in enumerate(image_labels) if label==0]
print("Number of Positive Smaples:",len(image_labels_pos))
print("Number of Negative Smaples:",len(image_labels_neg_org))
# Appling the random sample selection
accept, reject = train_test_split(image_labels_neg_org, test_size=0.56, random_state=42)
image_labels_neg=accept[0:len(image_labels_pos)]
print("After Random Sampling, Number of Negative Smaples:",len(image_labels_neg))

# create a dataset
classes=["Positive Class","Negative Class"]
height = [len(image_labels_pos),len(image_labels_neg_org)]
bars = ["Positive Class","Negative Class"]
bars_position=[i for i in range(len(bars))]
x_pos = np.arange(len(bars))

# Create bars with different colors
plt.bar(x_pos, height, color=['olive', 'cyan'])

# Text on the top of each bar
for i in range(len(bars)):
    plt.text(x = bars_position[i] , y = height[i], s = height[i], size = 10)

# Create names on the x-axis
# plt.xticks(x_pos, bars)
# Rotation of the bar names
plt.xticks(x_pos, bars, rotation=90)

# Custom the subplot layout
plt.subplots_adjust(bottom=0.3, top=0.99)


# Show graph
plt.show()


# In[5]:


pos_images=[];neg_images=[]
for i in range(len(image_labels_neg)):
    pos_image=image_names[image_labels_pos[i]]
    neg_image=image_names[image_labels_neg[i]]
    # adding the images into the respective array
    pos_images.append(cv.imread("blazar_test/patches_candidates/"+pos_image))
    neg_images.append(cv.imread("blazar_test/patches_candidates/"+neg_image))
#     print("Positive image:",pos_image,"| Negative image:",neg_image)
    if i>5:
        break

print("The First two columns are corresponding to Positive Class \n While the last two columns are corresponding to negative Class")
fig, axarr = plt.subplots(3,4)
axarr[0,0].imshow(pos_images[0]);axarr[0,1].imshow(pos_images[1]);axarr[0,2].imshow(neg_images[0]);axarr[0,3].imshow(neg_images[1])
axarr[1,0].imshow(pos_images[2]);axarr[1,1].imshow(pos_images[3]);axarr[1,2].imshow(neg_images[2]);axarr[1,3].imshow(neg_images[3])
axarr[2,0].imshow(pos_images[4]);axarr[2,1].imshow(pos_images[5]);axarr[2,2].imshow(neg_images[4]);axarr[2,3].imshow(neg_images[5])
plt.show()


# In[6]:


X=[];Y=np.array([])
for i in tqdm(range(len(image_labels_pos))):
    pos_image=image_names[image_labels_pos[i]]
    img=cv.imread("blazar_test/patches_candidates/"+pos_image)
    if img.shape!=(100,100,3):
        continue
    X.append(img)
#     if i>10:
#         break
number_of_samples=len(X)
# number_of_samples=12
Y = np.zeros((number_of_samples, 1))
print(Y.shape) 
neg_class_count=0
for i in tqdm(range(len(image_labels_neg))):
    pos_image=image_names[image_labels_neg[i]]
    img=cv.imread("blazar_test/patches_candidates/"+pos_image)
    if img.shape!=(100,100,3):
        continue
    X.append(img)
    neg_class_count=neg_class_count+1
#     if i>10:
#         break
# number_of_samples=len(image_labels_neg)
number_of_samples=neg_class_count
Y = np.concatenate((Y, np.ones((number_of_samples, 1))), axis = 0 ) 
X=np.array(X)
print(X.shape,Y.shape)
X_train,X_test,Y_train,Y_test=train_test_split(X,Y, test_size=0.30, random_state=42)
print("Training Population:",X_train.shape,Y_train.shape)
print("Testing Population:",X_test.shape,Y_test.shape)


# In[9]:


print("Image shape:",pos_images[0].shape)
number_of_classes=2
IMG_SHAPE=pos_images[0].shape
#base_model = tf.keras.applications.ResNet50(input_shape=IMG_SHAPE, include_top=False)
#base_model = tf.keras.applications.efficientnet.EfficientNetB7(input_shape=IMG_SHAPE, include_top=False)
base_model = tf.keras.applications.inception_resnet_v2.InceptionResNetV2(input_shape=IMG_SHAPE, include_top=False)
# base_model = tf.keras.applications.inception_v3.InceptionV3(input_shape=IMG_SHAPE, include_top=False)
# base_model = tf.keras.applications.vgg16.VGG16(input_shape=IMG_SHAPE, include_top=False)
# base_model = tf.keras.applications.xception.Xception(input_shape=IMG_SHAPE, include_top=False)


base_model.trainable = False

model = tf.keras.Sequential([
  base_model,
  tf.keras.layers.GlobalAveragePooling2D(),
  tf.keras.layers.Dense(number_of_classes, activation='softmax')
])

# # Define the model
# model = models.Sequential()
# feature = base_model(input_tensor)
# # the low level features
# model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32,32,3)))
# model.add(layers.MaxPooling2D((2, 2)))

model.summary()

model.compile(optimizer=tf.keras.optimizers.Adam(),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])


# In[10]:


# Define the Keras TensorBoard callback.
timestamp=datetime.now().strftime("%Y%m%d-%H%M%S")
logdir="logs/" + timestamp
tensorboard_callback = keras.callbacks.TensorBoard(log_dir=logdir)


# In[11]:


# fit the keras model on the dataset
model.fit(X_train, Y_train,validation_data=(X_test, Y_test), epochs=10, batch_size=256,callbacks=[tensorboard_callback])


# In[12]:


# Visualize tensorboard
# tensorboard --logdir logs


# In[13]:


# save model
saved_model_dir = "models/" + timestamp
tf.saved_model.save(model, saved_model_dir)

