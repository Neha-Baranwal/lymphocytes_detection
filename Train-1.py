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
import copy

# creating the Deep neural network
import tensorflow as tf
import tensorflow_hub as hub
from tensorflow import keras
print("TF version:", tf.__version__)
print("Hub version:", hub.__version__)
print("GPU is", "available" if tf.config.list_physical_devices('GPU') else "NOT AVAILABLE")

# detect contours
import random as rng
from statistics import mean
rng.seed(12345)


# In[2]:


def get_countour_count(img):
    # Grayscale
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    # cv.imshow('Gray Image', gray)
   
    # Applying Otsu's method setting the flag value into cv.THRESH_OTSU.
    # Use a bimodal image as an input.
    # Optimal threshold value is determined automatically.
    otsu_threshold, otsu_image = cv.threshold(
        gray, 230, 255, cv.THRESH_BINARY + cv.THRESH_OTSU,
    )
    # print("Obtained threshold: ", otsu_threshold)
    # cv.imshow('Otsu Image', otsu_image)

    # Find Canny edges
    edged = cv.Canny(otsu_image, 10, 50,True)
    # cv.imshow('Canny Edges After Contouring', edged)
    # cv.waitKey(0)

    # Finding Contours
    # Use a copy of the image e.g. edged.copy()
    # since findContours alters the image
    contours, hierarchy = cv.findContours(edged, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
    # contours, hierarchy = cv.findContours(edged, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    
    # print("Number of Contours found = " + str(len(contours)))

    # Find the convex hull object for each contour
    hull_list = []
    for i in range(len(contours)):
        hull = cv.convexHull(contours[i])
        hull_list.append(hull)
    # Draw contours + hull results
    drawing = np.zeros((edged.shape[0], edged.shape[1], 3), dtype=np.uint8)

    count=0
    for i in range(len(contours)):
        # print("Area:",cv.contourArea(contours[i]))
        if cv.contourArea(contours[i])>2 and cv.contourArea(contours[i])<5:
            color = (rng.randint(0,256), rng.randint(0,256), rng.randint(0,256))
            # cv.drawContours(drawing, contours, i, color)
            cv.drawContours(drawing, hull_list, i, color)
            count=count+1

    # for cnt in contours:
    #     if cv.isContourConvex(cnt) == True:
    #         # closed_contours.append(cnt)
    #         cv.drawContours(img, cnt, -1, (0, 255, 0), 3)
    #     else:
    #         pass
    
    # Draw all contours
    # -1 signifies drawing all contours
    # cv.drawContours(img, contours, -1, (0, 255, 0), 3)
    
    # cv.imshow('Contours', img)
    # cv.imshow('Drawing', drawing)
    # cv.waitKey(0)
    # cv.destroyAllWindows()
    return count


# In[3]:


data = pd.read_csv("blazar_test/labels.csv") 


# In[4]:


data.head(10)


# In[5]:


image_names=data['name']
image_labels=data['label']

image_labels_pos=[indx for indx,label in enumerate(image_labels) if label==1]
image_labels_neg_org=[indx for indx,label in enumerate(image_labels) if label==0]
print("Number of Positive Smaples:",len(image_labels_pos))
print("Number of Negative Smaples:",len(image_labels_neg_org))
# Appling the random sample selection
# accept, reject = train_test_split(image_labels_neg_org, test_size=0.56, random_state=42)
# image_labels_neg=accept[0:len(image_labels_pos)]
# print("After Random Sampling, Number of Negative Smaples:",len(image_labels_neg))


# In[6]:


image_labels_neg=[];negative_contour_count=[];positive_contour_count=[]
pos_images=[];neg_images=[]
for i in tqdm(range(len(image_labels_neg_org))):
    if i<len(image_labels_pos):
        image_name = "blazar_test/patches_candidates/"+image_names[image_labels_pos[i]]
        img = cv.imread(image_name)
        if img.shape!=(100,100,3):
            continue
        pos_count=get_countour_count(img)
        positive_contour_count.append(pos_count)
        pos_images.append(img)
    image_name = "blazar_test/patches_candidates/"+image_names[image_labels_neg_org[i]]
    img = cv.imread(image_name)
    neg_count=get_countour_count(img)
    if neg_count>12:
#     print("Number of positive and negative contours:",pos_count,neg_count,image_name)
        if img.shape!=(100,100,3):
            continue
        image_labels_neg.append(image_labels_neg_org[i])
        negative_contour_count.append(neg_count)
        neg_images.append(img)

print("Number of positive images:",len(pos_images))
print("Number of negative images:",len(neg_images))


# In[ ]:


# print(pos_images_count,neg_images_count)
print("Number of Negative Smaples:",len(image_labels_neg))
print("Number of negaitve count:",len(negative_contour_count))
print("Average of Positive and negative class:",mean(positive_contour_count),mean(negative_contour_count))
# sample_count=len(positive_contour_count)+len(negative_contour_count)
# class_colours=np.array(['olive' if i<len(positive_contour_count) else 'cyan' for i in range(sample_count)])
# print("Total sample count:",sample_count)
# # create a dataset
# classes=["Positive Class","Negative Class"]
# height = positive_contour_count
# height.extend(negative_contour_count)
# height=np.array(height)
# bars = [str(i) for i in range(sample_count)]
# print("Length of bars:",len(bars))
# # bars_position=[i for i in range(len(bars))]
# bars_position=[i for i in range(sample_count)]
# x_pos = np.arange(len(bars))

# print(x_pos.shape,height.shape,class_colours.shape)
# # print(class_colours)
# # Create bars with different colors
# plt.bar(x_pos, height, color=['olive', 'cyan'])
# plt.bar(x_pos, height, color=class_colours)

# # Show graph
# plt.show()


# In[ ]:


print("The First two columns are corresponding to Positive Class \n While the last two columns are corresponding to negative Class")
fig, axarr = plt.subplots(3,4)
axarr[0,0].imshow(pos_images[0]);axarr[0,1].imshow(pos_images[1]);axarr[0,2].imshow(neg_images[0]);axarr[0,3].imshow(neg_images[1])
axarr[1,0].imshow(pos_images[2]);axarr[1,1].imshow(pos_images[3]);axarr[1,2].imshow(neg_images[2]);axarr[1,3].imshow(neg_images[3])
axarr[2,0].imshow(pos_images[4]);axarr[2,1].imshow(pos_images[5]);axarr[2,2].imshow(neg_images[4]);axarr[2,3].imshow(neg_images[5])
plt.show()


# In[ ]:


print("Number of positive images:",len(pos_images))
print("Number of negative images:",len(neg_images))
X=copy.deepcopy(pos_images)
# print(len(X))
X.extend(neg_images)
# print(len(pos_images),len(neg_images))
# print(len(X))
X=np.array(X)
print("Population:",X.shape)
Y = np.zeros((len(pos_images), 1))
Y = np.concatenate((Y, np.ones((len(neg_images), 1))), axis = 0 ) 
print("Target:",Y.shape)
print(X.shape,Y.shape)
X_train,X_test,Y_train,Y_test=train_test_split(X,Y, test_size=0.30, random_state=42)
print("Training Population:",X_train.shape,Y_train.shape)
print("Testing Population:",X_test.shape,Y_test.shape)


# In[ ]:


print("Image shape:",pos_images[0].shape)
number_of_classes=2
IMG_SHAPE=pos_images[0].shape
# base_model = tf.keras.applications.ResNet50(input_shape=IMG_SHAPE, include_top=False)
base_model = tf.keras.applications.efficientnet.EfficientNetB7(input_shape=IMG_SHAPE, include_top=False)
# base_model = tf.keras.applications.inception_resnet_v2.InceptionResNetV2(input_shape=IMG_SHAPE, include_top=False)
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


# In[ ]:


# Define the Keras TensorBoard callback.
timestamp=datetime.now().strftime("%Y%m%d-%H%M%S")
logdir="logs/" + timestamp
tensorboard_callback = keras.callbacks.TensorBoard(log_dir=logdir)


# In[ ]:


# fit the keras model on the dataset
model.fit(X_train, Y_train,validation_data=(X_test, Y_test), epochs=10, batch_size=256,callbacks=[tensorboard_callback])


# In[ ]:


# Visualize tensorboard
# tensorboard --logdir logs


# In[ ]:


# save model
saved_model_dir = "models/" + timestamp
tf.saved_model.save(model, saved_model_dir)


# In[ ]:




