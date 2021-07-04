import tensorflow as tf
import pandas as pd
import random
import cv2 as cv
import numpy as np

# Load model
model_path = "models/20210704-163941"
model = tf.keras.models.load_model(model_path)

print("-----------------------------------------------------------------------")

# Read csv files
data = pd.read_csv("blazar_test/labels.csv") 
# reading images
image_names=data['name']
image_labels=data['label']

image_labels_pos=[indx for indx,label in enumerate(image_labels) if label==1]
image_labels_neg_org=[indx for indx,label in enumerate(image_labels) if label==0]

print("Total Population:",len(image_names))
print("Positive class samples:",len(image_labels_pos))
print("Negative class samples:",len(image_labels_neg_org))
print("Total Population:",len(image_labels_pos)+len(image_labels_neg_org))

class_names=["Positive class","Negative class"]
idx = random.randint(0, len(image_labels_pos)-1)
image_path = "blazar_test/patches_candidates/"+image_names[image_labels_pos[idx]]
# print("Image path:",image_path)
img = cv.imread(image_path)
# print("Image shape:",img.shape)

# print(model.input.name,"|",model.input.shape)
# print(model.output.name,"|",model.output.shape)
x=np.array([img])
print("Input:",x.shape)
prediction=model.predict(x)[0]
print("-----------------------------------------------------------------------")
print("Predictions:",prediction)
predicted_class = prediction.argmax(axis=-1)
if predicted_class==0:
    print("Actual Class: POSITIVE \t Prediction Class: POSITIVE")
else:
    print("Actual Class: POSITIVE \t Prediction Class: NEGATIVE")
print("Probability:",prediction[predicted_class])

# Testing for negative class
idx = random.randint(0, len(image_labels_neg_org)-1)
image_path = "blazar_test/patches_candidates/"+image_names[image_labels_neg_org[idx]]
# print("Image path:",image_path)
img = cv.imread(image_path)
# print("Image shape:",img.shape)
x=np.array([img])
# print("Input:",x.shape)
prediction=model.predict(x)[0]
print("-----------------------------------------------------------------------")
print("Predictions:",prediction)
predicted_class = prediction.argmax(axis=-1)
if predicted_class==0:
    print("Actual Class: NEGATIVE \t Prediction Class: POSITIVE")
else:
    print("Actual Class: NEGATIVE \t Prediction Class: NEGATIVE")
print("Probability:",prediction[predicted_class])
