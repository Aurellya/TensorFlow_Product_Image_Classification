import os
import cv2 
import random
import pickle
import numpy as np
import pandas as pd

# preparing data
print("Preparing Train Data ...")

FILEPATH = "shopee-product-detection-dataset/train.csv"
df = pd.read_csv(FILEPATH)

CATEGORIES = df['category'].unique().tolist()
CATEGORIES.sort()

for c in range (len(CATEGORIES)):
    if c < 10:
        CATEGORIES[c] = '0' + str(CATEGORIES[c])
    else:
        CATEGORIES[c] = str(CATEGORIES[c])


DATADIR = "shopee-product-detection-dataset/train/train"
training_data = []
IMG_SIZE = 100 

def create_training_data():
    for category in CATEGORIES:
        path = os.path.join(DATADIR, category)
        class_num = CATEGORIES.index(category)
        for img in os.listdir(path):
            try:
                img_array = cv2.imread(os.path.join(path,img),cv2.IMREAD_GRAYSCALE)
                img_array = cv2.resize(img_array, (IMG_SIZE,IMG_SIZE))
                training_data.append([img_array, class_num])
            except Exception as e:
                pass

create_training_data()
print("Finished Preparing Train Data!")

# Shuffling training data
print("Shuffling Data ...")
random.shuffle(training_data)
print("Shuffled ...")

# Save Data to Pickle
print("Saving Data ...")

X = []
y = []

for features, label in training_data:
    X.append(features)
    y.append(label)

X = np.array(X).reshape(-1, IMG_SIZE,IMG_SIZE, 1)

pickle_out = open("X.pickle", "wb")
pickle.dump(X, pickle_out)
pickle_out.close()

pickle_out = open("y.pickle", "wb")
pickle.dump(y, pickle_out)
pickle_out.close()

print("Data Saved!")





