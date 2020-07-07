import tensorflow as tf 
import numpy as np
#import pickle
import os
import cv2 
import pandas as pd

# preparing dataset
print("Prepare Testing data ...")
FILEPATH = "shopee-product-detection-dataset/test/test"
test_image = []
IMG_SIZE =100

def create_test_data(): 
    for img in os.listdir(FILEPATH):
        try:
        	img_array = cv2.imread(os.path.join(FILEPATH,img),cv2.IMREAD_GRAYSCALE)
        	img_array = cv2.resize(img_array, (IMG_SIZE,IMG_SIZE))
        	test_image.append([img_array, img, 0])
        except Exception as e:
        	pass

create_test_data()


X = []
filenames = []

for image, filename, label in test_image:
    X.append(image)
    filenames.append(filename)

X = np.array(X).reshape(-1, IMG_SIZE,IMG_SIZE, 1)
X = X.astype('float32')
X /= 255.0

print("Test Data Created!")
print("Ready to test the data!")

# testing the data
print("Testing the data ...")
new_model = tf.keras.models.load_model('train.model')
pred = new_model.predict([X])

category =[]

for i in range (len(X)):
	category.append(int(np.argmax(pred[i])))

print("Finished Testing the data!")
print("Writing to CSV!")

data = {'filename':filenames, 'category':category}
df = pd.DataFrame(data=data)

df.to_csv("answer.csv", header = True, index=False)
print("Finished!")





