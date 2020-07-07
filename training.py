import pickle
import numpy as np

from sklearn import metrics
from sklearn.model_selection import train_test_split

import tensorflow as tf 
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D

from keras.utils import to_categorical
from keras.layers.normalization import BatchNormalization

# Load pickle
print("Loading pickle ...")

X = pickle.load(open("X.pickle", "rb"))
y = pickle.load(open("y.pickle", "rb"))

X = X.astype('float32')
X /= 255

y = to_categorical(y)
y = np.array(y)

# Split data for train and test
print("Splitting data ...")
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, test_size=0.1)

# create model
print("Creating model ...")

model = Sequential()
model.add(Conv2D(32, kernel_size = (3, 3), activation='relu', input_shape=(100, 100, 1)))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(BatchNormalization())

model.add(Conv2D(64, kernel_size=(3,3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(BatchNormalization())
model.add(Conv2D(64, kernel_size=(3,3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(BatchNormalization())

model.add(Conv2D(96, kernel_size=(3,3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(BatchNormalization())

model.add(Conv2D(32, kernel_size=(3,3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(BatchNormalization())

model.add(Dropout(0.2))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(42, activation = 'softmax'))

model.summary()

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

print("Model Created!")

# Fit the data to the model
print("Start training the data!")
model.fit(X_train, y_train, batch_size=128, epochs=120, verbose=1)
print("Training Finished!")

# Evaluate model
print("Testing the model ...")
loss,acc=model.evaluate(X_test, y_test, verbose=0)
print(acc*100)

# Save model
print("Saving the model ...")
model.save('train.model')
print("Model Saved!")









