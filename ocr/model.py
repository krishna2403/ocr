# code to create and train the model
# run this file as a script from root folder
# or import this file as module from root folder and invoke the train method

import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import cv2
from PIL import Image
import numpy as np
import os

# processes the image data before training the model
def preprocess_data(path):
  data = []
  labels = []
  if os.path.exists(path) and len(os.listdir(path)) == 92:    # verifying if dataset exists
    folders = os.listdir(path)
    folders.sort()
    for (idx, dir) in enumerate(folders):   # iterating over the folders of each character in the dataset
      for img in os.listdir(path+dir):      # iterating over each image file in character folder
        char = cv2.imread(path+dir+'/'+img, 0)
        # gray scaling the image
        _, char = cv2.threshold(char, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)

        # resizing the image to 28x28
        char = char.astype(np.uint8)
        char[char > 0] = 1
        mask = char
        char = char[np.ix_(mask.any(1),mask.any(0))]
        char = Image.fromarray(char)
        char = char.resize((28,28), Image.ANTIALIAS)
        char = np.array(char).astype(np.uint8)

        # applying padding on all sides of image array by 2 units
        char = np.expand_dims(char, axis=2)
        char = np.pad(char, ((2,2),(2,2),(0,0)), 'constant')

        data.append(char)
        labels.append(idx)
  else:
    print("Dataset not available\nEXITING!")
  data = np.array(data, dtype='float32')
  labels = np.array(labels, dtype='int')
  return data, labels

# creation and training of model
def train():
  data, labels = preprocess_data('ocr/datasets/eng/')

  if(data.shape[0] == 0):
    return

  # dividing the data into train and test data
  train_data, test_data, train_labels, test_labels = train_test_split(data, labels, test_size=0.15, random_state=42)

  # neural network architecture with 92 classes 
  model = tf.keras.Sequential([
    keras.layers.Flatten(input_shape=(32, 32, 1,)),
    keras.layers.Dense(900, activation='sigmoid'),
    keras.layers.Dense(300, activation='sigmoid'),
    keras.layers.Dropout(0.2),
    keras.layers.Dense(92)
  ])

  # compiling the model
  model.compile(optimizer='adam', loss=tf.losses.SparseCategoricalCrossentropy(from_logits=True), metrics=[tf.metrics.SparseCategoricalAccuracy()])

  cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath='ocr/checkpoints/creg', save_weights_only=True, verbose=1)

  # trains the model
  model.fit(
    train_data,
    train_labels,
    epochs=100,
    batch_size=64,
    validation_data=(test_data, test_labels),
    verbose=1,
    callbacks=[cp_callback]
  )

  # validating the performance of the model
  chars = [l for l in '!"#$%&()*+,-./0123456789:;<=>?@ABCDEFGHIJKLMNOPQRSTUVWXYZ[\]^_`abcdefghijklmnopqrstuvwxyz{|}']
  chars = np.array(chars)
  predictions = tf.argmax(model.predict(test_data), axis=1)

  print("Test Report")
  print(classification_report(test_labels, predictions, labels=np.arange(92), target_names=chars, zero_division=0))

  # saves the model
  model.save('ocr/model/model.h5')

if __name__ == '__main__':
  train()
