# code to predict the text in an image file

import os
import random
import cv2
import numpy as np
from PIL import Image
import tensorflow as tf
from tf_slim import flatten

# computes the borders of a paragraph in the image with DFS approach
def computeBorder(ocr_img, row, col, rows, cols):
  pixels = [[row, col]]
  border = [col, col, row, row]
  while(len(pixels)):
    [r, c] = pixels.pop()
    ocr_img[r, c] = 0
    if(r-1 >= 0 and ocr_img[r-1, c] != 0):    # checking if the top pixel is part of the paragraph
      pixels.append([r-1, c])
    if(r+1 < rows and ocr_img[r+1, c] != 0):    # checking if the bottom pixel is part of the paragraph
      pixels.append([r+1, c])
    if(c-1 >= 0 and ocr_img[r, c-1] != 0):    # checking if the left pixel is part of the paragraph
      pixels.append([r, c-1])
    if(c+1 < cols and ocr_img[r, c+1] != 0):    # checking if the right pixel is part of the paragraph
      pixels.append([r, c+1])
    border = [min(border[0], c), max(border[1], c), min(border[2], r), max(border[3], r)]
  return border, ocr_img

# 1st step in Optical Character Recognition
# detects and computes borders of paragraphs in the image
def paragraphDetection():
  ocr_img = cv2.imread("image.png", 0)

  # blacking out the background and grey scaling the image
  ocr_img = cv2.bitwise_not(ocr_img)
  _, ocr_img = cv2.threshold(ocr_img, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
  kernel = np.ones((5, 5), np.uint8)
  ocr_img = cv2.dilate(ocr_img, kernel, iterations=10)
  ocr_img = cv2.GaussianBlur(ocr_img, (5, 5), 0)
  _, ocr_img = cv2.threshold(ocr_img, 0, 1, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

  # computes the borders of all paragraphs in the image
  borders = []
  rows, cols = ocr_img.shape
  for row in range(rows):
    for col in range(cols):
      if(ocr_img[row, col] == 1):
        border, ocr_img = computeBorder(ocr_img, row, col, rows, cols)
        borders.append(border)
  return borders

# 2nd step in Optical Character Recognition
# extracts characters from detected paragraphs in the image
def characterDetection(borders):
  ocr_img = cv2.imread("image.png", 0)
  # grey scaling the image
  _, binary = cv2.threshold(ocr_img, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)

  extractedChars = []
  for (index, para) in enumerate(borders):    # iterating over each paragraph
    paraRows = []
    prevSum = 0
    for row in range(para[2], para[3]+1):   # separating all the rows in each paragraph
      currSum = np.sum(binary[row, para[0]:para[1]+1])
      if(prevSum == 0 and currSum != 0):
        paraRows.append([row])
      elif(prevSum != 0 and currSum == 0):
        paraRows[-1].append(row)
      prevSum = currSum
    if(prevSum != 0 and currSum != 0):
      paraRows[-1].append(para[3])

    paraColumns = []
    charWidthSum = 0
    charCount = 0
    for (idx, row) in enumerate(paraRows):    # separating all the characters in each row of each paragraph
      prevSum = 0
      paraColumns.append([])
      for col in range(para[0], para[1]+1):
        currSum = np.sum(binary[row[0]:row[1]+1, col])
        if(prevSum == 0 and currSum != 0):
          paraColumns[idx].append([col])
        elif(prevSum != 0 and currSum == 0):
          paraColumns[idx][-1].append(col)
          aspectRatio = (paraColumns[idx][-1][1] - paraColumns[idx][-1][0] + 1) / (row[1] - row[0] + 1) if (row[1] - row[0] + 1) > 0 else 0
          if(aspectRatio > 0.6):
            charWidthSum += paraColumns[idx][-1][1] - paraColumns[idx][-1][0] + 1
            charCount += 1
        prevSum = currSum
      if(prevSum != 0 and currSum != 0):
        paraColumns[idx][-1].append(para[1])
        aspectRatio = (paraColumns[idx][-1][1] - paraColumns[idx][-1][0] + 1) / (row[1] - row[0] + 1) if (row[1] - row[0] + 1) > 0 else 0
        if(aspectRatio > 0.6):
          charWidthSum += paraColumns[idx][-1][1] - paraColumns[idx][-1][0] + 1
          charCount += 1
    avgCharWidth = charWidthSum / charCount if charCount > 0 else 0

    extractedParaChars = []
    for (rowIdx, row) in enumerate(paraRows):
      for (colIdx, col) in enumerate(paraColumns[rowIdx]):
        charBinary = binary[row[0]:row[1]+1, col[0]:col[1]+1]   # extracting the character image
        height = charBinary.shape[0]
        width = charBinary.shape[1]

        # padding the character image
        charBinary = np.pad(charBinary, ((max(5, width//10),), (max(5, height//10),)), 'constant')
        charBinary = charBinary.astype(np.uint8)

        # resizing the image to 28x28
        charBinary[charBinary > 0] = 1
        mask = charBinary
        charBinary = charBinary[np.ix_(mask.any(1),mask.any(0))]
        charBinary = Image.fromarray(charBinary)
        charBinary = charBinary.resize((28,28), Image.ANTIALIAS)
        charBinary = np.array(charBinary).astype(np.uint8)
        extractedParaChars.append(charBinary)

        # calculating the spaces between the words
        spaceWidth = 0
        if(colIdx == len(paraColumns[rowIdx])-1):
          spaceWidth = avgCharWidth
        else:
          spaceWidth = paraColumns[rowIdx][colIdx+1][0] - paraColumns[rowIdx][colIdx][1] + 1
        if(spaceWidth >= (avgCharWidth*2)/5):
          extractedParaChars.append(' ')
    extractedChars.append(extractedParaChars)
  return extractedChars

# neural network model is loaded and used to predict the text 
def predictText(extractedChars):
  text = ''
  model = tf.keras.models.load_model('ocr/model/model.h5')   # loads the model
  for para in extractedChars:   # iterating over each paragraph array
    if(len(para) > 0):
      spaces = []
      for idx in range(0, len(para)):   # storing the space indices in the paragraph
        if(type(para[idx]) == type(' ')):
          spaces.append(idx)
          para[idx] = np.zeros((28,28))

      # reshaping the characters arrays to 28x28
      para = np.array(para)
      img = []
      for el in para:
        for a in el:
          for b in a:
            img.append(b)
      para = np.array([img]).reshape(-1,28,28)

      # adding an extra dimension and padding to the character arrays as per the input shape of the model
      para = np.expand_dims(para, axis=3)
      para = np.pad(para, ((0,0),(2,2),(2,2),(0,0)), 'constant')

      # model predicts each character from the image
      predictions = tf.argmax(model.predict(para), axis=1)
      predictions = [chr(code+34) for code in predictions]

      # spaces are restored in the model predictions
      for idx in range(0, len(spaces)):
        predictions[spaces[idx]] = ' '

      predictions = ''.join(predictions)
      text += predictions
    else:
      text += ' '
  return text


def predict():
  borders = paragraphDetection()                      # detects and computes the borders of all paragraphs in the image
  extractedChars = characterDetection(borders)        # extracts characters from the detected paragraphs
  text = predictText(extractedChars)                  # predicts the text in the image from the detected characters
  return text
