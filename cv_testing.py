import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

filename = 'words_binarize.png'
img = cv.imread(filename)

tCNN = cv.text.TextDetectorCNN_create('textbox.prototxt', 'TextBoxes_icdar13.caffemodel')

Bbox, confidence = tCNN.detect(img)

for i in range(len(confidence)):
    if confidence[i] > .03:
        cv.rectangle(img,(Bbox[i][0],Bbox[i][1]),(Bbox[i][0] + Bbox[i][2], Bbox[i][1] + Bbox[i][3]),(0,255,0),3)


#retval = cv.text.loadOCRHMMClassifierCNN(filename)
#print(retval)
        
plt.imshow(img, cmap = 'gray', interpolation = 'bicubic')
plt.xticks([]), plt.yticks([])  # to hide tick values on X and Y axis

plt.show()
