import cv2
import imutils
import numpy as np
from tensorflow import keras
model = keras.models.load_model('digit_model_train')





def preprocessing(img):
    img=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    img=cv2.equalizeHist(img)
    img=img/255
    return img

img = cv2.imread("10.png")
img=np.asarray(img)
img=cv2.resize(img,(32,32))
img=preprocessing(img)
img=img.reshape(1,32,32,1)
classIn=int(model.predict_classes(img))
print(classIn)
