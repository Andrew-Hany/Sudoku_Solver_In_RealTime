import  numpy as np
import pandas as pd
import cv2
import os
from sklearn.model_selection import train_test_split
from keras.preprocessing.image import ImageDataGenerator
from keras.utils.np_utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense,Flatten,Dropout
from keras.optimizers import Adam
from keras.layers.convolutional import Conv2D,MaxPooling2D



############# Dependencies
path="myData"
mylist=os.listdir(path)
nomOfClass=len(mylist)
images=[]
classNum=[]
print(nomOfClass)
######loading the images....
print("loading the images....")
for x in range(0,nomOfClass):
    myDatalist=os.listdir(path+"/"+str(x))
    for y in myDatalist:
        curImg=cv2.imread(path+"/"+str(x)+"/"+y)
        curImg=cv2.resize(curImg,(32,32))
        images.append(curImg)
        classNum.append(x)

    print(x,end=" ")
print(" ")
print(len(images))

images=np.array(images)
classNum=np.array(classNum)

########### Split train, test
x_train,x_test, y_train,y_test=train_test_split(images,classNum,test_size=0.2)


def preprocessing(img):
    img=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(img, (5, 5), 1)
    th3 = cv2.adaptiveThreshold(blur, 255, 1, 1, 11, 2)
    img=cv2.equalizeHist(th3)
    img=img/255
    return img

x_train=np.array(list(map(preprocessing,x_train)))
x_train=x_train.reshape(x_train.shape[0],x_train.shape[1],x_train.shape[2],1)

x_test=np.array(list(map(preprocessing,x_test)))
x_test=x_test.reshape(x_test.shape[0],x_test.shape[1],x_test.shape[2],1)


dataGen=ImageDataGenerator(width_shift_range=0.1,
                           height_shift_range=0.1,zoom_range=0.2,
                           shear_range=0.1,rotation_range=10 )
dataGen.fit(x_train)

y_train=to_categorical(y_train, nomOfClass)
y_test=to_categorical(y_test, nomOfClass)



model=Sequential()
model.add((Conv2D(60,(5,5),input_shape=(32,32,1),activation="relu")))
model.add((Conv2D(60,(5,5),activation="relu")))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add((Conv2D(30,(3,3),activation="relu")))
model.add((Conv2D(30,(3,3),activation="relu")))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Dropout(0.5))
model.add(Flatten())

model.add(Dense(500,activation="relu"))
model.add(Dropout(0.5))
model.add(Dense(nomOfClass,activation="softmax"))
model.compile(Adam(lr=0.001),loss='categorical_crossentropy',metrics=['accuracy'])

history=model.fit_generator(dataGen.flow(x_train,y_train,batch_size=50),epochs=10,shuffle=1)
model.save("digit_model_train")
