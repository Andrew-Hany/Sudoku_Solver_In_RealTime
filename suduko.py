import cv2
import imutils
import numpy as np
from tensorflow import keras
from sudoko_solver import *




# processing the image
def process_image(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 1)
    th3 = cv2.adaptiveThreshold(blur, 255, 1, 1, 11, 2)
    return th3
# cv2.imshow("th22", th3)
def finging_contours(th3):
    cnts = contours, hierarchy = cv2.findContours(th3, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    max_area = -100
    valid=0
    max = np.array([])
    for c in cnts:
        # if the contour is bad, draw it on the mask
        peri = cv2.arcLength(c, True)
        area = cv2.contourArea(c)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)
        if area>100 and area > max_area and len(approx)==4:
            max_area = area
            second = max
            max = approx
            valid=1

    return max,valid
def draw_Contours(max,img):
    imgcontours = img.copy()
    cv2.drawContours(imgcontours, max, -1, (0, 255, 0), 3)
    return imgcontours

def reorder(max):
    max = max.reshape([4, 2])
    add = max.sum(1)
    diff=np.diff(max,axis=1)
    pts = np.zeros((4, 1, 2), dtype=np.int32)
    pts[0]=max[np.argmin(add)]
    pts[3]=max[np.argmax(add)]
    pts[1] = max[np.argmin(diff)]
    pts[2] = max[np.argmax(diff)]
    return pts

def wrapper(points,img):
    width = 450
    height = 450
    pts1=np.float32(points)
    pts2=np.float32([[0,0],[width,0],[0,height],[width,height]])
    matrix=cv2.getPerspectiveTransform(pts1,pts2)
    imgWrappercolor=cv2.warpPerspective(img,matrix,(width,height))
    return imgWrappercolor

def overlay(points,img,solvedimg):
    width = 450
    height = 450
    pts2=np.float32(points)
    pts1=np.float32([[0,0],[width,0],[0,height],[width,height]])
    matrix=cv2.getPerspectiveTransform(pts1,pts2)
    height1, width1 = img.shape[0:2]
    original=cv2.warpPerspective(solvedimg,matrix,(width1,height1))
    inv_prese = cv2.addWeighted(original, 1, img,0.5,0)
    return inv_prese,original

def detect_digits(img):
    digits=[]
    rows=np.vsplit(img,9)
    for r in rows:
        cols=np.hsplit(r,9)
        for box in cols:
            box=box[4:47,4:47]
            # cv2.rectangle(box,(0,0),(50,50),(255,255,255),10)
            digits.append(box)


    return digits

def load_hand_recognition_model():

    loaded_model = keras.models.load_model('digit_model_train')
    return loaded_model

def preprocessing(img):
    img=cv2.equalizeHist(img)
    img=img/255
    return img
def recognize_digits(digits,model):
    numbers=[]
    valid2=0
    for i in digits:
        img = np.asarray(i)
        img = cv2.resize(img, (32, 32))
        img = preprocessing(img)
        img = img.reshape(1, 32, 32, 1)
        classIn = int(model.predict_classes(img))

        predictions=model.predict(img)
        probval=np.amax(predictions)
        if ( classIn==0):
            numbers.append(0)
        else:
            valid2 = 1
            numbers.append(classIn)

        print(classIn,probval)
    return numbers,valid2

def write_numbers(img,img2,numbers,boolen):
    for i in range(0,450,50):
        for j in range (0, 450, 50):
            if(boolen[i//50][j//50]==1):
                img = cv2.putText(img, str(numbers[i//50][j//50]), (j+22,i+40), cv2.FONT_HERSHEY_SIMPLEX,
                                1, (0,255,0), 2, cv2.LINE_AA)
                img2 = cv2.putText(img2, str(numbers[i // 50][j // 50]), (j + 22, i + 40), cv2.FONT_HERSHEY_SIMPLEX,
                                  1, (0, 255, 0), 2, cv2.LINE_AA)
            else:
                img = cv2.putText(img, str(numbers[i // 50][j // 50]), (j + 22, i + 40), cv2.FONT_HERSHEY_SIMPLEX,
                                  1, (255, 0, 0), 2, cv2.LINE_AA)
    return img,img2

def get_empty_indices(numbers): #to get the board with ones and zeros to know what to print
    boolen=[]
    for i in range(len(numbers)):
        for j in range (len(numbers[0])):
            if(numbers[i][j]==0):
                boolen.append(1)
            else:
                boolen.append(0)
    boolen = np.reshape(boolen, [[9, 9]][0])
    return boolen




def main(img):
    normal=img.copy()


    height, width = img.shape[0:2]
    img=img[20:height,20:width,:]

    processed=process_image(img)
    biggest,valid=finging_contours(processed)


    print("__________________")

    cont=draw_Contours(biggest, img)
    points=reorder(biggest)
    imgWrappercolor=wrapper(points,img)
    warpperGrey=cv2.cvtColor(imgWrappercolor,cv2.COLOR_BGR2GRAY)
    th3 = cv2.adaptiveThreshold(warpperGrey, 255, 1, 1, 11, 2)

    #getting small pics to recognise
    digits=detect_digits(th3)

    #pass these pics to be classified
    model = load_hand_recognition_model()
    numbers,valid1=recognize_digits(digits,model)

    #covert it to 2d array

    numbers=np.reshape(numbers,[[9,9]][0])
    boolen=get_empty_indices(numbers)#getting the array that will let us know the indeices we will print
    #Finally we solve the sudoko
    solve(numbers)
    # print_board(numbers)

    #now we will print the solved sudoko
    impBlank = np.zeros((450, 450, 3), np.uint8)
    board_all = impBlank.copy()
    solved = impBlank.copy()
    board_all,solved=write_numbers(solved,board_all,numbers,boolen)

    overlayed,imgInvWarpColored=overlay(points,img,solved)

    # cv2.imshow('orignal', img)
    cv2.imshow('Thresholding', processed)
            # cv2.imshow('cont', cont)
            # cv2.imshow('warpped', warpperGrey)
            # cv2.imshow('warpped', warpperGrey)
            # cv2.imshow('board_all', board_all)
    cv2.imshow('solved', solved)
            # cv2.imshow('imgInvWarpColored', imgInvWarpColored)
    cv2.imshow('overlayed', overlayed)
    return solved

#this is for camera only (to make it easier for the cam)
def next( img,solved):

    processed = process_image(img)
    biggest, valid = finging_contours(processed)
    cont = draw_Contours(biggest, img)
    points = reorder(biggest)
    overlayed,imgInvWarpColored=overlay(points,img,solved)
    # cv2.imshow('overlayed', img)
    cv2.imshow('Thresholding', processed)
    # cv2.imshow('cont', cont)
    cv2.imshow('solved', solved)
    cv2.imshow('overlayed', overlayed)




# ___Using single photo
img = cv2.imread("TestCases/pic2.jpg")
main(img)
cv2.waitKey(0)
cv2.destroyAllWindows()





## _____________when opening camera
# cap = cv2.VideoCapture(0)
# if not cap.isOpened():
#     print("Cannot open camera")
#     exit()
# value=0
# solved=np.zeros((450, 450,3),np.uint8)
# while True:
#     # Capture frame-by-frame
#     ret, frame = cap.read()
#     frame = cv2.resize(frame, None, None, fx=0.5, fy=0.5)
#
#     if value==0:
#             solved=main(frame)
#             value=1
#     else:
#             next(frame, solved)
#
#
#     if cv2.waitKey(1) == ord('q'):
#         break
# # When everything done, release the capture
# cap.release()
# cv2.destroyAllWindows()