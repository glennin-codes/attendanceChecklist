import cv2
import numpy as np
import face_recognition

imgNatalie = cv2.imread('imagesbasic/glen.jpg')
imgNatalie = cv2.cvtColor(imgNatalie, cv2.COLOR_BGR2RGB)
imgTest = cv2.imread('imagesbasic/glenTest.jpg')
imgTest = cv2.cvtColor(imgTest, cv2.COLOR_BGR2RGB)

cv2.imshow('Natalie Tewa', imgNatalie)
cv2.imshow('Natalie Test', imgTest)
cv2.waitKey(0)

Natalie_encoding = face_recognition.face_encodings(imgNatalie)[0]
test_encoding = face_recognition.face_encodings(imgTest)[0]

results = face_recognition.compare_faces([Natalie_encoding], test_encoding)

if results[0] == True:
    print('They are same person')
else:
    print('They are not same person')