import cv2
import numpy as np
import face_recognition

imgNatalie = face_recognition.load_image_file('imagesbasic/glen.jpg')
imgNatalie = cv2.cvtColor(imgNatalie, cv2.COLOR_BGR2RGB)
imgTest = face_recognition.load_image_file('imagesbasic/glenTest.jpg')
imgTest = cv2.cvtColor(imgTest, cv2.COLOR_BGR2RGB)

faceLocs = face_recognition.face_locations(imgTest)
if len(faceLocs) > 0:
    faceLocTest = faceLocs[0]
    encodeNatalie = face_recognition.face_encodings(imgNatalie)[0]
    encodeTest = face_recognition.face_encodings(imgTest)[0]
    result = face_recognition.compare_faces([encodeNatalie], encodeTest)
    cv2.rectangle(imgNatalie, (faceLocTest[3], faceLocTest[0]),
                  (faceLocTest[1], faceLocTest[2]), (255, 0, 255), 2)

    print(result)

    cv2.imshow('Natalie Tewa', imgNatalie)
    cv2.imshow('Natalie Test', imgTest)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
