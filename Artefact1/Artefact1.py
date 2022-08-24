#!/usr/bin/env python
# coding: utf-8


#importing OpenCV library
import cv2

#loading cascade classifiers 
face_cascade = cv2.CascadeClassifier('data/cascade.xml')
face_cascade2 = cv2.CascadeClassifier('data2/cascade.xml')

def detect_faces(f_cascade, colored_img):
    img_copy = colored_img.copy()
    gray = cv2.cvtColor(img_copy, cv2.COLOR_BGR2GRAY)
    faces = f_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5);
    #print the number of faces found
    #print('Faces found: ', len(faces))
    for (x, y, w, h) in faces:
        cv2.rectangle(img_copy, (x, y), (x+w, y+h), (0, 255, 0), 2)
    return img_copy


img = cv2.imread('data/test5.jpg')
detected_img = detect_faces(face_cascade, img)
cv2.imshow('Figure 22', detected_img)
cv2.waitKey(0)
cv2.destroyAllWindows()

img = cv2.imread('data/test5.jpg')
detected_img = detect_faces(face_cascade2, img)
cv2.imshow('Figure 11', detected_img)
cv2.waitKey(0)
cv2.destroyAllWindows()

img = cv2.imread('data/test3.jpg')
detected_img = detect_faces(face_cascade, img)
cv2.imshow('Figure 22', detected_img)
cv2.waitKey(0)
cv2.destroyAllWindows()

img = cv2.imread('data/test3.jpg')
detected_img = detect_faces(face_cascade2, img)
cv2.imshow('Figure 11', detected_img)
cv2.waitKey(0)
cv2.destroyAllWindows()

img = cv2.imread('data/test4.jpg')
detected_img = detect_faces(face_cascade, img)
cv2.imshow('Figure 22', detected_img)
cv2.waitKey(0)
cv2.destroyAllWindows()

img = cv2.imread('data/test4.jpg')
detected_img = detect_faces(face_cascade2, img)
cv2.imshow('Figure 11', detected_img)
cv2.waitKey(0)
cv2.destroyAllWindows()
