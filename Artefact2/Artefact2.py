#!/usr/bin/env python
# coding: utf-8

# In[1]:


#importing OpenCV library
import cv2

#loading eye detection cascade classifier training files for haarcascade
haar_eye_cascade = cv2.CascadeClassifier('data/haarcascade_eye.xml')
haar_eye_cascade_glasses = cv2.CascadeClassifier('data/haarcascade_eye_tree_eyeglasses.xml')

#loading facial detection cascade classifier training files for haarcascade
haar_face_cascade = cv2.CascadeClassifier('data/haarcascade_frontalface_alt.xml')
haar_face_cascade_default = cv2.CascadeClassifier('data/haarcascade_frontalface_default.xml')
haar_face_cascade_tree = cv2.CascadeClassifier('data/haarcascade_frontalface_alt_tree.xml')

def detect_eyes_faces(eye_cascade, face_cascade, coloured_img):
    #just making a copy of image passed, so that passed image is not changed
    img_copy = coloured_img.copy()
    #convert the test image to gray image as opencv face detector expects gray images
    gray = cv2.cvtColor(img_copy, cv2.COLOR_BGR2GRAY)       
    #let's detect multiscale (some images may be closer to camera than others) images
    eyes = eye_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=6);
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5);   
    #print the number of faces found
    #print('Faces found: ', len(faces))
    #print the number of faces found
    #print('Eyes found: ', len(eyes))
    #go over list of faces and draw them as rectangles on original colored img
    for (x, y, w, h) in eyes:
        cv2.rectangle(img_copy, (x, y), (x+w, y+h), (0, 0, 255), 2)
    #go over list of faces and draw them as rectangles on original colored img
    for (x, y, w, h) in faces:
        cv2.rectangle(img_copy, (x, y), (x+w, y+h), (0, 255, 0), 2)
    return img_copy



# Testing haarcascade_eye.xml with haarcascade_frontalface_alt.xml

#loading an image
img = cv2.imread('data/test3.jpg')
#calling the function to detect faces and eyes
detected_img = detect_eyes_faces(haar_eye_cascade, haar_face_cascade, img)
#displaying the final image
cv2.imshow('Testing haarcascade_eye.xml with haarcascade_frontalface_alt.xml', detected_img)
#window is destroyed when 0 is pressed
cv2.waitKey(0)
cv2.destroyAllWindows()

#loading another image
img = cv2.imread('data/test5.jpg')
detected_img = detect_eyes_faces(haar_eye_cascade, haar_face_cascade, img)
cv2.imshow('Testing haarcascade_eye.xml with haarcascade_frontalface_alt.xml', detected_img)
cv2.waitKey(0)
cv2.destroyAllWindows()



# Testing haarcascade_eye.xml with haarcascade_frontalface_default.xml

img = cv2.imread('data/test3.jpg')
detected_img = detect_eyes_faces(haar_eye_cascade, haar_face_cascade_default, img)
cv2.imshow('Testing haarcascade_eye.xml with haarcascade_frontalface_default.xml', detected_img)
cv2.waitKey(0)
cv2.destroyAllWindows()

img = cv2.imread('data/test5.jpg')
detected_img = detect_eyes_faces(haar_eye_cascade, haar_face_cascade_default, img)
cv2.imshow('Testing haarcascade_eye.xml with haarcascade_frontalface_default.xml', detected_img)
cv2.waitKey(0)
cv2.destroyAllWindows()



# Testing haarcascade_eye.xml with haarcascade_frontalface_alt_tree.xml

img = cv2.imread('data/test3.jpg')
detected_img = detect_eyes_faces(haar_eye_cascade, haar_face_cascade_tree, img)
cv2.imshow('Testing haarcascade_eye.xml with haarcascade_frontalface_alt_tree.xml', detected_img)
cv2.waitKey(0)
cv2.destroyAllWindows()

img = cv2.imread('data/test5.jpg')
detected_img = detect_eyes_faces(haar_eye_cascade, haar_face_cascade_tree, img)
cv2.imshow('Testing haarcascade_eye.xml with haarcascade_frontalface_alt_tree.xml', detected_img)
cv2.waitKey(0)
cv2.destroyAllWindows()



# Testing haarcascade_eye_tree_eyeglasses.xml with haarcascade_frontalface_alt.xml

img = cv2.imread('data/test3.jpg')
detected_img = detect_eyes_faces(haar_eye_cascade_glasses, haar_face_cascade, img)
cv2.imshow('Testing haarcascade_eye_tree_eyeglasses.xml with haarcascade_frontalface_alt.xml', detected_img)
cv2.waitKey(0)
cv2.destroyAllWindows()

img = cv2.imread('data/test5.jpg')
detected_img = detect_eyes_faces(haar_eye_cascade_glasses, haar_face_cascade, img)
cv2.imshow('Testing haarcascade_eye_tree_eyeglasses.xml with haarcascade_frontalface_alt.xml', detected_img)
cv2.waitKey(0)
cv2.destroyAllWindows()



# Testing more images with haarcascade_eye.xml with haarcascade_frontalface_alt.xml

#loading an image
img = cv2.imread('data/test1.jpg')
detected_img = detect_eyes_faces(haar_eye_cascade, haar_face_cascade, img)
cv2.imshow('Testing haarcascade_eye.xml with haarcascade_frontalface_alt.xml', detected_img)
cv2.waitKey(0)
cv2.destroyAllWindows()

#loading another image
img = cv2.imread('data/test4.jpg')
detected_img = detect_eyes_faces(haar_eye_cascade, haar_face_cascade, img)
cv2.imshow('Testing haarcascade_eye.xml with haarcascade_frontalface_alt.xml', detected_img)
cv2.waitKey(0)
cv2.destroyAllWindows()



# Testing live webcam feed with haarcascade_eye.xml with haarcascade_frontalface_alt.xml

#launching the main camera
cam = cv2.VideoCapture(0)

#looping until e is pressed
while True:
    #getting the frames
    ret, frame = cam.read()
    #caling the function on the returned frame
    frame = detect_eyes_faces(haar_eye_cascade, haar_face_cascade, frame)
    
    #displaying the frame with the rectangles 
    cv2.imshow('Testing haarcascade_eye.xml with haarcascade_frontalface_alt.xml', frame)
    
    #if e is pressed, exit the loop
    if(cv2.waitKey(1) == ord('e')):
        break

#camera is closed and window is destroyed when e is pressed
cam.release()
cv2.destroyAllWindows()



# def detect_faces(face_cascade, colored_img, scaleFactor = 1.1):
#     img_copy = colored_img.copy()
#     gray = cv2.cvtColor(img_copy, cv2.COLOR_BGR2GRAY)       
#     faces = face_cascade.detectMultiScale(gray, scaleFactor=scaleFactor, minNeighbors=5);   
#     #print the number of faces found
#     #print('Faces found: ', len(faces))
#     for (x, y, w, h) in faces:
#         cv2.rectangle(img_copy, (x, y), (x+w, y+h), (0, 255, 0), 2)
#     return img_copy


# def detect_eyes_glasses(e_g_cascade, colored_img, scaleFactor = 1.3):
#     img_copy = colored_img.copy()
#     gray = cv2.cvtColor(img_copy, cv2.COLOR_BGR2GRAY)       
#     eyes = e_g_cascade.detectMultiScale(gray, scaleFactor=scaleFactor, minNeighbors=1);   
#     #print the number of faces found
#     print('Eyes found: ', len(eyes))
#     for (x, y, w, h) in eyes:
#         cv2.rectangle(img_copy, (x, y), (x+w, y+h), (0, 0, 255), 2)
#     return img_copy


# def detect_eyes(e_cascade, colored_img, scaleFactor = 1.3):
#     img_copy = colored_img.copy()
#     gray = cv2.cvtColor(img_copy, cv2.COLOR_BGR2GRAY)       
#     eyes = e_cascade.detectMultiScale(gray, scaleFactor=scaleFactor, minNeighbors=6);   
#     #print the number of faces found
#     print('Eyes found: ', len(eyes))
#     for (x, y, w, h) in eyes:
#         cv2.rectangle(img_copy, (x, y), (x+w, y+h), (0, 0, 255), 2)
#     return img_copy


# test2 = cv2.imread('data/test5.jpg')
# detected_img = detect_faces(haar_face_cascade, test2)
# #detected_img = detect_eyes_glasses(haar_eye_cascade, test2)
# cv2.imshow('Test Imag', detected_img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()


# test2 = cv2.imread('data/test3.jpg')
# detected_img = detect_faces(haar_face_cascade, test2)
# detected_img = detect_eyes_glasses(haar_eye_cascade, test2)
# cv2.imshow('Test Imag', detected_img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()


# test2 = cv2.imread('data/test5.jpg')
# eyes_detected_img = detect_eyes_glasses(haar_eye_cascade, test2)
# cv2.imshow('Test Imag', eyes_detected_img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()


# test2 = cv2.imread('data/test5.jpg')
# eyes_detected_img = detect_eyes(haar_eye_cascade, test2)
# cv2.imshow('Test Imag', eyes_detected_img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

