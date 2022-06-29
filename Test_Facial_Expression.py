
# Libraries Importation

import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing import image
import cv2
import numpy as np

# face detection with mtcnn on a photograph
from matplotlib import pyplot
from matplotlib.patches import Rectangle
from mtcnn.mtcnn import MTCNN

import logging
tf.get_logger().setLevel(logging.ERROR)

model=load_model('facial_emotion_model.h5')
face_classifier=cv2.CascadeClassifier(r'haarcascade_frontalface_default.xml')

class_labels=['Angry','Happy','Sad','Neutral','Surprised']


# ## INFERENCE FOR FACIAL EXPRESSION ON IMAGES CNN+CASCADE CLASSIFIER AND CNN+MTCNN

# ### USING CASCADE CLASSIFIER




image =cv2.imread("test_image_1.jpg")
gray=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
faces=face_classifier.detectMultiScale(gray,1.3,5)
for (x,y,w,h) in faces:
    cv2.rectangle(image,(x,y),(x+w,y+h),(255,255,0),2) 
    roi_gray=gray[y:y+h,x:x+w] # Get the region of interst
    roi_gray=cv2.resize(roi_gray,(48,48),interpolation=cv2.INTER_AREA)
    if np.sum([roi_gray])!=0: # if there is face
        roi=roi_gray.astype('float')/255.0 #resizing the ROI by dividing it by factor of 1/255
        roi=img_to_array(roi) # Convert the image into array
        roi=np.expand_dims(roi,axis=0)
        preds=model.predict(roi)[0] # Apply the prediction model on detected faces
        label=class_labels[preds.argmax()] 
        label_position=(x,y)
        cv2.putText(image,label,label_position,cv2.FONT_HERSHEY_SIMPLEX,2,(0,255,0),3)
    else:
        cv2.putText(image,'NO FACE FOUND',(20,60),cv2.FONT_HERSHEY_SIMPLEX,2,(0,255,0),3)
cv2.imshow('Emotion Detector',image)
cv2.waitKey(0)
cv2.destroyAllWindows()


# ### USING MTCNN



filename = 'set_of_images.PNG'
image = cv2.imread(filename)
gray=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
# create the detector, using default weights
detector = MTCNN()
# detect faces in the image
faces = detector.detect_faces(image)

for result in faces:
    # get coordinates
    x, y, width, height = result['box']
    # create the shape
    rect = Rectangle((x, y), width, height, fill=False, color='red')
    roi_gray=gray[y:y+height,x:x+width] # Get the region of interst
    roi_gray=cv2.resize(roi_gray,(48,48),interpolation=cv2.INTER_AREA)
    if np.sum([roi_gray])!=0: # if there is face
        roi=roi_gray.astype('float')/255.0 #resizing the ROI by dividing it by factor of 1/255
        roi=img_to_array(roi) # Convert the image into array
        roi=np.expand_dims(roi,axis=0)
        preds=model.predict(roi)[0] # Apply the prediction model on detected faces
        label=class_labels[preds.argmax()]
        label_position=(x,y)
        cv2.putText(image,label,label_position,cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),2)
        cv2.imwrite("emotion_detected_2.png", image)
    else:
        cv2.putText(image,'NO FACE FOUND',(20,60),cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),2)
        
cv2.imshow('Facial Emotion Detector',image)
cv2.waitKey(0)
cv2.destroyAllWindows()




# ## INFERENCE ON VIDEO


cap=cv2.VideoCapture(0)

while True:
    ret,frame=cap.read()
    gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    faces=face_classifier.detectMultiScale(gray,1.3,5)
    for (x,y,w,h) in faces:
        cv2.rectangle(frame,(x,y),(x+w,y+h),(255,255,0),2)
        roi_gray=gray[y:y+h,x:x+w]
        roi_gray=cv2.resize(roi_gray,(48,48),interpolation=cv2.INTER_AREA)
        if np.sum([roi_gray])!=0:
            roi=roi_gray.astype('float')/255.0
            roi=img_to_array(roi)
            roi=np.expand_dims(roi,axis=0)
            preds=model.predict(roi)[0]
            label=class_labels[preds.argmax()] 
            label_position=(x,y)
            cv2.putText(frame,label,label_position,cv2.FONT_HERSHEY_SIMPLEX,2,(0,255,0),3)
        else:
            cv2.putText(frame,'NO FACE FOUND',(20,60),cv2.FONT_HERSHEY_SIMPLEX,2,(0,255,0),3)
    cv2.imshow('Emotion Detector',frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()





# ### Inference on Video USING MTCNN



cap=cv2.VideoCapture(0)

while True: 
    
    ret,frame=cap.read()
    gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    # create the detector, using default weights
    detector = MTCNN()
    # detect faces in the image
    faces = detector.detect_faces(frame)

    for result in faces:
        # get coordinates
        x, y, width, height = result['box']
        # create the shape
        rect = Rectangle((x, y), width, height, fill=False, color='red')
        roi_gray=gray[y:y+height,x:x+width]
        roi_gray=cv2.resize(roi_gray,(48,48),interpolation=cv2.INTER_AREA)
        if np.sum([roi_gray])!=0:
            roi=roi_gray.astype('float')/255.0
            roi=img_to_array(roi)
            roi=np.expand_dims(roi,axis=0)
            preds=model.predict(roi)[0] 
            label=class_labels[preds.argmax()]
            label_position=(x,y)
            cv2.putText(frame,label,label_position,cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),2)
        else:
            cv2.putText(frame,'NO FACE FOUND',(20,60),cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),2)

    cv2.imshow('Facial Emotion Detector',frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
        
cap.release()    
cv2.destroyAllWindows()

