import cv2 as cv
import numpy as np
import os 
recognizer = cv.face.LBPHFaceRecognizer_create()
recognizer.read('trainer/trainer.yml')
cascadePath = "haarcascade_frontalface_default.xml"
faceCascade = cv.CascadeClassifier(cascadePath);
font = cv.FONT_HERSHEY_SIMPLEX
#iniciate id counter
id = 0
# names related to ids: example ==> Marcelo: id=1,  etc
names = ['None', 'Shubhang', 'None', 'None', 'None', 'None'] 
# Initialize and start realtime video capture
cam = cv.VideoCapture(0)
cam.set(3, 640) # set video widht
cam.set(4, 480) # set video height
# Define min window size to be recognized as a face
minW = 0.1*cam.get(3)
minH = 0.1*cam.get(4)
while True:
    ret, img =cam.read()
    img = cv.flip(img, 1) 
    gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
    
    faces = faceCascade.detectMultiScale( 
        gray,
        scaleFactor = 1.2,
        minNeighbors = 5,
        minSize = (int(minW), int(minH)),
       )
    for(x,y,w,h) in faces:
        cv.rectangle(img, (x,y), (x+w,y+h), (0,0,0), 2)
        id, confidence = recognizer.predict(gray[y:y+h,x:x+w])
        
        # If confidence is less them 100 ==> "0" : perfect match 
        if (confidence < 100):
            id = names[id]
            confidence = "  {0}%".format(round(100 - confidence))
        else:
            id = "unknown"
            confidence = "  {0}%".format(round(100 - confidence))
        
        cv.putText(
                    img, 
                    str(id), 
                    (x+5,y-5), 
                    font, 
                    1, 
                    (255,0,0), 
                    2
                   )
        cv.putText(
                    img, 
                    str(confidence), 
                    (x+5,y+h-5), 
                    font, 
                    1, 
                    (0,255,255), 
                    1
                   )  
    
    cv.imshow('camera',img) 
    k = cv.waitKey(10) & 0xff # Press 'ESC' for exiting video
    if k == 27:
        break
# Do a bit of cleanup
cam.release()
cv.destroyAllWindows()
