import numpy as np
import cv2
import PIL

face=cv2.CascadeClassifier('C:\\Users\\Nikhil\\AppData\\Local\\Programs\\Python\\Python35\\lib\\site-packages\\cv2\\data\\haarcascade_frontalface_alt2.xml')
faces1=cv2.CascadeClassifier('C:\\Users\\Nikhil\\AppData\\Local\\Programs\\Python\\Python35\\lib\\site-packages\\cv2\\data\\haarcascade_frontalface_alt.xml')
recognizer=cv2.face.LBPHFaceRecognizer_create()
recognizer.read("trained.yml")
names={'harsha': 1, 'sid': 5, 'srikar': 6, 'kathy': 2, 'eshaan': 0, 'me': 4, 'mallik': 3}
cap=cv2.VideoCapture(0)
name=""


while 1:
    ret,frame=cap.read()
    gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    faces=face.detectMultiScale(gray,scaleFactor=1.5,minNeighbors=5)
    
    
    for (x,y,w,h) in faces:
        colour=(100,150,100)
        stroke=2
        cv2.rectangle(frame,(x,y),(x+w,y+h),colour,stroke)
        roi=gray[y:y+h,x:x+w]
        text=cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(frame,name,(x+5,y+h),text,1,(225,0,100),2,cv2.LINE_AA,)
    try:
        id_,con=recognizer.predict(roi)
        if con >= 80:
            name=list(names.keys())[list(names.values()).index(id_)]

    except NameError:
        pass
    

    cv2.imshow("OK",frame)
    


    cv2.waitKey(20)
    if cv2.getWindowProperty('OK',1) == -1 :
        break

    
cap.release()