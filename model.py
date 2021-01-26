import numpy
import cv2
import os
from PIL import Image
import pickle

faces=cv2.CascadeClassifier('C:\\Users\\Nikhil\\AppData\\Local\\Programs\\Python\\Python35\\lib\\site-packages\\cv2\\data\\haarcascade_frontalface_alt2.xml')
recognizer=cv2.face.LBPHFaceRecognizer_create()

base=os.path.dirname(os.path.abspath(__file__))
images=os.path.join(base,"faces")

c=0
label_id={}
y_labels=[]
x_train=[]

for root,dirs,files in os.walk(images):
    for file in files:
        if file.endswith(".jpg") or file.endswith(".jpeg"):
            path=os.path.join(root,file)
            label=os.path.basename(root)

            pil=Image.open(path).convert("L")
            im2ar=pil.resize((100,100))
            im2ar=numpy.array(pil)
            faces1=faces.detectMultiScale(im2ar,scaleFactor=1.5,minNeighbors=5)

            if label not in label_id:
                label_id[label]=c
                c+=1
            
            id_=label_id[label]

            for (x,y,w,h) in faces1:
                roi=im2ar[y:y+h,x:x+w]

            x_train.append(roi)
            y_labels.append(id_)

with open("labels.pickle",'wb') as f:
    pickle.dump(label_id,f)

recognizer.train(x_train,numpy.array(y_labels))
recognizer.save("trained.yml")
print(label_id)

