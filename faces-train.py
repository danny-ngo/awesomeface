import os
import numpy as np
from PIL import Image
import cv2
import pickle

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
image_dir = os.path.join(BASE_DIR, "images")

face_cascade = cv2.CascadeClassifier('cascades/data/haarcascade_frontalface_alt2.xml')
recognizer = cv2.face.LBPHFaceRecognizer_create()


current_id = 0
label_ids = {}
y_labels = []
x_train = []

for root1, dirs1, files1 in os.walk(image_dir):
    for dir in dirs1:
        celeb_dir = os.path.join(image_dir,dir)
        for root, dirs, files in os.walk(celeb_dir):
            # print(files)
            print(celeb_dir)
    
            index = 0
            for file in files:
                
                # print(file)
                if index < 6:
                    if file.endswith("png") or file.endswith("jpg"):
                        index += 1
                        path = os.path.join(root, file)
                        label = os.path.basename(os.path.dirname(path)).replace(" ","-").lower()
                        # print(label, path)
                        if not label in label_ids:
                            label_ids[label] = current_id
                            current_id += 1
                        id_ = label_ids[label]
                        # print(label_ids)
                        # y_labels.append(path) # some number
                        # x_train.append(path) # verify this image, turn into a numpy array, gray
                        pil_image = Image.open(path).convert("L") # grayscale
                        size = (250, 250)
                        final_image = pil_image.resize(size, Image.ANTIALIAS)
                        image_array = np.array(final_image, "uint8")
                        # print(image_array)
                        faces = face_cascade.detectMultiScale(image_array,scaleFactor=1.5, minNeighbors=5)

                        for (x,y,w,h) in faces:
                            roi = image_array[y:y+h, x:x+w]
                            x_train.append(roi)
                            y_labels.append(id_)
                

                            
                    

with open("labels.pickle", "wb") as f:
    pickle.dump(label_ids, f)

recognizer.train(x_train, np.array(y_labels))
recognizer.save("trainer.yml")

# print(y_labels)
# print(x_train)
