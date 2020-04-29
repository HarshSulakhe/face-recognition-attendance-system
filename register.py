import face_recognition.api as face_recognition
import cv2,os,time,pickle
import numpy as np
from mtcnn import MTCNN
import json

video_capture = cv2.VideoCapture(0)
student_id = input("Enter your id")

PATH = "/home/harsh/Backup/face-recognition/data"

try:
    os.mkdir(PATH)
except:
    pass
IMAGE_PATH = os.path.join(PATH,student_id)

try:
    with open("known_face_ids.txt","rb") as fp:
        known_face_ids = pickle.load(fp)
    with open("known_face_encodings.txt","rb") as fp:
        known_face_encodings = pickle.load(fp)
    # known_face_ids = np.load("known_face_ids.npy")
    # known_face_encodings = np.load("known_face_encodings.npy")
except:
    known_face_encodings = []
    known_face_ids = []

try:
    with open("id_idx.json","r") as fp:
        id_idx = json.load(fp)
except:
    id_idx = {}

try:
    start = id_idx[student_id]
except :
    start = 0

tic = time.time()

i = 0
j = start
while j<start+10:
    i+=1
    ret,image = video_capture.read()

    try:
        os.mkdir(IMAGE_PATH)
    except:
        pass

    cv2.imshow("Now taking images",image)
    cv2.waitKey(1)
    if(i%30==0):
        cv2.imwrite(IMAGE_PATH + "/test_"+ str(j) +".jpg",image)
        try:
            known_face_encodings.append(face_recognition.face_encodings(image)[0])
            known_face_ids.append(student_id)
        except:
            continue
        j+=1

with open("known_face_ids.txt","wb") as fp:
    pickle.dump(known_face_ids,fp)
with open("known_face_encodings.txt","wb") as fp:
    pickle.dump(known_face_encodings,fp)

id_idx[student_id] = start + 10

cv2.destroyAllWindows()

with open("id_idx.json", 'w') as outfile:
    json.dump(id_idx, outfile)
toc = time.time()
print(toc-tic)
