import face_recognition.api as face_recognition
import cv2, os, time, pickle
import numpy as np
# from mtcnn import MTCNN
import json

# PATH = "/home/harsh/Backup/face-recognition/data"
PATH = "/home/vishal/Documents/face-recognition-attendance-system/static/data"
STORAGE_PATH = "/home/vishal/Documents/face-recognition-attendance-system/storage"

try:
    os.makedirs(PATH)
except:
    pass

try:
    with open( os.path.join(STORAGE_PATH, "known_face_ids.pickle"),"rb") as fp:
        known_face_ids = pickle.load(fp)
    with open( os.path.join(STORAGE_PATH, "known_face_encodings.pickle"),"rb") as fp:
        known_face_encodings = pickle.load(fp)
    # known_face_ids = np.load("known_face_ids.npy")
    # known_face_encodings = np.load("known_face_encodings.npy")
except:
    known_face_encodings = []
    known_face_ids = []

try:
    with open( os.path.join(STORAGE_PATH, "id_idx.json"),"r") as fp:
        id_idx = json.load(fp)
except:
    id_idx = {}

def register_yourself(student_id):
    video_capture = cv2.VideoCapture(0)
    # student_id = input("Enter your id: ")

    IMAGE_PATH = os.path.join(PATH, student_id)

    try:
        os.makedirs(IMAGE_PATH)
    except:
        pass

    try:
        start = id_idx[student_id]
    except :
        start = 0

    #Entry time
    tic = time.time()

    i = 0
    j = start

    while j < start + 10:   # Take 10 more images
        i += 1
        check, image = video_capture.read()

        cv2.imshow("Now taking images (Wait for some time)", image)
        cv2.waitKey(1)  #Display frame for 1 ms

        # Take 30 frames
        if(i % 30 == 0):
            cv2.imwrite(IMAGE_PATH + "/test_" + str(j) + ".jpg", image)
            try:
                known_face_encodings.append(face_recognition.face_encodings(image)[0])
                known_face_ids.append(student_id)
            except:
                continue
            j += 1

    with open( os.path.join(STORAGE_PATH, "known_face_ids.pickle"),"wb") as fp:
        pickle.dump(known_face_ids,fp)
    with open( os.path.join(STORAGE_PATH, "known_face_encodings.pickle"),"wb") as fp:
        pickle.dump(known_face_encodings,fp)

    id_idx[student_id] = start + 10

    cv2.destroyAllWindows()

    with open( os.path.join(STORAGE_PATH, "id_idx.json"),"w") as outfile:
        json.dump(id_idx, outfile)

    # Exit time
    toc = time.time()
    print(toc - tic)


# register_yourself()