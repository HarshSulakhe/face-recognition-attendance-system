import face_recognition.api as face_recognition
import cv2, os, time, pickle
import numpy as np
# from mtcnn import MTCNN
import json
import matplotlib.pyplot as plt
import matplotlib as mpl
# load image using cv2....and do processing.


def register_yourself(student_id):

    mpl.rcParams['toolbar'] = 'None'
    # PATH = "/home/harsh/Backup/face-recognition/data"
    PATH = "/home/harsh/face-recognition-attendance-system/static/data"
    STORAGE_PATH = "/home/harsh/face-recognition-attendance-system/storage"

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

    check, image = video_capture.read()

    # print("BEFORE SHOWING")
    plot = plt.subplot(1,1,1)
    plt.axis('off')
    plt.title("Registering face, wait for a bit")
    im1 = plot.imshow(cv2.cvtColor(image,cv2.COLOR_BGR2RGB))
# print("WORKS reg")

    while j < start + 10:   # Take 10 more images

        # if not video_capture.isOpened():
        #     print("ERROR OPENING CV")
        i += 1
        check, image = video_capture.read()

        # print("BEFORE SHOWING")
        plt.ion()
        im1.set_data(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        plt.pause(0.001)
        # as opencv loads in BGR format by default, we want to show it in RGB.
        # cv2.imshow("Now taking images (Wait for some time)", image)
        # cv2.waitKey(20)  #Display frame for 1 ms
        # print("AFTER SHOWING")
        # Take 30 frames
        plt.show()
        if(i % 30 == 0):
            cv2.imwrite(IMAGE_PATH + "/{}_".format(student_id) + str(j) + ".jpg", image)
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

    video_capture.release()
    cv2.destroyAllWindows()
    # video_capture.

    with open( os.path.join(STORAGE_PATH, "id_idx.json"),"w") as outfile:
        json.dump(id_idx, outfile)

    # Exit time
    toc = time.time()
    # print(toc - tic)
    plt.close()
# register_yourself()
