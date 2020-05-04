import face_recognition.api as face_recognition
import cv2,pickle,os,csv,stat
import numpy as np
from datetime import datetime

video_capture = cv2.VideoCapture(0)

with open("known_face_ids.txt","rb") as fp:
    known_face_ids = pickle.load(fp)
with open("known_face_encodings.txt","rb") as fp:
    known_face_encodings = pickle.load(fp)

CSV_PATH = "/home/harsh/Backup/face-recognition/data/attendance.csv"

if(os.path.exists(CSV_PATH)):
    csv_file = open(CSV_PATH, "a")
    writer = csv.writer(csv_file)
else:
    os.mknod(CSV_PATH)
    csv_file = open(CSV_PATH, "w")
    writer = csv.writer(csv_file)
    writer.writerow(["Student ID","Date","Time of Entry"])


face_locations = []
face_encodings = []
face_names = []
process_this_frame = True
sanity_count = 0
unknown_count = 0

while True:
    # Grab a single frame of video
    ret, frame = video_capture.read()

    # Resize frame of video to 1/4 size for faster face recognition processing
    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)

    # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
    rgb_small_frame = small_frame[:, :, ::-1]

    # Only process every other frame of video to save time
    if process_this_frame:
        # Find all the faces and face encodings in the current frame of video
        face_locations = face_recognition.face_locations(rgb_small_frame)
        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

        face_names = []
        for face_encoding in face_encodings:
            # See if the face is a match for the known face(s)
            matches = face_recognition.compare_faces(known_face_encodings, face_encoding, tolerance = 0.35)
            name = "Unknown"

            # # If a match was found in known_face_encodings, just use the first one.
            # if True in matches:
            #     first_match_index = matches.index(True)
            #     name = known_face_ids[first_match_index]

            # Or instead, use the known face with the smallest distance to the new face
            face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
            # print(face_distances)
            best_match_index = np.argmin(face_distances)
            if matches[best_match_index]:
                name = known_face_ids[best_match_index]

            face_names.append(name)
    if(sanity_count == 0):
        prev_name = name
        sanity_count+=1
    elif(sanity_count<60):
        if(prev_name == name and name != "Unknown"):
            sanity_count+=1
            prev_name = name
        else:
            sanity_count = 0
    elif(sanity_count == 60):
        print("face registered")
        cv2.destroyAllWindows()
        sanity_count = 0
        now = datetime.now()
        dt_string = now.strftime("%d/%m/%Y %H:%M:%S")
        date = dt_string.split(" ")[0]
        time = dt_string.split(" ")[1]
        writer.writerow([name,date,time])
        # print(name + date + time)
        break


    if(name == "Unknown"):
        unknown_count+=1
    else:
        unknown_count = 0
    if(unknown_count == 600):
        cv2.destroyAllWindows()
        print("You haven't been registered")
        unknown_count = 0
        break

    process_this_frame = not process_this_frame


    # Display the results
    for (top, right, bottom, left), name in zip(face_locations, face_names):
        # Scale back up face locations since the frame we detected in was scaled to 1/4 size
        top *= 4
        right *= 4
        bottom *= 4
        left *= 4

        # Draw a box around the face
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

        # Draw a label with a name below the face
        cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)

    # Display the resulting image
    cv2.imshow('Video', frame)
    cv2.waitKey(1)
    # Hit 'q' on the keyboard to quit!


# Release handle to the webcam
video_capture.release()
cv2.destroyAllWindows()