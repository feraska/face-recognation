import os
import sys
import face_recognition as fr
import cv2
import numpy as np

video_capture = cv2.VideoCapture(0)

path_dir = "images/"

known_face_encodings = []
known_face_names = []

face_locations = []
face_encodings = []
face_names = []
process_this_frame = True

try:
    for name in os.listdir(path_dir):
        image = fr.load_image_file(path_dir+ name)
        face_encoding = fr.face_encodings(image)
        if len(face_encoding) > 0:
            face_encoding = face_encoding[0]
            known_face_encodings.append(face_encoding)
            known_face_names.append(name[0:name.index(".")])
        else:
            print("No faces found in the image!")
        

    while True:

        ret, frame = video_capture.read()
        if not ret:
            break
        if process_this_frame:

            small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)

            rgb_small_frame = small_frame[:, :, ::-1]

            face_locations = fr.face_locations(rgb_small_frame)
            face_encodings = fr.face_encodings(rgb_small_frame, face_locations)

            face_names = []
            for face_encoding in face_encodings:

                matches = fr.compare_faces(known_face_encodings, face_encoding)
                name = "Unknown"
                face_distances = fr.face_distance(known_face_encodings, face_encoding)
                best_match_index = np.argmin(face_distances)
                if matches[best_match_index]:
                    name = known_face_names[best_match_index]

                face_names.append(name)

        process_this_frame = not process_this_frame
        for (top, right, bottom, left), name in zip(face_locations, face_names):
            top *= 4
            right *= 4
            bottom *= 4
            left *= 4

            cv2.rectangle(frame, (left, top), (right, bottom), (255, 0, 0), 2)

            cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (255, 0, 0), cv2.FILLED)
            font = cv2.FONT_HERSHEY_DUPLEX
            cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)

        cv2.imshow('frame', frame)
        key = cv2.waitKey(1)
        if key == 27:
            break
                
    video_capture.release()
    cv2.destroyAllWindows()
except Exception as e:
    print(e)
    sys.exit(1)
