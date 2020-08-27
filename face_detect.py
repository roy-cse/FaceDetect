import cv2
from os import listdir
from ffpyplayer.player import MediaPlayer

import os

# pre-trained data to classify front faces
face_data_frontal_trained = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# pre-trained data to classify smile in the face( will do it later)
# smile_data_trained = cv2.CascadeClassifier('haarcascade_smile.xml')
# Asking user for a input
required = input("Select the type of face detection you want.\n1.)Enter 1 for Image.\n2.)Via Webcam.\n3.)Via Video.\n")

# for detecting via images
if required == '1':
    # reading the image to be detected
    img_name = input("Enter the name of the image file with extension\n")
    img = cv2.imread(img_name)
    # converting to gray scale where BGR is RGB
    gray_scaled_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # detect different sizes of image
    face_structures = face_data_frontal_trained.detectMultiScale(gray_scaled_img)
    # print(face_structures)
    # draw rectangle around the gray scaled image based on face structure for multiple faces
    for (x, y, w, h) in face_structures:
     cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
    # show the output window
    cv2.imshow('Face Detector Image', img)
    # waiting for every millisecond
    cv2.waitKey()

# for detecting via Webcam
elif required == '2':
    # Reading from a default Web cam
    webcam_detect = cv2.VideoCapture(0)
    while True:
        # Read current frame
        successful_read_frame, frame = webcam_detect.read()
        # Converting the frame to gray scale
        gray_scaled_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        face_structures = face_data_frontal_trained.detectMultiScale(gray_scaled_frame)
        # draw rectangle around the gray scaled frame based on face structure
        for (x, y, w, h) in face_structures:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        # opens a window with Webcam and detect face
        cv2.imshow('Face Detector Webcam', frame)
        # automatically presses a key every 1 ms
        cv2.waitKey(1)

# for detecting via video
elif required == '3':
    WindowName = "Face Detection Video"
    vid_name = input("Enter name of the video file with extension\n")
    # Reading from the file
    video_detect = cv2.VideoCapture(vid_name)
    # Using ffpyplayer to play the video
    player = MediaPlayer(vid_name)
    while True:
        # Read current frame in the video
        successful_read_frame, vframe = video_detect.read()
        audio_frame, val = player.get_frame()
        # Converting the frame in the video to gray scale
        gray_scaled_frame = cv2.cvtColor(vframe, cv2.COLOR_BGR2GRAY)

        face_structures = face_data_frontal_trained.detectMultiScale(gray_scaled_frame)
        # draw rectangle around the gray scaled frame from the video based on face structure
        for (x, y, w, h) in face_structures:
            cv2.rectangle(vframe, (x, y), (x + w, y + h), (100, 172, 0), 2)
        cv2.namedWindow(WindowName, cv2.WINDOW_NORMAL)
        # opens a window from the video and detect face
        cv2.imshow(WindowName, vframe)
        # automatically presses a key every 1 ms
        key = cv2.waitKey(7)
        if key == 113:
            break
    video_detect.release()
else:
    print("Invalid Input")
    exit(0)

print("End of execution")