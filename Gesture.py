import cv2
import mediapipe as mp
import pyautogui
from mediapipe.framework.formats import landmark_pb2 as mediapipe_landmarks 
import numpy as np


webcam = cv2.VideoCapture(0)
user_hand = mp.solutions.hands.Hands()
x1, y1, x2, y2 = 0, 0, 0, 0
drawing_utils = mp.solutions.drawing_utils
while True:
    _, image = webcam.read()
    image = cv2.flip(image, 1)
    frame_height, frame_width, _ = image.shape
    rgb_img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    output = user_hand.process(rgb_img)

    hands = output.multi_hand_landmarks
    if hands:
        for hand in hands:
            drawing_utils.draw_landmarks(image, hand, mp.solutions.hands.HAND_CONNECTIONS)
            landmarks = hand.landmark
            for id, landmark in enumerate(landmarks):
                x = int(landmark.x * frame_width)
                y = int(landmark.y * frame_height)

                if id == 8: # Point finger
                    cv2.circle(img = image, center = (x,y), radius = 8, color = (0, 255, 255), thickness = 3)
                    x1, y1 = x, y
                if id == 4: # Thumb finger
                    cv2.circle(img = image, center =(x,y), radius = 8, color = (0, 0, 255), thickness = 3)
                    x2, y2 = x, y
                dist = (((x2-x1)**2 + (y2-y1)**2)**0.5) //4
                cv2.line(image, (x1,y1), (x2,y2), (0, 255, 0), 3)
        if dist > 30:
            pyautogui.press("volumeup")
        else:
            pyautogui.press("volumedown")

    cv2.imshow("Using gesture to adjust volume", image)
    cv2.moveWindow("Using gesture to adjust volume", 500, 400)  
    key = cv2.waitKey(10)
    if key == 27:
        break

webcam.release()
cv2.destroyAllWindows()
