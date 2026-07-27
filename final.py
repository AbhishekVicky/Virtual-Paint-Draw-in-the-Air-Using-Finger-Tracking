import cv2
import mediapipe as mp
import numpy as np
import time
from collections import deque


# Webcam
cap = cv2.VideoCapture(1)
cap.set(3,1280)
cap.set(4,720)

# Mediapipe
mpHands = mp.solutions.hands
hands = mpHands.Hands(max_num_hands=1,
                      min_detection_confidence=0.7,
                      min_tracking_confidence=0.7)

mpDraw = mp.solutions.drawing_utils

# Canvas
canvas = None

# Stroke smoothing buffer
points = deque(maxlen=15)

# UI Settings
color = (255,0,0)
brush = 6
header_h = 100

# FPS
pTime = 0

def fingers_up(lmList):
    fingers = []

    # Thumb
    if lmList[4][0] > lmList[3][0]:
        fingers.append(1)
    else:
        fingers.append(0)

    # Other fingers
    tips = [8,12,16,20]
    for tip in tips:
        if lmList[tip][1] < lmList[tip-2][1]:
            fingers.append(1)
        else:
            fingers.append(0)

    return fingers

while True:
    ret, frame = cap.read()
    if not ret:
        continue

    frame = cv2.flip(frame,1)

    if canvas is None:
        canvas = np.zeros_like(frame)

    # Header UI
    cv2.rectangle(frame,(0,0),(1280,header_h),(40,40,40),-1)
    cv2.rectangle(frame,(40,20),(160,80),(255,0,0),-1)
    cv2.rectangle(frame,(200,20),(320,80),(0,255,0),-1)
    cv2.rectangle(frame,(360,20),(480,80),(0,0,255),-1)
    cv2.rectangle(frame,(520,20),(640,80),(0,0,0),-1)
    cv2.rectangle(frame,(700,20),(860,80),(200,200,200),-1)

    cv2.putText(frame,"CLEAR",(710,70),
                cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,0),3)

    # Hand detect
    rgb = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
    res = hands.process(rgb)

    if res.multi_hand_landmarks:
        for handLms in res.multi_hand_landmarks:

            mpDraw.draw_landmarks(frame,handLms,
                                  mpHands.HAND_CONNECTIONS)

            h,w,c = frame.shape
            lmList = []

            for lm in handLms.landmark:
                lmList.append([int(lm.x*w),int(lm.y*h)])

            cx,cy = lmList[8]

            cv2.circle(frame,(cx,cy),12,(0,255,0),-1)

            finger_state = fingers_up(lmList)

            # ---------- Selection Mode (2 fingers up) ----------
            if finger_state[1] == 1 and finger_state[2] == 1:

                points.clear()

                if cy < header_h:
                    if 40 < cx < 160:
                        color = (255,0,0)
                    elif 200 < cx < 320:
                        color = (0,255,0)
                    elif 360 < cx < 480:
                        color = (0,0,255)
                    elif 520 < cx < 640:
                        color = (0,0,0)
                    elif 700 < cx < 860:
                        canvas = np.zeros_like(frame)

            # ---------- Drawing Mode (Only index up) ----------
            elif finger_state[1] == 1 and finger_state[2] == 0:

                points.appendleft((cx,cy))

                for i in range(1,len(points)):
                    if points[i-1] is None or points[i] is None:
                        continue
                    cv2.line(canvas,points[i-1],points[i],
                             color,brush)

            else:
                points.clear()

    frame = cv2.add(frame,canvas)

    # FPS
    cTime = time.time()
    fps = 1/(cTime-pTime)
    pTime = cTime

    cv2.putText(frame,f'FPS:{int(fps)}',
                (20,700),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.5,(0,255,255),3)

    cv2.imshow("Virtual Paint",frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()