import cv2
import mediapipe as mp
import matplotlib
matplotlib.use('TkAgg')
from matplotlib import pyplot as plt
import numpy as np
import math
import threading
 
mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands
 
cap = cv2.VideoCapture(0)
studyState = False
focusRate = 0
focus_x = [10, 20, 30, 40, 50, 60]
focus_y = []
shutDown = False
time = [10.0, 20.0, 30.0, 40.0, 50.0, 60.0]

#손가락의 거리 구하는 함수
def dist(x1, y1, x2, y2):
    return math.sqrt(math.pow(x1 - x2,2)) + math.sqrt(math.pow(y1 - y2,2))

open = [False, False, False, False, False]
directions = [[True, False, False, False, False, "STUDY1"],
              [True, True, False, False, False, "STUDY2"]]

#10초 단위로 집중도 체크하는 함수
def focusTime():
    global focus_y
    global focusRate
    focus_y.append(focusRate)
    print(focus_y)
    focusRate = 0

#그래프 그려주는 함수
def drawGraph():
    global focus_x
    global focus_y
    print(focus_x)
    print(focus_y)
    x = np.array(focus_x)
    y = np.array(focus_y)
    plt.plot(x, y)
    plt.show()

def finishcv():
    global shutDown
    shutDown = True

#timer = threading.Timer(30.0, focusTime)
#timer.start()

timer10 = threading.Timer(time[0], focusTime)
timer20 = threading.Timer(time[1], focusTime)
timer30 = threading.Timer(time[2], focusTime)
timer40 = threading.Timer(time[3], focusTime)
timer50 = threading.Timer(time[4], focusTime)
timer60 = threading.Timer(time[5], focusTime)
timerdraw = threading.Timer(61.0, finishcv)
timer10.start() 
timer20.start() 
timer30.start()
timer40.start() 
timer50.start() 
timer60.start()  
timerdraw.start()
 
with mp_hands.Hands(
    max_num_hands=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5) as hands:
 
    while cap.isOpened():
        success, image = cap.read()
        if not success:
            continue
        image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
 
        results = hands.process(image)
 
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
 
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                if (dist(hand_landmarks.landmark[4].x, hand_landmarks.landmark[4].y, hand_landmarks.landmark[0].x, hand_landmarks.landmark[0].y) < dist(hand_landmarks.landmark[18].x, hand_landmarks.landmark[18].y, hand_landmarks.landmark[0].x, hand_landmarks.landmark[0].y)):   
                    open[0] = False
                else:

                    open[0] = True

                for i in range(1, 5):
                    if (dist(hand_landmarks.landmark[i*4+4].x, hand_landmarks.landmark[i*4+4].y, hand_landmarks.landmark[0].x, hand_landmarks.landmark[0].y) < dist(hand_landmarks.landmark[i*4+2].x, hand_landmarks.landmark[i*4+2].y, hand_landmarks.landmark[0].x, hand_landmarks.landmark[0].y)):
                        open[i] = False
                    else:
                        open[i] = True

                #손모양 확인하는 부분
                for i in range(0, len(directions)):
                    flag = True
                    for j in range(0, 5):
                        if (directions[i][j] != open[j]):
                            flag = False
                    if (flag == True):
                        cv2.putText(image, text='%s' % (directions[i][5]), org=(10, 50), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1,color=255,thickness=3)
                        if (directions[i][5] == "STUDY1"):
                            print("study")
                            studyState = True
                            focusRate += 1
                            #print(focusRate)
                            print(studyState)
                        elif (directions[i][5] == "STUDY2"):
                            print("study")
                            studyState = True
                            focusRate += 1
                            #print(focusRate)
                            print(studyState)
                        else:
                            studyState = False
                            print(studyState)
                            focusRate = 0
                            
                cv2.putText(
                    image, text='f1=%d f2=%d f3=%d f4=%d f5 = %d' % (open[0], open[1], open[2], open[3], open[4]), org=(10, 30),
                    fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1,
                    color=255, thickness=3)
 
                mp_drawing.draw_landmarks(
                    image, hand_landmarks, mp_hands.HAND_CONNECTIONS)
 
        cv2.imshow('MediaPipe', image) 
        
        if cv2.waitKey(1) == ord('q'):
            break
        if shutDown == True:
            drawGraph()
            break
 
cap.release()