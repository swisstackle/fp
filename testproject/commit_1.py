#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import cv2 as cv2
import mediapipe as mp
import numpy as np
import pandas as pd
import time

if __name__ == "__main__":
    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles
    mp_pose = mp.solutions.pose
    cap = cv2.VideoCapture('naudi_without_outro.mp4')
    # used to record the time when we processed last frame
    prev_frame_time = 0
 
    # used to record the time at which we processed current frame
    new_frame_time = 0
    # VIDEO FEED
    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        slope_shoulder_l,slope_hip_l,rate_slope_shoulder_l,rate_slope_shoulder_length_l=[],[],[],[]

        differenceHipsList = []
        differenceShouldersList = []
        leftshoulderList = []
        rightshoulderList = []
        lefthipList = []
        righthipList = []
        framecounter = 0
        while cap.isOpened():

            ret, frame = cap.read()
            if not ret:
                break
            if frame is not None:
                image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                image.flags.writeable = False
                results = pose.process(image)
                image.flags.writeable = True
                image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            

                font = cv2.FONT_HERSHEY_SIMPLEX
                new_frame_time = time.time()
            # Calculating the fps
 
            # fps will be number of frame processed in given time frame
            # since their will be most of time error of 0.001 second
            # we will be subtracting it to get more accurate result
                fps = 1 / (new_frame_time - prev_frame_time)
                prev_frame_time = new_frame_time

            # converting the fps into integer
                fps = int(fps)

            # converting the fps to string so that we can display it on frame
            # by using putText function
                fps_1 = str(fps)

            # putting the FPS count on the frame
                cv2.putText(image, fps_1, (7, 70), font, 3, (100, 255, 0), 3, cv2.LINE_AA)



                try:
                    landmarks = results.pose_landmarks.landmark
                    leftshoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value]
                    rightshoulder = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value]
                    differenceShoulders = np.absolute(rightshoulder.z - leftshoulder.z)
                    righthip = landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value]
                    lefthip = landmarks[mp_pose.PoseLandmark.LEFT_HIP.value]
                    differenceHips = np.absolute(righthip.z - lefthip.z)
                    differenceHipsList.append(differenceHips)
                    differenceShouldersList.append(differenceShoulders)
                    leftshoulderList.append([leftshoulder.x, leftshoulder.y])
                    rightshoulderList.append([rightshoulder.x, rightshoulder.y])
                    lefthipList.append([lefthip.x, lefthip.y])
                    righthipList.append([righthip.x, righthip.y])

                except:
                    pass
            

                differenceHipsAverage = np.mean(differenceHipsList)
                differenceShouldersAverage = np.mean(differenceShouldersList)

                
                slope_shoulder = np.arctan((leftshoulderList[framecounter][1] - rightshoulderList[framecounter][1]) / (leftshoulderList[framecounter][0] - rightshoulderList[framecounter][0]))
                slope_hip = np.arctan((lefthipList[framecounter][1] - righthipList[framecounter][1]) / (lefthipList[framecounter][0] - righthipList[framecounter][0]))

                rate_slope_shoulder = slope_shoulder * fps

                rate_slope_shoulder_length = differenceShouldersAverage * fps
                framecounter += 1
                
            
            
            #To solve shoulder fluctuation problem with being stuck on one side, we should be able to detect in which phase of the gait cycle the subject is in each frame.
            #In order to determine phase of gait cycle, we have to know the neutral position (standing neutral position) and the corresponding bones positions.
            #From there, we should be able to detect in which phase the subject is in each frame.

                mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                        mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=2),
                                        mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2)
                                        )
                cv2.imshow('Mediapipe Feed', image)

                if cv2.waitKey(10) & 0xFF == ord('q'):
                    break
                slope_shoulder_l.append(slope_shoulder),slope_hip_l.append(slope_hip),rate_slope_shoulder_l.append(rate_slope_shoulder),rate_slope_shoulder_length_l.append(rate_slope_shoulder_length)
    cap.release()
    print(slope_shoulder_l)
    print("Average Slope: ", np.mean(slope_shoulder_l))
    cv2.destroyAllWindows()


# In[ ]:




