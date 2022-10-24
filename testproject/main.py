import cv2 as cv2
import mediapipe as mp
import numpy as np

if __name__ == "__main__":
    mp_drawing = mp.solutions.drawing_utils
    mp_pose = mp.solutions.pose
    cap = cv2.VideoCapture(0)
    # VIDEO FEED
    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        while cap.isOpened():
            ret, frame = cap.read()
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image.flags.writeable = False
            results = pose.process(image)
            image.flags.writeable = True
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            differenceHipsList = []
            differenceShouldersList = []
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

            except:
                pass

            differenceHipsAverage = np.mean(differenceHipsList)
            differenceShouldersAverage = np.mean(differenceShouldersList)



            print("The average hip fluctuation is ", differenceHipsAverage) #better if smaller.
            print("The average shoulder fluctuation is ", differenceShouldersAverage) #better if bigger. But if subject is stuck on one side, the average is big even thought the gait cycle is bad

            #To solve shoulder fluctuation problem with being stuck on one side, we should be able to detect in which phase of the gait cycle the subject is in each frame.
            #In order to determine phase of gait cycle, we have to no the neutral position (standing neutral position) and the corresponding bones positions.
            #From there, we should be able to detect in which phase the subject is in each frame.

            mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                      mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=2),
                                      mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2)
                                      )
            cv2.imshow('Mediapipe Feed', image)

            if cv2.waitKey(10) & 0xFF == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()
