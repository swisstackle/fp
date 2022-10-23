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

            try:
                landmarks = results.pose_landmarks.landmark
                leftshoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value]
                rightshoulder = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value]
                difference = rightshoulder.z - leftshoulder.z
                if(np.absolute(difference) > 0.4):
                    print(" Shoulder Tilted")
                else:
                    print("Shoulder Not Tilted")

               #print('{0:.16f}'.format(rightshoulder.x - leftshoulder.x))

            except:
                pass

            mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                      mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=2),
                                      mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2)
                                      )
            cv2.imshow('Mediapipe Feed', image)

            if cv2.waitKey(10) & 0xFF == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()
