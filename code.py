import cv2 as cv2
import mediapipe as mp
import numpy as np
import pandas as pd
import time
import os
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.model_selection import train_test_split
# from tensorflow import keras
# from tensorflow.keras.utils import to_categorical
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import LSTM, Dense, Dropout
# from tensorflow.keras.callbacks import TensorBoard, EarlyStopping
# from tensorflow.keras.optimizers import Adam

from keras.utils import plot_model
from sklearn.metrics import multilabel_confusion_matrix, accuracy_score



mp_holistic = mp.solutions.holistic
mp_pose=mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
data_path_i = r'C:\Users\surya\OneDrive\Desktop\Sem 2 UB\ML_Exp\fp\Training_Videos_Final'
# used to record the time when we processed last frame
prev_frame_time = 0

    # used to record the time at which we processed current frame
new_frame_time = 0
    # VIDEO FEED
d = 0

def get_keypoints(results):
    pose = np.array([[res.x, res.y, res.z, res.visibility] for res in results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(132)

def get_data(x):
    for i in range(len(x)):
        y=os.path.join(data_path_i, str(x[i]))
        #print(y)
        vid=os.listdir(y)
        #print(lab_vid_l)
        #print(lab_vid_l[0])
        for j in range(len(vid)):
            #print(vid)
            cap = cv2.VideoCapture(os.path.join(data_path_i, x[i],str(vid[j])))
            #print(cap)
            # used to record the time when we processed last frame
            prev_frame_time = 0

            # used to record the time at which we processed current frame
            new_frame_time = 0
            # VIDEO FEED
            d=0
            with mp_pose.Pose(model_complexity=0,smooth_landmarks=True,min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:

                differenceHipsList = [];differenceShouldersList = [];
                leftshoulderList = [];rightshoulderList = [];
                lsp=[];rsp=[];lhp=[];rhp=[];lefthipList = [];
                righthipList = [];midpointsholder_l=[];
                midpointhip_l=[]
                spinelength_l=[]
                s_len=[]
                h_len=[]
                ratio_l=[]
                slope_shoulder_l=[]
                slope_hip_l=[]
                rate_slope_shoulder_l=[]
                rate_slope_shoulder_length_l=[]
                vid_data=[]
                while cap.isOpened():
                    #print('a')
                    framecounter = 0

                    #print('a') 
                    #print('f')
                    ret, frame = cap.read()
                    #print('g')
                    if not ret:
                        #print('a')
                        break
                    if frame is not None:
                        #print('b')
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
                            leftshoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value] if mp_pose.PoseLandmark.LEFT_SHOULDER.value else np.zeros(1,3)
                            rightshoulder = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value] if mp_pose.PoseLandmark.RIGHT_SHOULDER.value else np.zeros(1,3)
                            differenceShoulders = np.absolute(rightshoulder.z - leftshoulder.z)
                            righthip = landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value] if mp_pose.PoseLandmark.RIGHT_HIP.value else np.zeros(1,3)
                            lefthip = landmarks[mp_pose.PoseLandmark.LEFT_HIP.value] if mp_pose.PoseLandmark.LEFT_HIP.value else np.zeros(1,3)
                            differenceHips = np.absolute(righthip.z - lefthip.z)
                            differenceHipsList.append(differenceHips)
                            differenceShouldersList.append(differenceShoulders)
                            leftshoulderList.append([leftshoulder.x, leftshoulder.y])
                            rightshoulderList.append([rightshoulder.x, rightshoulder.y])
                            lefthipList.append([lefthip.x, lefthip.y])
                            righthipList.append([righthip.x, righthip.y])
                            differenceHipsAverage = np.mean(differenceHipsList)
                            differenceShouldersAverage = np.mean(differenceShouldersList)


                            slope_shoulder = np.arctan((leftshoulderList[framecounter][1] - rightshoulderList[framecounter][1]) / (leftshoulderList[framecounter][0] - rightshoulderList[framecounter][0]))
                            slope_hip = np.arctan((lefthipList[framecounter][1] - righthipList[framecounter][1]) / (lefthipList[framecounter][0] - righthipList[framecounter][0]))

                            rate_slope_shoulder = slope_shoulder * fps

                            rate_slope_shoulder_length = differenceShouldersAverage * fps
                            midpointsholder_p_x = (leftshoulder.x+rightshoulder.x)/2
                            midpointsholder_p_y = (leftshoulder.y+rightshoulder.y)/2
                            leftsholder_poin=np.array([leftshoulder.x,leftshoulder.y])
                            rightsholder_poin=np.array([rightshoulder.x,rightshoulder.y])
                            lefthip_poin=np.array([lefthip.x,lefthip.y])
                            righthip_poin=np.array([righthip.x,righthip.y])
                            midpointsholder_poin=np.array([midpointsholder_p_x,midpointsholder_p_y])
                            midpointhip_p_x = (lefthip.x+righthip.x)/2
                            midpointhip_p_y = (lefthip.y+righthip.y)/2
                            midpointhip_poin=np.array([midpointhip_p_x,midpointhip_p_y])

                            s_len_v=np.linalg.norm(leftsholder_poin-rightsholder_poin)
                            h_len_v=np.linalg.norm(lefthip_poin-righthip_poin)
                            ratio_v=s_len_v/h_len_v
                            spinelength_v=np.linalg.norm(midpointsholder_poin-midpointhip_poin)
                            framecounter += 1



                        #To solve shoulder fluctuation problem with being stuck on one side, we should be able to detect in which phase of the gait cycle the subject is in each frame.
                        #In order to determine phase of gait cycle, we have to know the neutral position (standing neutral position) and the corresponding bones positions.
                        #From there, we should be able to detect in which phase the subject is in each frame.

                            mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                                    mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=2),
                                                    mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2)
                                                    )


                            cv2.imshow('Mediapipe Feed', image)
                            #image_l.append(image)
                            #print(image.shape)

                            lsp.append([leftshoulder.x,leftshoulder.y]),rsp.append([rightshoulder.x,rightshoulder.y]),lhp.append([lefthip.x,lefthip.y]),rhp.append([righthip.x,righthip.y]),ratio_l.append(ratio_v),spinelength_l.append(spinelength_v),midpointsholder_l.append([midpointsholder_p_x,midpointsholder_p_y]),midpointhip_l.append([midpointhip_p_x,midpointhip_p_y]),slope_shoulder_l.append(slope_shoulder),slope_hip_l.append(slope_hip),rate_slope_shoulder_l.append(rate_slope_shoulder),rate_slope_shoulder_length_l.append(rate_slope_shoulder_length),s_len.append(s_len_v),h_len.append(h_len_v)
                            print('test')


                        except:
                            pass
                        d=d+1
                        if cv2.waitKey(10) & 0xFF == ord('q'):
                            break
                data={'difference_shoulder': differenceShouldersList,
                          'difference_hip': differenceHipsList,
                          'Spine_Length':spinelength_l,
                          'Shoulder_Length':s_len,
                          'Slope_shoulder':slope_shoulder_l,
                          'Slope_hip':slope_hip_l}
                data=pd.DataFrame(data)
                data_n=np.array(data)
                differenceHipsList = [];differenceShouldersList = [];
                leftshoulderList = [];rightshoulderList = [];
                lsp=[];rsp=[];lhp=[];rhp=[];lefthipList = [];
                righthipList = [];midpointsholder_l=[];
                midpointhip_l=[]
                spinelength_l=[]
                s_len=[]
                h_len=[]
                ratio_l=[]
                slope_shoulder_l=[]
                slope_hip_l=[]
                rate_slope_shoulder_l=[]
                rate_slope_shoulder_length_l=[]
                vid_data=[]

                print('test')
                npy_path_m = os.path.join(DATA_PATH, x[i])
                print(npy_path_m)
                npy_path = os.path.join(npy_path_m,str(j),str(j))
                print(npy_path)
                print(data_n)
                np.save(npy_path,data_n,allow_pickle=True)
                vid_data.append([npy_path,data_n])

             

                        


 
            cap.release()
            #print(midpointhip_l)
            cv2.destroyAllWindows()
    return vid_data





# out[np.argmax(res, axis=1)]
#
#
# out[np.argmax(y_test[2])]
#
#
# out[np.argmax(res[2])]



def prob_viz(res, out, input_frame, colors):
    output_frame = input_frame.copy()
    for num, prob in enumerate(res):
        cv2.rectangle(output_frame, (0,60+num*40), (int(prob*100), 90+num*40), colors[num], -1)
        cv2.putText(output_frame, out[num], (0, 85+num*40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2, cv2.LINE_AA)
        
    return output_frame


# with mp_pose.Pose(model_complexity=0,smooth_landmarks=True,min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
#
#     differenceHipsList = [];differenceShouldersList = [];
#     leftshoulderList = [];rightshoulderList = [];
#     lsp=[];rsp=[];lhp=[];rhp=[];lefthipList = [];
#     righthipList = [];midpointsholder_l=[];
#     midpointhip_l=[]
#     spinelength_l=[]
#     s_len=[]
#     h_len=[]
#     ratio_l=[]
#     slope_shoulder_l=[]
#     slope_hip_l=[]
#     rate_slope_shoulder_l=[]
#     rate_slope_shoulder_length_l=[]
#     vid_data=[]
#     while cap.isOpened():
#         #print('a')
#         framecounter = 0
#
#         #print('a')
#         #print('f')
#         ret, frame = cap.read()
#         #print('g')
#         if not ret:
#             #print('a')
#             break
#         if frame is not None:
#             #print('b')
#             image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#
#             image.flags.writeable = False
#             results = pose.process(image)
#             image.flags.writeable = True
#             image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#
#
#             font = cv2.FONT_HERSHEY_SIMPLEX
#             new_frame_time = time.time()
#         # Calculating the fps
#
#         # fps will be number of frame processed in given time frame
#         # since their will be most of time error of 0.001 second
#         # we will be subtracting it to get more accurate result
#             fps = 1 / (new_frame_time - prev_frame_time)
#             prev_frame_time = new_frame_time
#
#         # converting the fps into integer
#             fps = int(fps)
#
#         # converting the fps to string so that we can display it on frame
#         # by using putText function
#             fps_1 = str(fps)
#
#         # putting the FPS count on the frame
#             cv2.putText(image, fps_1, (7, 70), font, 3, (100, 255, 0), 3, cv2.LINE_AA)
#
#
#             landmarks = results.pose_landmarks.landmark
#             leftshoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value] if mp_pose.PoseLandmark.LEFT_SHOULDER.value else np.zeros(1,3)
#             rightshoulder = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value] if mp_pose.PoseLandmark.RIGHT_SHOULDER.value else np.zeros(1,3)
#             differenceShoulders = np.absolute(rightshoulder.z - leftshoulder.z)
#             righthip = landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value] if mp_pose.PoseLandmark.RIGHT_HIP.value else np.zeros(1,3)
#             lefthip = landmarks[mp_pose.PoseLandmark.LEFT_HIP.value] if mp_pose.PoseLandmark.LEFT_HIP.value else np.zeros(1,3)
#             differenceHips = np.absolute(righthip.z - lefthip.z)
#             differenceHipsList.append(differenceHips)
#             differenceShouldersList.append(differenceShoulders)
#             leftshoulderList.append([leftshoulder.x, leftshoulder.y])
#             rightshoulderList.append([rightshoulder.x, rightshoulder.y])
#             lefthipList.append([lefthip.x, lefthip.y])
#             righthipList.append([righthip.x, righthip.y])
#             differenceHipsAverage = np.mean(differenceHipsList)
#             differenceShouldersAverage = np.mean(differenceShouldersList)
#
#
#             slope_shoulder = np.arctan((leftshoulderList[framecounter][1] - rightshoulderList[framecounter][1]) / (leftshoulderList[framecounter][0] - rightshoulderList[framecounter][0]))
#             slope_hip = np.arctan((lefthipList[framecounter][1] - righthipList[framecounter][1]) / (lefthipList[framecounter][0] - righthipList[framecounter][0]))
#
#             rate_slope_shoulder = slope_shoulder * fps
#
#             rate_slope_shoulder_length = differenceShouldersAverage * fps
#             midpointsholder_p_x = (leftshoulder.x+rightshoulder.x)/2
#             midpointsholder_p_y = (leftshoulder.y+rightshoulder.y)/2
#             leftsholder_poin=np.array([leftshoulder.x,leftshoulder.y])
#             rightsholder_poin=np.array([rightshoulder.x,rightshoulder.y])
#             lefthip_poin=np.array([lefthip.x,lefthip.y])
#             righthip_poin=np.array([righthip.x,righthip.y])
#             midpointsholder_poin=np.array([midpointsholder_p_x,midpointsholder_p_y])
#             midpointhip_p_x = (lefthip.x+righthip.x)/2
#             midpointhip_p_y = (lefthip.y+righthip.y)/2
#             midpointhip_poin=np.array([midpointhip_p_x,midpointhip_p_y])
#
#             s_len_v=np.linalg.norm(leftsholder_poin-rightsholder_poin)
#             h_len_v=np.linalg.norm(lefthip_poin-righthip_poin)
#             ratio_v=s_len_v/h_len_v
#             spinelength_v=np.linalg.norm(midpointsholder_poin-midpointhip_poin)
#             framecounter += 1
#
#
#
#         #To solve shoulder fluctuation problem with being stuck on one side, we should be able to detect in which phase of the gait cycle the subject is in each frame.
#         #In order to determine phase of gait cycle, we have to know the neutral position (standing neutral position) and the corresponding bones positions.
#         #From there, we should be able to detect in which phase the subject is in each frame.
#
#             mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
#                                     mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=2),
#                                     mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2)
#                                     )
#
#
#             #cv2.imshow('Mediapipe Feed', image)
#             #image_l.append(image)
#             #print(image.shape)
#
#             lsp.append([leftshoulder.x,leftshoulder.y]),rsp.append([rightshoulder.x,rightshoulder.y]),lhp.append([lefthip.x,lefthip.y]),rhp.append([righthip.x,righthip.y]),ratio_l.append(ratio_v),spinelength_l.append(spinelength_v),midpointsholder_l.append([midpointsholder_p_x,midpointsholder_p_y]),midpointhip_l.append([midpointhip_p_x,midpointhip_p_y]),slope_shoulder_l.append(slope_shoulder),slope_hip_l.append(slope_hip),rate_slope_shoulder_l.append(rate_slope_shoulder),rate_slope_shoulder_length_l.append(rate_slope_shoulder_length),s_len.append(s_len_v),h_len.append(h_len_v)
#             #print('test')
#
#             data={'difference_shoulder': differenceShouldersList,
#                           'difference_hip': differenceHipsList,
#                           'Spine_Length':spinelength_l,
#                           'Shoulder_Length':s_len,
#                           'Slope_shoulder':slope_shoulder_l,
#                           'Slope_hip':slope_hip_l}
#             data=pd.DataFrame(data)
#             data_n=np.array(data)
#             print(((np.expand_dims(data_n,axis=0))).shape)
#
#             if len(data_n)>30:
#                 res=model.predict(np.expand_dims(data_n,axis=0))[0]
#                 if res[np.argmax(res)] > threshold:
#                     if len(sentence) > 0:
#                         if out[np.argmax(res)] != sentence[-1]:
#                             sentence.append(out[np.argmax(res)])
#
#
#                     else:
#                         sentence.append(out[np.argmax(res)])
#             if len(data_n)>30:
#                 res = model.predict(np.expand_dims(data_n,axis=0))[0]
#                 print(res)
#                 if res[np.argmax(res)] > threshold:
#                     if len(sentence) > 0:
#                         if out[np.argmax(res)] != sentence[-1]:
#                             sentence.append(out[np.argmax(res)])
#                     else:
#                         sentence.append(out[np.argmax(res)])
#
#                 if len(sentence) > 10:
#                     sentence = sentence[-3:]
#
#                 image_1=prob_viz(res,out,image,colors)
#                 cv2.imshow('Open CV Feed',image_1)
#
#
#             if cv2.waitKey(10) & 0xFF == ord('q'):
#                    break
#
#     cap.release()
#     cv2.destroyAllWindows()


if __name__ == "__main__":
    tf.test.is_built_with_cuda()

    out = np.array(['glr', 'grl', 'blr', 'brl', 'und'])

    vid_q = ['lr.mp4', 'rl1.mp4', 'rl2.mp4']

    DATA_PATH = os.path.join(r'C:\Users\surya\OneDrive\Desktop\Sem 2 UB\ML_Exp\fp\Training Videos Final')

    for i in range(len(out)):
        os.makedirs(os.path.join(DATA_PATH, out[i]))
        for j in range(8):
            os.makedirs(os.path.join(DATA_PATH, out[i], str(j)))

    vid_q_1 = os.listdir(DATA_PATH)

    vid_data_list = get_data(vid_q_1)

    vid_1_n = np.load(r"C:\Users\surya\OneDrive\Desktop\Sem 2 UB\ML_Exp\fp\Training Videos Final\blr\0\0.npy",
                      allow_pickle=True)

    vid_1_n = np.array(vid_1_n)

    label_map = {label: num for num, label in enumerate(out)}

    vid_q = np.array(os.listdir(os.path.join(DATA_PATH, str(out[0])))).astype(int)
    seq, labels = [], []
    for i in range(len(out)):
        for sequence in np.array(os.listdir(os.path.join(DATA_PATH, out[i]))).astype(int):
            window = []
            for sq in range(len(os.listdir(os.path.join(DATA_PATH, out[i], str(sequence))))):
                res = np.load(os.path.join(DATA_PATH, out[i], str(sequence), "{}.npy".format(sequence)))
                res = np.array(res)
                res = res[:30][:]
                print(res.shape)
                window.append(res)
            seq.append(window)
            # seq=np.array(seq)
            print(len(seq))
            labels.append(label_map[out[i]])
    a = label_map.items()

    seq = np.array(seq)
    x = seq
    x = np.reshape(x, (len(x), 30, 6))
    print(x.shape)

    y = to_categorical(labels).astype(int)

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, shuffle=True, random_state=27)

    tb_callback = EarlyStopping(monitor='val_loss', patience=30, mode='min', restore_best_weights=True)

    model = Sequential()
    model.add(LSTM(64, return_sequences=True, activation='relu', input_shape=(30, 6)))
    model.add(LSTM(128, return_sequences=True, activation='relu'))
    # model.add(Dropout(0.12))
    # model.add(LSTM(256,return_sequences=True, activation='relu'))  #added layer
    # model.add(LSTM(128, return_sequences=True, activation='relu')) #added layer
    model.add(LSTM(64, return_sequences=False, activation='relu'))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(out.shape[0], activation='softmax'))

    model.summary()

    model.compile(optimizer='RMSProp', loss='categorical_crossentropy', metrics=['accuracy'])

    history = model.fit(x_train, y_train, epochs=1000, callbacks=tb_callback, shuffle=True)

    history1 = model.fit(x_test, y_test, epochs=1000, callbacks=tb_callback, shuffle=True)

    model.save('models/LSTM_model_test7_es_adagrad.h5')  # change name of model to prevent overwrite

    # uncomment to load previous models from the models folder

    model = keras.models.load_model('models/LSTM_model_test7_es_adagrad.h5')

    loss_train = history.history['loss']
    loss_val = history1.history['loss']
    acc = history.history['accuracy']
    val_acc = history1.history['accuracy']

    epochs = range(0, np.shape(loss_train)[0])
    plt.plot(epochs, loss_train, 'g', label='training loss')
    plt.plot(epochs, loss_val, 'b', label='validation loss')
    plt.title('Training and Validation loss')
    plt.xlabel('Epochs')
    plt.ylabel('loss')
    plt.legend()
    plt.show()
    plt.figure()

    plt.plot(epochs, acc, 'b', label='epoch_accuracy')
    plt.plot(epochs, val_acc, 'r', label='val_accuracy')
    plt.title('Training and validation accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('acc')
    plt.legend()
    plt.show()
    plt.figure()

    plot_model(model, to_file='model.png', show_shapes=True, show_layer_names=True)

    yhat = model.predict(x_test)

    ytrue = np.argmax(y_test, axis=1).tolist()
    yhat = np.argmax(yhat, axis=1).tolist()

    multilabel_confusion_matrix(ytrue, yhat)

    accuracy_score(ytrue, yhat)

    #res = model.predict(x_test)

    # threshold = 0.92
    # sentence = []
    # colors = [(245, 117, 16), (117, 245, 16), (16, 117, 245), (50, 117, 30), (50, 117, 30)]
    # image_1 = []
    # cap = cv2.VideoCapture('vid_f_1.mp4')

