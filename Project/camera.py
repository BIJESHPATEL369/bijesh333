import cv2
import numpy as np
import math

faceProto="opencv_face_detector.pbtxt"
faceModel="opencv_face_detector_uint8.pb"
ageProto="age_deploy.prototxt"
ageModel="age_net.caffemodel"
genderProto="gender_deploy.prototxt"
genderModel="gender_net.caffemodel"

facec = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
font = cv2.FONT_HERSHEY_SIMPLEX

faceNet=cv2.dnn.readNet(faceModel,faceProto)
ageNet=cv2.dnn.readNet(ageModel,ageProto)
genderNet=cv2.dnn.readNet(genderModel,genderProto)

class VideoCamera(object):
    def __init__(self):
        #self.video = cv2.VideoCapture("C:/Users/bijes/Downloads/test_Coursera/Project/videos/presidential_debate.mp4")
        self.video = cv2.VideoCapture(0)

    def __del__(self):
        self.video.release()

    # returns camera frames along with bounding boxes and predictions
    def get_frame(self):
        hasFrame,frame = self.video.read()

        gray_fr = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = facec.detectMultiScale(gray_fr, 1.3, 5)

        frameOpencvDnn=frame.copy()
        frameHeight=frameOpencvDnn.shape[0]
        frameWidth=frameOpencvDnn.shape[1]
        blob=cv2.dnn.blobFromImage(frameOpencvDnn, 1.0, (300, 300), [104, 117, 123], True, False)

        net = faceNet
        conf_threshold = 0.7

        net.setInput(blob)
        detections=net.forward()
        faceBoxes=[]
        for i in range(detections.shape[2]):
            confidence=detections[0,0,i,2]
            if confidence>conf_threshold:
                x1=int(detections[0,0,i,3]*frameWidth)
                y1=int(detections[0,0,i,4]*frameHeight)
                x2=int(detections[0,0,i,5]*frameWidth)
                y2=int(detections[0,0,i,6]*frameHeight)
                faceBoxes.append([x1,y1,x2,y2])
                cv2.rectangle(frameOpencvDnn, (x1,y1), (x2,y2), (0,255,0), int(round(frameHeight/150)), 8)

        resulting = frameOpencvDnn
        padding=20
        MODEL_MEAN_VALUES=(78.4263377603, 87.7689143744, 114.895847746)
        ageList=['(0-2)', '(4-6)', '(8-12)', '(15-20)', '(25-32)', '(38-43)', '(48-53)', '(60-100)']
        genderList=['Male','Female']
        for faceBox in faceBoxes:
            face=frame[max(0,faceBox[1]-padding):
                    min(faceBox[3]+padding,frame.shape[0]-1),max(0,faceBox[0]-padding)
                    :min(faceBox[2]+padding, frame.shape[1]-1)]

            blob=cv2.dnn.blobFromImage(face, 1.0, (227,227), MODEL_MEAN_VALUES, swapRB=False)
            genderNet.setInput(blob)
            genderPreds=genderNet.forward()
            gender=genderList[genderPreds[0].argmax()]
            print(f'Gender: {gender}')

            ageNet.setInput(blob)
            agePreds=ageNet.forward()
            age=ageList[agePreds[0].argmax()]
            print(f'Age: {age[1:-1]} years')
        
            for (x, y, w, h) in faces:
                cv2.putText(frame, f'{gender}, {age}', (x, y), font, 1, (255, 255, 0), 2)
                cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)

        _, jpeg = cv2.imencode('.jpg', frame)
        return jpeg.tobytes()
