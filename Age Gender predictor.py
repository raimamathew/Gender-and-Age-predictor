
# coding: utf-8

# In[1]:


import cv2
import numpy as np
import pafy


cap = cv2.VideoCapture('WhatsApp Video 2018-12-17 at 12.56.55 PM.mp4')
#cap = cv2.imread('rahul.jpg',0)
print(cap)
cap.set(3, 480)  # set width of the frame
cap.set(4, 500)  # set height of the frame
MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)
age_list = ['(0, 2)', '(4, 6)', '(8, 12)', '(15, 20)',
            '(25, 32)', '(38, 43)', '(48, 53)', '(60, 100)']
gender_list = ['Male', 'Female']


def load_caffe_models():

    age_net = cv2.dnn.readNetFromCaffe('deploy_age.prototxt', 'age_net.caffemodel')


    gender_net = cv2.dnn.readNetFromCaffe('deploy_gender.prototxt', 'gender_net.caffemodel')
    return(age_net, gender_net)


def video_detector(age_net, gender_net):
    font = cv2.FONT_HERSHEY_SIMPLEX


# In[2]:


while(True):
    ret, frame = cap.read()

    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    #print (frame.shape[2])
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 5)
    if(len(faces) > 0):
        print("Found {} faces".format(str(len(faces))))
    for (x, y, w, h)in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 255, 0), 2)
# Get Face
        face_img = frame[y:y+h, h:h+w].copy()
        blob = cv2.dnn.blobFromframe( face_img, 1, (227, 227), MODEL_MEAN_VALUES, swapRB=False)
# Predict Gender
        gender_net.setInput(blob)
        gender_preds = gender_net.forward()
        gender = gender_list[gender_preds[0].argmax()]
        print("Gender : " + gender)
    # Predict Age
        age_net.setInput(blob)
        age_preds = age_net.forward()
        age = age_list[age_preds[0].argmax()]
        print("Age Range: " + age)
        overlay_text = "%s %s" % (gender, age)
        cv2.putText(frame, overlay_text, (x, y), font,1, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.imshow('frame', frame)
# 0xFF is a hexadecimal constant which is 11111111 in binary.
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    
if __name__ == "__main__":
    age_net, gender_net = load_caffe_models()
    video_detector(age_net, gender_net)

