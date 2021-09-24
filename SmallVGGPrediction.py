from keras.models import load_model
import numpy as np
import cv2
import os


folder = "dataset"  # type: str
CATEGORIES = [ f.path for f in os.scandir(folder) if f.is_dir() ]
for i in range(len(CATEGORIES)):
    CATEGORIES[i] = CATEGORIES[i].replace('dataset\\', '')

model = load_model("D:\MSCS\Research\image processing\mnist\mnist.model")
cap = cv2.VideoCapture('D:\MSCS\Research\image processing\mnist\clip.mp4')


def prepare(img_array):
    IMG_SIZE = 28
    img_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
    img_array = np.array(img_array, dtype="float") / 255.0
    img_array = img_array.reshape(-1, IMG_SIZE, IMG_SIZE, 3)
    return img_array


def getDigitName(prediction):
    maximum = 0
    index = 0
    for i in range(len(CATEGORIES)):
        if (prediction[0][i]) > maximum:
            index = i
            maximum = prediction[0][i]
    text = "{}".format(CATEGORIES[index])
    return(text)


out = cv2.VideoWriter('mnistvideo.avi', cv2.VideoWriter_fourcc(*'DIVX'), 20, (224, 224))
while(cap.isOpened):
    _, frame = cap.read()
    preds = model.predict(prepare(frame))
    frame = cv2.resize(frame, (224, 224))
    cv2.putText(frame, getDigitName(preds), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    out.write(frame)
    cv2.imshow('window-name', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
out.release()
cv2.destroyAllWindows()
cv2.waitKey(0)