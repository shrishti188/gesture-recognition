import cv2
from cvzone.HandTrackingModule import HandDetector
from cvzone.ClassificationModule import Classifier
import numpy as np
import math
import time

cap = cv2.VideoCapture(0)
detector = HandDetector(maxHands=2)  # Set to detect up to 2 hands
classifier = Classifier(r"D:\Sign-Language-detection\Model\keras_model.h5", r"D:\Sign-Language-detection\Model\labels.txt")

offset = 20
imgSize = 300
counter = 0
folder = r"D:\Sign-Language-detection\data\hello"
labels = ["good", "i love u", "hello"]

while True:
    success, img = cap.read()
    imgOutput = img.copy()
    hands, img = detector.findHands(img)

    if hands:
        for hand in hands:
            x, y, w, h = hand['bbox']

            x_start = max(x - offset, 0)
            y_start = max(y - offset, 0)
            x_end = min(x + w + offset, img.shape[1])
            y_end = min(y + h + offset, img.shape[0])

            imgCrop = img[y_start:y_end, x_start:x_end]

            if imgCrop.size == 0:
                print("Empty cropped image, skipping resize")
                continue

            imgWhite = np.ones((imgSize, imgSize, 3), np.uint8) * 255

            aspectRatio = h / w

            if aspectRatio > 1:
                k = imgSize / h
                wCal = math.ceil(k * w)
                imgResize = cv2.resize(imgCrop, (wCal, imgSize))
                wGap = math.ceil((imgSize - wCal) / 2)
                imgWhite[:, wGap: wCal + wGap] = imgResize

            else:
                k = imgSize / w
                hCal = math.ceil(k * h)
                imgResize = cv2.resize(imgCrop, (imgSize, hCal))
                hGap = math.ceil((imgSize - hCal) / 2)
                imgWhite[hGap: hCal + hGap, :] = imgResize

            # Classification of the cropped and resized hand image
            prediction, index = classifier.getPrediction(imgWhite)
            label = labels[index]
            print(f"Prediction: {label}, Confidence: {prediction[index]}")

            # Calculate the size of the text box
            (text_width, text_height), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)
            cv2.rectangle(imgOutput, (x, y - text_height - 20), (x + text_width + 20, y), (255, 0, 255), cv2.FILLED)
            cv2.putText(imgOutput, label, (x + 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

            cv2.imshow('ImageCrop', imgCrop)
            cv2.imshow('ImageWhite', imgWhite)

            key = cv2.waitKey(1)
            if key == ord("s"):
                counter += 1
                cv2.imwrite(f'{folder}/Image_{time.time()}.jpg', imgWhite)
                print(counter)

    cv2.imshow('Image', imgOutput)
    cv2.waitKey(1)
