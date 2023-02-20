import cv2
import os
from gtts import gTTS
import pygame

from Detector import YOLOV5_Detector

video_path = ""
vid = cv2.VideoCapture(0) # 0: For built-in webcam,  1: For external webcam
detector = YOLOV5_Detector(weights='yolov5s.pt',
                           img_size=640,
                           confidence_thres=0.5,
                           iou_thresh=0.45,
                           agnostic_nms=True,
                           augment=True)
count = 0
while True:
    ret, frame = vid.read()
    if ret:
        res, lab = detector.Detect(frame)
        res = cv2.resize(res, (400,400))
        if count % 5 == 0:
            if lab:
                tts = gTTS(text=lab, lang='en')
                tts.save("detected_object.mp3")
                pygame.init()
                pygame.mixer.music.load("speech.mp3")
                pygame.mixer.music.play()
                while pygame.mixer.music.get_busy() and not pygame.key.get_pressed()[pygame.K_q]:
                    continue
                pygame.mixer.music.stop()
                # Quit pygame
                pygame.quit()
                # os.system("start detected_object.mp3")

        cv2.imshow("Detection", res)
        count += 1
        key = cv2.waitKey(1)
        if key == ord('q'):
            break

vid.release()
cv2.destroyAllWindows()
