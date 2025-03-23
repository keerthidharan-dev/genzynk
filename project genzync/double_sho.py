import cv2
import time
import numpy as np
import PoseModule as pm

cap = cv2.VideoCapture(0)
detector = pm.poseDetector()
count = 0
count2 = 0
pos = None
pos2 = None
per = 0
per2 = 0
bar = 0
bar2 = 0
direction1 = 0
direction2 = 0

while True:
    success, img = cap.read()
    img = cv2.flip(img, 1)
    img = detector.findPose(img, False)
    lmList = detector.findPosition(img, False)

    if len(lmList) != 0:
        # Right Arm
        angle = detector.findAngle(img, 12, 14, 16)
        # Left Arm
        angle2 = detector.findAngle(img, 11, 13, 15)

        # Percentage for Progress Bar (Right)
        per = np.interp(angle, (75, 160), (0, 100))
        bar = np.interp(angle, (75, 160), (650, 100))

        # Percentage for Progress Bar (Left)
        per2 = np.interp(angle2, (75, 160), (0, 100))
        bar2 = np.interp(angle2, (75, 160), (650, 100))

        # Feedback for Right Arm
        if angle > 160:
            pos = "Fix Form"
        elif per == 0:
            pos = "Up"
            direction1 = 0
        elif per == 100:
            pos = "Down"
            if direction1 == 0:
                count += 1
                direction1 = 1
                time.sleep(0.2)

        # Feedback for Left Arm
        if angle2 > 160:
            pos2 = "Fix Form"
        elif per2 == 0:
            pos2 = "Up"
            direction2 = 0
        elif per2 == 100:
            pos2 = "Down"
            if direction2 == 0:
                count2 += 1
                direction2 = 1
                time.sleep(0.2)

    # Right Arm Bar
    cv2.rectangle(img, (100, 100), (150, 650), (0, 255, 0), 3)
    cv2.rectangle(img, (100, int(bar)), (150, 650), (0, 255, 0), cv2.FILLED)
    cv2.putText(img, f'{int(per)}%', (100, 75), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 0), 3)
    cv2.putText(img, str(count), (50, 100), cv2.FONT_HERSHEY_PLAIN, 4, (0, 0, 255), 4)

    # Left Arm Bar
    cv2.rectangle(img, (400, 100), (450, 650), (0, 255, 0), 3)
    cv2.rectangle(img, (400, int(bar2)), (450, 650), (0, 255, 0), cv2.FILLED)
    cv2.putText(img, f'{int(per2)}%', (400, 75), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 0), 3)
    cv2.putText(img, str(count2), (350, 100), cv2.FONT_HERSHEY_PLAIN, 4, (0, 0, 255), 4)

    # Feedback Text
    if pos is not None:
        cv2.putText(img, pos, (500, 70), cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 3)
    if pos2 is not None:
        cv2.putText(img, pos2, (500, 120), cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 3)

    cv2.imshow("Double Hand Shoulder Press", img)
    key = cv2.waitKey(1)
    if key == 27:  # Press 'ESC' to exit
        break

cap.release()
cv2.destroyAllWindows()
