import cv2
import time
import numpy as np
import PoseModule as pm

cap = cv2.VideoCapture(0)
detector = pm.poseDetector()
count = 0
pos = None
per = 0
bar = 0

while True:
    success, img = cap.read()
    img = cv2.flip(img, 1)
    img = detector.findPose(img, False)
    lmList = detector.findPosition(img, False)

    if len(lmList) != 0:
        # Right Arm
        angle = detector.findAngle(img, 12, 14, 16)

        # Percentage for Progress Bar
        per = np.interp(angle, (25, 160), (100, 0))
        bar = np.interp(angle, (25, 160), (650, 100))

        # Feedback for Position
        if angle > 160:
            pos = "Fix Form"
        elif angle > 140:
            pos = "Up"
        elif angle < 35:
            if pos == "Up":
                count += 1
                pos = "Down"
                time.sleep(0.2)
            else:
                pos = "Down"

    cv2.rectangle(img, (100, 100), (150, 650), (0, 255, 0), 3)
    cv2.rectangle(img, (100, int(bar)), (150, 650), (0, 255, 0), cv2.FILLED)
    cv2.putText(img, f'{int(per)}%', (100, 75), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 0), 3)

    if pos is not None:
        cv2.putText(img, pos, (300, 70), cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 3)

    cv2.putText(img, str(count), (50, 100), cv2.FONT_HERSHEY_PLAIN, 4, (0, 0, 255), 4)

    cv2.imshow("Bicep Curl", img)
    key = cv2.waitKey(1)
    if key == 27:  # Press 'ESC' to exit
        break

cap.release()
cv2.destroyAllWindows()