import cv2
import mediapipe as mp
import numpy as np
import csv
from datetime import datetime

# Initialize Mediapipe pose detection
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()
mp_drawing = mp.solutions.drawing_utils

# Custom drawing styles
landmark_color = (0, 0, 255)  # Red color for points
connection_color = (255, 255, 255)  # White color for lines
landmark_radius = 5  # Radius of the landmark points

# CSV file setup
csv_filename = "shoulder_press_data.csv"
with open(csv_filename, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(["Timestamp", "Shoulder Angle", "Elbow Angle"])  # Column headers

# Function to calculate angle between three points
def calculate_angle(a, b, c):
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)
    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radians * 180.0 / np.pi)
    if angle > 180.0:
        angle = 360 - angle
    return angle

# Load the recorded video file
video_path = 'shoulderpress.mp4'
cap = cv2.VideoCapture(video_path)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    # Convert the frame to RGB
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(image)

    # Revert to BGR for OpenCV display
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    # If landmarks are detected
    if results.pose_landmarks:
        landmarks = results.pose_landmarks.landmark
        
        # Get coordinates for left and right arms
        left_shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,
                         landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
        left_elbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x,
                      landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
        left_wrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x,
                      landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]
        right_shoulder = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,
                          landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
        right_elbow = [landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x,
                       landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y]
        right_wrist = [landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].x,
                       landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y]

        # Calculate angles
        left_shoulder_angle = calculate_angle(left_elbow, left_shoulder, [left_shoulder[0], left_shoulder[1]-0.1])
        left_elbow_angle = calculate_angle(left_shoulder, left_elbow, left_wrist)
        right_shoulder_angle = calculate_angle(right_elbow, right_shoulder, [right_shoulder[0], right_shoulder[1]-0.1])
        right_elbow_angle = calculate_angle(right_shoulder, right_elbow, right_wrist)

        # Display angles on frame
        cv2.putText(image, f'Left Shoulder Angle: {int(left_shoulder_angle)}', (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
        cv2.putText(image, f'Left Elbow Angle: {int(left_elbow_angle)}', (10, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
        cv2.putText(image, f'Right Shoulder Angle: {int(right_shoulder_angle)}', (10, 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
        cv2.putText(image, f'Right Elbow Angle: {int(right_elbow_angle)}', (10, 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

        # Write angles to CSV file with timestamp
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        with open(csv_filename, mode='a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([timestamp, left_shoulder_angle, left_elbow_angle])
            writer.writerow([timestamp, right_shoulder_angle, right_elbow_angle])

        # Custom drawing of landmarks and connections for both sides
        for landmark in [mp_pose.PoseLandmark.LEFT_SHOULDER, mp_pose.PoseLandmark.LEFT_ELBOW, mp_pose.PoseLandmark.LEFT_WRIST,
                         mp_pose.PoseLandmark.RIGHT_SHOULDER, mp_pose.PoseLandmark.RIGHT_ELBOW, mp_pose.PoseLandmark.RIGHT_WRIST]:
            x, y = int(landmarks[landmark.value].x * frame.shape[1]), int(landmarks[landmark.value].y * frame.shape[0])
            cv2.circle(image, (x, y), landmark_radius, landmark_color, -1)
        
        # Draw lines between shoulder, elbow, and wrist for both sides
        cv2.line(image, 
                 (int(left_shoulder[0] * frame.shape[1]), int(left_shoulder[1] * frame.shape[0])), 
                 (int(left_elbow[0] * frame.shape[1]), int(left_elbow[1] * frame.shape[0])), 
                 connection_color, 2)
        cv2.line(image, 
                 (int(left_elbow[0] * frame.shape[1]), int(left_elbow[1] * frame.shape[0])), 
                 (int(left_wrist[0] * frame.shape[1]), int(left_wrist[1] * frame.shape[0])), 
                 connection_color, 2)
        cv2.line(image, 
                 (int(right_shoulder[0] * frame.shape[1]), int(right_shoulder[1] * frame.shape[0])), 
                 (int(right_elbow[0] * frame.shape[1]), int(right_elbow[1] * frame.shape[0])), 
                 connection_color, 2)
        cv2.line(image, 
                 (int(right_elbow[0] * frame.shape[1]), int(right_elbow[1] * frame.shape[0])), 
                 (int(right_wrist[0] * frame.shape[1]), int(right_wrist[1] * frame.shape[0])), 
                 connection_color, 2)

    # Display the frame with both arms and landmarks drawn
    cv2.imshow('Dumbbell Shoulder Press Detection', image)

    # Exit on pressing 'q'
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
pose.close()
