import cv2
import mediapipe as mp
import math
import csv

# Initialize MediaPipe pose detection
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()
mp_drawing = mp.solutions.drawing_utils

# Open the webcam
cap = cv2.VideoCapture(0)

# Load CSV dataset with angle data (this file should be prepared beforehand)
csv_filename = "shoulder_press_data.csv"

# Function to load the CSV dataset of correct angles
def load_csv_data():
    angles_data = {}
    with open(csv_filename, mode='r') as file:
        reader = csv.reader(file)
        next(reader)  # Skip the header
        for row in reader:
            joint = row[0]
            correct_angle = float(row[1])
            angles_data[joint] = correct_angle
    return angles_data

# Load the dataset
correct_angles = load_csv_data()

# Function to calculate angle between three points
def calculate_angle(a, b, c):
    angle = math.degrees(math.atan2(c.y - b.y, c.x - b.x) - math.atan2(a.y - b.y, a.x - b.x))
    return abs(angle)

# Function to check if the form is correct
def check_form(angle, correct_angle):
    tolerance = 10  # Tolerance in degrees
    if abs(angle - correct_angle) <= tolerance:
        return True, None
    else:
        return False, correct_angle

# OpenCV window size setup
frame_width = 640
frame_height = 480

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    # Resize the frame to fit the display
    frame = cv2.resize(frame, (frame_width, frame_height))

    # Convert the image to RGB (MediaPipe requires RGB input)
    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(image_rgb)

    # Check if any landmarks were detected
    if results.pose_landmarks:
        h, w, _ = frame.shape
        landmarks = results.pose_landmarks.landmark

        # Define pairs of points to draw lines between body joints
        body_connections = [
            (mp_pose.PoseLandmark.LEFT_SHOULDER, mp_pose.PoseLandmark.RIGHT_SHOULDER),
            (mp_pose.PoseLandmark.LEFT_SHOULDER, mp_pose.PoseLandmark.LEFT_ELBOW),
            (mp_pose.PoseLandmark.LEFT_ELBOW, mp_pose.PoseLandmark.LEFT_WRIST),
            (mp_pose.PoseLandmark.RIGHT_SHOULDER, mp_pose.PoseLandmark.RIGHT_ELBOW),
            (mp_pose.PoseLandmark.RIGHT_ELBOW, mp_pose.PoseLandmark.RIGHT_WRIST),
            (mp_pose.PoseLandmark.LEFT_SHOULDER, mp_pose.PoseLandmark.LEFT_HIP),
            (mp_pose.PoseLandmark.RIGHT_SHOULDER, mp_pose.PoseLandmark.RIGHT_HIP),
            (mp_pose.PoseLandmark.LEFT_HIP, mp_pose.PoseLandmark.RIGHT_HIP),
            (mp_pose.PoseLandmark.LEFT_HIP, mp_pose.PoseLandmark.LEFT_KNEE),
            (mp_pose.PoseLandmark.RIGHT_HIP, mp_pose.PoseLandmark.RIGHT_KNEE),
            (mp_pose.PoseLandmark.LEFT_KNEE, mp_pose.PoseLandmark.LEFT_ANKLE),
            (mp_pose.PoseLandmark.RIGHT_KNEE, mp_pose.PoseLandmark.RIGHT_ANKLE)
        ]

        # Draw the connections
        for connection in body_connections:
            start = landmarks[connection[0]]
            end = landmarks[connection[1]]
            start_point = (int(start.x * w), int(start.y * h))
            end_point = (int(end.x * w), int(end.y * h))
            cv2.line(frame, start_point, end_point, (255, 255, 255), 3)
            cv2.circle(frame, start_point, 5, (0, 0, 255), cv2.FILLED)
            cv2.circle(frame, end_point, 5, (0, 0, 255), cv2.FILLED)

        # Calculate angles and compare them to the CSV dataset
        feedback_lines = []
        
        # Define the points needed for checking angles
        joints_to_check = {
            "left_shoulder": (mp_pose.PoseLandmark.LEFT_ELBOW, mp_pose.PoseLandmark.LEFT_SHOULDER, mp_pose.PoseLandmark.LEFT_HIP),
            "right_shoulder": (mp_pose.PoseLandmark.RIGHT_ELBOW, mp_pose.PoseLandmark.RIGHT_SHOULDER, mp_pose.PoseLandmark.RIGHT_HIP),
            "left_elbow": (mp_pose.PoseLandmark.LEFT_SHOULDER, mp_pose.PoseLandmark.LEFT_ELBOW, mp_pose.PoseLandmark.LEFT_WRIST),
            "right_elbow": (mp_pose.PoseLandmark.RIGHT_SHOULDER, mp_pose.PoseLandmark.RIGHT_ELBOW, mp_pose.PoseLandmark.RIGHT_WRIST)
        }

        # Check angles and provide feedback
        for joint_name, points in joints_to_check.items():
            point1, point2, point3 = points
            angle = calculate_angle(landmarks[point1], landmarks[point2], landmarks[point3])
            is_correct, correct_angle = check_form(angle, correct_angles.get(joint_name, 0))
            
            if not is_correct:
                feedback_lines.append(f"{joint_name.replace('_', ' ').capitalize()}: {angle:.1f}° (Target: {correct_angle:.1f}°)")
        
        # Display feedback on the frame
        if feedback_lines:
            y_offset = 30
            for line in feedback_lines:
                cv2.putText(frame, line, (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2, cv2.LINE_AA)
                y_offset += 30  # Move down for the next line
        else:
            cv2.putText(frame, "Perfect Form!", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2, cv2.LINE_AA)

    # Display the image with body connections and feedback
    cv2.imshow('Pose Detection - Form Check', frame)

    # Break the loop on pressing 'q'
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

# Release the webcam and close windows
cap.release()
cv2.destroyAllWindows()
