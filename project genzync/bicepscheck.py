import cv2
import mediapipe as mp
import math
import numpy as np

# Initialize MediaPipe Pose Detection
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()
mp_drawing = mp.solutions.drawing_utils

# Function to calculate the angle between three points
def calculate_angle(a, b, c):
    angle = math.degrees(math.atan2(c.y - b.y, c.x - b.x) - math.atan2(a.y - b.y, a.x - b.x))
    return abs(angle)

# Function to get pose landmarks
def get_pose_landmarks(frame):
    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(image_rgb)
    
    if results.pose_landmarks:
        return results.pose_landmarks.landmark
    return None

# Function to get biceps curl angle (left arm)
def get_biceps_curl_angle(landmarks):
    left_shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER]
    left_elbow = landmarks[mp_pose.PoseLandmark.LEFT_ELBOW]
    left_wrist = landmarks[mp_pose.PoseLandmark.LEFT_WRIST]
    
    left_angle = calculate_angle(left_shoulder, left_elbow, left_wrist)
    return left_angle

# Create dataset by processing the video
def create_dataset(video_path):
    frames = extract_frames(video_path)
    dataset = []
    
    for frame in frames:
        landmarks = get_pose_landmarks(frame)
        if landmarks:
            angle = get_biceps_curl_angle(landmarks)
            dataset.append(angle)  # Collect the angle
            
    return dataset

# Function to extract frames from the video
def extract_frames(video_path):
    cap = cv2.VideoCapture(video_path)
    frame_list = []
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame_list.append(frame)
    
    cap.release()
    return frame_list

# Function to draw landmarks and connections on the frame
def draw_landmarks_and_connections(frame, landmarks):
    h, w, _ = frame.shape
    # Draw landmarks
    for landmark in landmarks:
        x, y = int(landmark.x * w), int(landmark.y * h)
        cv2.circle(frame, (x, y), 5, (0, 0, 255), cv2.FILLED)
    
    # Draw connections (lines between body joints)
    mp_drawing.draw_landmarks(frame, landmarks, mp_pose.POSE_CONNECTIONS)

# Compare real-time angle with the average dataset angle
def compare_with_dataset(current_angle, dataset):
    # Calculate the average angle from the dataset
    avg_angle = np.mean(dataset)
    # Set a tolerance threshold (adjustable)
    threshold = 10  # Tolerance in degrees (can be adjusted)
    
    if abs(current_angle - avg_angle) < threshold:
        return "Good"
    else:
        return "Not Perfect"

# Main function to capture live video and compare with dataset
def real_time_feedback(video_path):
    dataset = create_dataset(video_path)  # Train dataset from video (can be a training video)
    
    # Capture live video feed (using webcam)
    cap = cv2.VideoCapture(0)  # Use 0 for the default webcam, change to the video source if needed
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        landmarks = get_pose_landmarks(frame)
        if landmarks:
            # Draw landmarks and connections
            mp_drawing.draw_landmarks(frame, landmarks, mp_pose.POSE_CONNECTIONS)
            
            # Calculate the biceps curl angle in real-time
            angle = get_biceps_curl_angle(landmarks)
            
            # Compare real-time angle with the trained dataset
            feedback = compare_with_dataset(angle, dataset)
            
            # Display the feedback on the frame
            cv2.putText(frame, f"Angle: {angle:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
            cv2.putText(frame, f"Feedback: {feedback}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
        
        # Show the frame with feedback
        cv2.imshow('Biceps Curl Feedback', frame)
        
        # Press 'q' to exit the loop
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # Release the video capture and close all windows
    cap.release()
    cv2.destroyAllWindows()

# Run the real-time feedback with a training video
if __name__ == "__main__":
    video_path = "bicepscurl.mp4"  # Replace with the path to your training video
    real_time_feedback(video_path)
