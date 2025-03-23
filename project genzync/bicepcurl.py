import cv2
import mediapipe as mp
import math

# Initialize MediaPipe Pose Detection
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()
mp_drawing = mp.solutions.drawing_utils

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

# Function to calculate the angle between three points
def calculate_angle(a, b, c):
    angle = math.degrees(math.atan2(c.y - b.y, c.x - b.x) - math.atan2(a.y - b.y, a.x - b.x))
    return abs(angle)

# Get pose landmarks
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
            # Collect joint data
            left_shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER]
            left_elbow = landmarks[mp_pose.PoseLandmark.LEFT_ELBOW]
            left_wrist = landmarks[mp_pose.PoseLandmark.LEFT_WRIST]
            
            joint_data = {
                'angle': angle,
                'shoulder': (left_shoulder.x, left_shoulder.y),
                'elbow': (left_elbow.x, left_elbow.y),
                'wrist': (left_wrist.x, left_wrist.y)
            }
            
            dataset.append(joint_data)
    
    return dataset

# Function to draw landmarks and connections on the frame
def draw_landmarks_and_connections(frame, landmarks):
    h, w, _ = frame.shape
    # Draw landmarks
    for landmark in landmarks:
        x, y = int(landmark.x * w), int(landmark.y * h)
        cv2.circle(frame, (x, y), 5, (0, 0, 255), cv2.FILLED)
    
    # Draw connections (lines between body joints)
    mp_drawing.draw_landmarks(frame, landmarks, mp_pose.POSE_CONNECTIONS)

# Create normal window size for display
def set_normal_window():
    cv2.namedWindow("Biceps Curl Feedback", cv2.WINDOW_NORMAL)

# Main function to create the dataset from the recorded video
if __name__ == "__main__":
    video_path = "bicepscurl.mp4"  # Replace with the path to your recorded video
    dataset = create_dataset(video_path)
    
    # Display the dataset angles (you can also save this data to a CSV or a file)
    print("Generated Dataset (Angles and Joint Positions from video frames):")
    for data in dataset:
        print(data)
    
    # Optional: Save to a CSV for later training
    with open("biceps_curl_dataset.csv", "w") as file:
        for data in dataset:
            file.write(f"Angle: {data['angle']}, Shoulder: {data['shoulder']}, Elbow: {data['elbow']}, Wrist: {data['wrist']}\n")

    # Set normal window before starting the preview
    set_normal_window()

    # Display the frames with angles for preview
    cap = cv2.VideoCapture(video_path)
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        landmarks = get_pose_landmarks(frame)
        if landmarks:
            # Draw landmarks and connections
            draw_landmarks_and_connections(frame, landmarks)
            
            # Calculate and display the biceps curl angle
            angle = get_biceps_curl_angle(landmarks)
            cv2.putText(frame, f"Angle: {angle:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
            
            # Display joint coordinates (shoulder, elbow, wrist) on the frame
            left_shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER]
            left_elbow = landmarks[mp_pose.PoseLandmark.LEFT_ELBOW]
            left_wrist = landmarks[mp_pose.PoseLandmark.LEFT_WRIST]
            
            cv2.putText(frame, f"Shoulder: ({int(left_shoulder.x*100)}%, {int(left_shoulder.y*100)}%)", 
                        (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            cv2.putText(frame, f"Elbow: ({int(left_elbow.x*100)}%, {int(left_elbow.y*100)}%)", 
                        (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            cv2.putText(frame, f"Wrist: ({int(left_wrist.x*100)}%, {int(left_wrist.y*100)}%)", 
                        (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

        # Display the frame with landmarks and the calculated angle
        cv2.imshow('Biceps Curl Feedback', frame)
        
        # Press 'q' to exit the loop
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # Release the video capture and close all windows
    cap.release()
    cv2.destroyAllWindows()
