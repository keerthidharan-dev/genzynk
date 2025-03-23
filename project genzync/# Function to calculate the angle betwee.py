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

# Function to detect if a push-up is performed correctly
def check_pushup_form(landmarks):
    left_shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER]
    left_elbow = landmarks[mp_pose.PoseLandmark.LEFT_ELBOW]
    left_wrist = landmarks[mp_pose.PoseLandmark.LEFT_WRIST]
    
    right_shoulder = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER]
    right_elbow = landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW]
    right_wrist = landmarks[mp_pose.PoseLandmark.RIGHT_WRIST]
    
    # Calculate angles for both arms
    left_angle = calculate_angle(left_shoulder, left_elbow, left_wrist)
    right_angle = calculate_angle(right_shoulder, right_elbow, right_wrist)

    return left_angle, right_angle

# Function to provide feedback based on the push-up form
def provide_feedback(left_angle, right_angle):
    feedback = []
    
    # Check for correct push-up form
    if left_angle < 30 or right_angle < 30:
        feedback.append("Keep your elbows bent more.")
    elif left_angle > 150 or right_angle > 150:
        feedback.append("Don't lock your elbows.")
    else:
        feedback.append("Good form!")
    
    return feedback
