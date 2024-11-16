import cv2
import numpy as np
from ultralytics import YOLO
import math
from transformers import CLIPProcessor, CLIPModel
import torch

class ViolenceTracker:
    def __init__(self, yolo_model_path, clip_model_name="openai/clip-vit-base-patch16"):
        # Load YOLO model for object detection
        self.model = YOLO(yolo_model_path)
        # Initialize CLIP model and processor for action classification
        self.processor = CLIPProcessor.from_pretrained(clip_model_name)
        self.clip_model = CLIPModel.from_pretrained(clip_model_name)
        self.clip_enabled = True
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.clip_model.to(self.device)

    def get_center_of_bbox(self, bbox):
        x1, y1, x2, y2 = bbox
        center_x = int((x1 + x2) / 2)
        center_y = int((y1 + y2) / 2)
        return (center_x, center_y)

    def measure_distance(self, p1, p2):
        return math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)

    def detect_frame(self, frame):
        # Detect objects (people) in the frame using YOLO
        results = self.model.track(frame, persist=True)[0]
        player_dict = {}
        for box in results.boxes:
            track_id = int(box.id.tolist()[0])
            bbox = box.xyxy.tolist()[0]
            if results.names[int(box.cls.tolist()[0])] == "person":
                player_dict[track_id] = bbox
        return player_dict

    def classify_action(self, cropped_image):
        actions = ["two people fighting", "a person walking", "a person sitting", "s person standing", "a person running"]
        inputs = self.processor(images=cropped_image, return_tensors="pt").to(self.device)
        text_inputs = self.processor(text=actions, return_tensors="pt", padding=True, truncation=True).to(self.device)
        with torch.no_grad():
            image_features = self.clip_model.get_image_features(**inputs)
            text_features = self.clip_model.get_text_features(**text_inputs)
            similarities = (image_features @ text_features.T)
            predicted_action_idx = similarities.argmax().item()
            predicted_action = actions[predicted_action_idx]
            
        return predicted_action

    def detect_violence(self, player_detections):
        violent_pairs = []
        for id1, bbox1 in player_detections.items():
            for id2, bbox2 in player_detections.items():
                if id1 >= id2:
                    continue
                center1, center2 = map(self.get_center_of_bbox, (bbox1, bbox2))
                distance = self.measure_distance(center1, center2)
                if distance < 100:  # Threshold for proximity
                    relative_velocity = np.linalg.norm(np.array(center1) - np.array(center2))
                    if relative_velocity > 1.5:  # Threshold for relative velocity (speed)
                        violent_pairs.append((id1, id2))
        return violent_pairs

    def classify_players(self, frame, player_detections):
        fighting_detected = False
        player_labels = {}
        for track_id, bbox in player_detections.items():
            x1, y1, x2, y2 = map(int, bbox)
            cropped_image = frame[y1:y2, x1:x2]
            if cropped_image.size > 0:
                action = self.classify_action(cropped_image)
                player_labels[track_id] = action
                if action == "fighting":
                    fighting_detected = True
        return player_labels,fighting_detected

    def draw_bboxes(self, frame, player_detections, violent_pairs, player_labels,fighting_detected):
        for track_id, bbox in player_detections.items():
            x1, y1, x2, y2 = bbox
            color = (0, 255, 0)  # Default color (Red)
            if any(track_id in pair for pair in violent_pairs):
                color = (0, 0, 255)  # Green color for violence
            label = f"Player ID: {track_id}"
            if track_id in player_labels:
                label += f" ({player_labels[track_id]})"
            cv2.putText(frame, label, (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
        if fighting_detected and len(violent_pairs) > 0:
            cv2.putText(frame, "Violence Detected!", (100, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)
        return frame

def process_frame(frame):
    # Detect players, classify actions, detect violence, and draw bounding boxes
    player_detections = tracker.detect_frame(frame)
    violent_pairs = tracker.detect_violence(player_detections)
    player_labels,fighting_detected = tracker.classify_players(frame, player_detections)
    frame = tracker.draw_bboxes(frame, player_detections, violent_pairs, player_labels,fighting_detected)
    cv2.imshow("Violence Detection", frame)

# Open the video source (webcam or video file)
def initialize_video_source(source="webcam", video_path="2.mp4"):
    if source == "webcam":
        # Initialize webcam (0 is the default webcam)
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("Error: Could not open webcam.")
            exit()
    elif source == "video":
        # Initialize video file
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"Error: Could not open video file {video_path}.")
            exit()
    return cap
source_type = "webcam"  # Change to "video" to test with a video file
video_path = "2.mp4"  # Specify the path to the video file if using "video"

# Initialize the video capture object based on the source type
cap = initialize_video_source(source=source_type, video_path=video_path)

tracker = ViolenceTracker(yolo_model_path="yolov8x.pt")  # Update path to YOLO model

frame_skip = 1  # Set frame_skip to 1 to process every frame (no skipping)
frame_count = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("Failed to read frame.")
        break
    frame_count += 1

    # Resize the frame for faster processing (optional but helpful for speed)
    frame = cv2.resize(frame, (640, 480))  # Resize to a smaller resolution (adjust as needed)

    # Debug: Print frame count
    print(f"Processing frame {frame_count}")
    
    # Process frame and display video with bounding boxes and labels
    process_frame(frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):  # Press 'q' to quit
        break

cap.release()
cv2.destroyAllWindows()
