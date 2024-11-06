from flask import Flask, render_template, Response, jsonify
from ultralytics import YOLO
import google.generativeai as genai
import logging
import cv2
import time
from dotenv import load_dotenv 
import os

load_dotenv() # Load environment variables from .env file

# Configure generative AI API
genai_api_key = os.getenv("GENERATIVE_AI_API_KEY")
genai.configure(api_key=genai_api_key)

# Load the YOLO model
model = YOLO('ModelForCAI.pt')

app = Flask(__name__)

# Suppress Ultralytics logging
logging.getLogger("ultralytics").setLevel(logging.WARNING)

# Global list to store recent detections and story text
detection_log = []
story_text = ""
last_story_update = 0

# Function to capture video frames and run model inference
def gen_frames():
    global detection_log
    cap = cv2.VideoCapture(0)  # Access the default camera
    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break
        else:
            # Get frame dimensions
            height, width, _ = frame.shape

            # Define quadrant boundaries
            third_width = width // 3
            third_height = height // 3

            # Perform inference with the model
            results = model(frame)
            detection_log.clear()  # Clear previous detections for the new frame

            # Process each detected object
            for result in results[0].boxes:
                x1, y1, x2, y2 = map(int, result.xyxy[0])
                label = model.names[int(result.cls)]
                confidence = result.conf[0].item()
                
                # Calculate the center of the bounding box
                center_x = (x1 + x2) // 2
                center_y = (y1 + y2) // 2
                box_width = x2 - x1
                box_height = y2 - y1

                # Check if bounding box size is at least 50
                if box_width >= 50 and box_height >= 50:
                    # Determine quadrant based on center point
                    if center_x < third_width:
                        if center_y < third_height:
                            quadrant = "Top Left"
                        elif center_y < 2 * third_height:
                            quadrant = "Center Left"
                        else:
                            quadrant = "Bottom Left"
                    elif center_x < 2 * third_width:
                        if center_y < third_height:
                            quadrant = "Center Top"
                        elif center_y < 2 * third_height:
                            quadrant = "Center Center"
                        else:
                            quadrant = "Center Bottom"
                    else:
                        if center_y < third_height:
                            quadrant = "Top Right"
                        elif center_y < 2 * third_height:
                            quadrant = "Center Right"
                        else:
                            quadrant = "Bottom Right"

                    # Log the detection and quadrant
                    detection_log.append(f"{label} in {quadrant}")

                    # Draw bounding box and label on the frame
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(frame, f'{label} ({quadrant})', (x1, y1 - 10), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

            # Encode frame to JPEG for streaming
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    cap.release()

# New route for the generative story
@app.route('/generate_story')
def generate_story():
    global story_text, last_story_update

    # Run this every 7 seconds
    if time.time() - last_story_update > 7:
        # Create a summary of the detections
        if detection_log:
            item_summary = ", ".join(detection_log)
            prompt = f"Create a short description of the scene, describing the following items and their positions of where they lie in the frame: {item_summary}."

            # Use Generative AI to create the story
            model = genai.GenerativeModel("gemini-1.5-flash")
            response = model.generate_content(prompt)
            story_text = response.text
        else:
            story_text = "No items detected in the scene at the moment."

        last_story_update = time.time()

    return jsonify({"story": story_text})

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/')
def index():
    return render_template('index.html')

# Route to provide detection log data
@app.route('/detection_log')
def get_detection_log():
    return jsonify(detection_log)

if __name__ == '__main__':
    app.run(debug=True)

