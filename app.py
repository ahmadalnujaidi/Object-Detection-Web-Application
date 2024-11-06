from flask import Flask, render_template, Response
from ultralytics import YOLO
import cv2

# Load the YOLO model
model = YOLO('ModelForCAI.pt')  # Make sure this path is correct

app = Flask(__name__)

def gen_frames():
    cap = cv2.VideoCapture(0)  # Access the default camera
    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break
        else:
            # Get the frame dimensions
            height, width, _ = frame.shape
            # Define quadrant boundaries
            third_width = width // 3
            third_height = height // 3

            # Perform inference with the model
            results = model(frame)

            # Process and draw results on the frame
            for result in results[0].boxes:
                x1, y1, x2, y2 = map(int, result.xyxy[0])  # Bounding box coordinates
                label = model.names[int(result.cls)]       # Class name
                confidence = result.conf[0].item()         # Confidence score

                # Calculate the center point of the bounding box
                center_x = (x1 + x2) // 2
                center_y = (y1 + y2) // 2

                # Determine the quadrant based on the center coordinates
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

                # Log the detected item and its quadrant
                print(f"Detected {label} in {quadrant}")

                # Draw bounding box and label on the frame
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, f'{label} {confidence:.2f} ({quadrant})', (x1, y1 - 10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

            # Encode frame to JPEG for streaming
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')  # Streaming response

    cap.release()

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/')
def index():
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
