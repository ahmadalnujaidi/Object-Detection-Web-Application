from flask import Flask, render_template, Response, jsonify, request, redirect, url_for
from ultralytics import YOLO
import google.generativeai as genai
import logging
import cv2
import time
from dotenv import load_dotenv
import os

load_dotenv()

genai_api_key = os.getenv("GENERATIVE_AI_API_KEY")
genai.configure(api_key=genai_api_key)

model = YOLO('yolo11n.pt')

app = Flask(__name__)

logging.getLogger("ultralytics").setLevel(logging.WARNING)

detection_log = []
story_text = ""
last_story_update = 0
video_path = None
use_webcam = True


def gen_frames(source=0):
    global detection_log
    cap = cv2.VideoCapture(source)
    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break
        else:
            height, width, _ = frame.shape
            third_width = width // 3
            third_height = height // 3
            results = model(frame)
            detection_log.clear()

            for result in results[0].boxes:
                x1, y1, x2, y2 = map(int, result.xyxy[0])
                label = model.names[int(result.cls)]
                confidence = result.conf[0].item()
                center_x = (x1 + x2) // 2
                center_y = (y1 + y2) // 2
                box_width = x2 - x1
                box_height = y2 - y1

                if box_width >= 50 and box_height >= 50:
                    quadrant = get_quadrant(center_x, center_y, third_width, third_height)
                    detection_log.append(f"{label} in {quadrant}")
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(frame, f'{label} ({quadrant})', (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
    cap.release()


def get_quadrant(center_x, center_y, third_width, third_height):
    if center_x < third_width:
        if center_y < third_height:
            return "Top Left"
        elif center_y < 2 * third_height:
            return "Center Left"
        else:
            return "Bottom Left"
    elif center_x < 2 * third_width:
        if center_y < third_height:
            return "Center Top"
        elif center_y < 2 * third_height:
            return "Center Center"
        else:
            return "Center Bottom"
    else:
        if center_y < third_height:
            return "Top Right"
        elif center_y < 2 * third_height:
            return "Center Right"
        else:
            return "Bottom Right"


@app.route('/generate_story')
def generate_story():
    global story_text, last_story_update

    if time.time() - last_story_update > 7:
        if detection_log:
            item_summary = ", ".join(detection_log)
            prompt = f"Simply describe the following items and their positions of where they lie in the frame: {item_summary}."
            model = genai.GenerativeModel("gemini-1.5-flash")
            response = model.generate_content(prompt)
            story_text = response.text
        else:
            story_text = "No items detected in the scene at the moment."
        last_story_update = time.time()

    return jsonify({"story": story_text})


@app.route('/video_feed')
def video_feed():
    global video_path, use_webcam
    source = 0 if use_webcam else video_path
    return Response(gen_frames(source), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/upload_video', methods=['POST'])
def upload_video():
    global video_path, use_webcam
    use_webcam = False
    if 'video' not in request.files:
        return redirect(url_for('index'))
    file = request.files['video']
    if file.filename == '':
        return redirect(url_for('index'))
    video_path = os.path.join('uploads', file.filename)
    file.save(video_path)
    return redirect(url_for('index'))


@app.route('/use_webcam', methods=['POST'])
def switch_to_webcam():
    global use_webcam
    use_webcam = True
    return redirect(url_for('index'))


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/detection_log')
def get_detection_log():
    return jsonify(detection_log)


if __name__ == '__main__':
    os.makedirs('uploads', exist_ok=True)
    app.run(debug=True)
