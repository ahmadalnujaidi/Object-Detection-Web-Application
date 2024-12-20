# YOLO Real-Time Object Detection Web App with Generative AI Description

This project is a real-time object detection web application using a YOLO model, combined with a Generative AI model to produce storytelling descriptions of detected objects. The application captures frames from a webcam, performs object detection, and displays detections on a web interface. Additionally, it logs the detected items with their approximate locations and generates a story-like description of the scene every 5 seconds.

## Features

- **Real-Time Object Detection**: Detect objects in webcam footage using a YOLO model.
- **Generative AI Storytelling**: Generate a short, human-like story about the detected items and their positions in the frame.
- **Visual Layout**: Displays the video feed at the center, a detection log table on the right, and a story description at the bottom of the page.
- **Easy-to-Use Web Interface**: Interactive and visually organized layout.

## Requirements

- Python 3.8 or higher
- Required Python packages:
  - Flask
  - opencv-python-headless
  - torch
  - ultralytics
  - google-generativeai

## Application Details

### `app.py`

This file contains the Flask server setup and handles the main functionality:

- Captures frames from the webcam.
- Runs YOLO model inference on each frame to detect objects.
- Classifies objects into 9 different frame quadrants based on bounding box center.
- Routes:
  - `/video_feed`: Streams the video feed.
  - `/detection_log`: Provides real-time object detection logs.
  - `/generate_story`: Uses Google Generative AI to create a storytelling description of detected objects.

### `templates/index.html`

The main HTML file for the web interface, which displays:

- The centered video feed.
- The detection log in a table format on the right.
- The generated storytelling description at the bottom.
- A footer with copyright information and links to GitHub and LinkedIn.

### `gemini.py`

A standalone script for testing Generative AI-based storytelling. This script:

- Configures the Google Generative AI API.
- Uses the model `gemini-1.5-flash` to generate descriptive content.

## Example Output

Upon running, the web page displays:

- **Video Feed**: Real-time object detection video feed from the webcam.
- **Detection Log**: A table showing detected objects and their frame positions (e.g., “Center Left,” “Top Right”).
- **Storytelling Description**: A generative story about the objects in the frame, updated every 5 seconds.

## Acknowledgments

- [Ultralytics YOLO](https://github.com/ultralytics/yolov5)
- [Google Generative AI](https://developers.google.com/generative-ai)
