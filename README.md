# Eye Contact Detector Web Application

This web application uses computer vision to detect and track eye contact through your webcam. It provides real-time feedback and statistics about your eye contact duration.

## Features

- Real-time webcam feed with eye contact detection overlay
- Start/Stop recording functionality
- Statistics tracking:
  - Total frames processed
  - Frames with eye contact
  - Eye contact percentage
- Visual progress bar
- Modern, responsive UI

## Setup

1. Install Python 3.7 or higher
2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Running the Application

1. Start the Flask server:
   ```bash
   python app.py
   ```
2. Open your web browser and navigate to:
   ```
   http://localhost:5000
   ```
3. Allow camera access when prompted
4. Click "Start Recording" to begin tracking eye contact
5. Click "Stop Recording" to end the session and view final statistics

## Notes

- Make sure you have good lighting for optimal detection
- Position yourself properly in front of the camera
- The application works best when your face is clearly visible 