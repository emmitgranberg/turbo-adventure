import cv2
import numpy as np
import time
import speech_recognition as sr
import threading
import matplotlib.pyplot as plt
from collections import Counter
import dlib
import pyaudio

# Import the EyeContactDetector class from the eye_contact_detector.py file
from eye_contact_detector import EyeContactDetector

class PresentationFeedback:
    def __init__(self):
        self.recording = False
        self.frames = []
        self.audio_frames = []
        self.start_time = None
        self.end_time = None
        self.recognizer = sr.Recognizer()
        self.emotions = []
        self.eye_contact_frames = 0
        self.total_frames = 0
        self.filler_words = {
            "um": 0, "uh": 0, "er": 0, "ah": 0, "like": 0, 
            "you know": 0, "so": 0, "basically": 0, "actually": 0, 
            "ok": 0, "kind of": 0, "sort of": 0, "I mean": 0
        }
        self.transcript = ""
        self.presentation_context = None
        
        # Initialize the EyeContactDetector from the imported class
        self.eye_detector = EyeContactDetector()
        
        # Initialize dlib's face detector and facial landmark predictor for emotion detection
        self.detector = dlib.get_frontal_face_detector()
        
        # You'll need to download this file and provide the correct path
        # Download from: http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2
        self.predictor_path = "shape_predictor_68_face_landmarks.dat"
        try:
            self.predictor = dlib.shape_predictor(self.predictor_path)
            self.landmark_detection_enabled = True
        except:
            print(f"Warning: Could not load facial landmark predictor. Make sure {self.predictor_path} exists.")
            print("Emotion detection will be simplified.")
            self.landmark_detection_enabled = False
        
    def start_recording(self):
        """Start recording video and audio"""
        self.recording = True
        self.frames = []
        self.audio_frames = []
        self.start_time = time.time()
        self.emotions = []
        self.eye_contact_frames = 0
        self.total_frames = 0
        
        # Calibrate the eye contact detector before starting
        print("Calibrating eye contact detector...")
        print("Please look directly at the camera for a few seconds...")
        self.eye_detector.calibrate()
        time.sleep(3)  # Give the user time to look at the camera
        
        # Start recording threads
        video_thread = threading.Thread(target=self.record_video)
        audio_thread = threading.Thread(target=self.record_audio)
        
        print("==== Starting Presentation Recording ====")
        print("Looking at the camera will be counted as eye contact.")
        print("Press 'q' to stop recording when finished.")
        
        video_thread.start()
        audio_thread.start()
        
        # Wait for threads to complete
        video_thread.join()
        audio_thread.join()
        
        # Process the recorded data
        self.analyze_recording()
        self.generate_feedback()
        
    def detect_facial_expression(self, frame):
        """More nuanced facial expression detection"""
        # Convert to grayscale for face detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect faces
        faces = self.detector(gray)

        if not faces:
            return "none", None  # No face detected

        face = faces[0]  # Get the first face

        # Get facial landmarks if available
        if self.landmark_detection_enabled:
            landmarks = self.predictor(gray, face)

            # Calculate eyebrow-to-eye distance (for detecting sadness/anger)
            eyebrow_left = np.array([landmarks.part(19).x, landmarks.part(19).y])
            eye_left = np.array([landmarks.part(37).x, landmarks.part(37).y])
            eyebrow_eye_dist_left = np.linalg.norm(eyebrow_left - eye_left)

            eyebrow_right = np.array([landmarks.part(24).x, landmarks.part(24).y])  
            eye_right = np.array([landmarks.part(44).x, landmarks.part(44).y])
            eyebrow_eye_dist_right = np.linalg.norm(eyebrow_right - eye_right)

            # Normalize by face height
            face_height = face.height()
            eyebrow_eye_ratio = (eyebrow_eye_dist_left + eyebrow_eye_dist_right) / (2 * face_height)

            # Extract mouth coordinates for smile detection
            mouth_left = np.array([landmarks.part(48).x, landmarks.part(48).y])
            mouth_right = np.array([landmarks.part(54).x, landmarks.part(54).y])
            mouth_top = np.array([landmarks.part(51).x, landmarks.part(51).y])
            mouth_bottom = np.array([landmarks.part(57).x, landmarks.part(57).y])

            # Calculate mouth aspect ratio
            mouth_width = np.linalg.norm(mouth_right - mouth_left)
            mouth_height = np.linalg.norm(mouth_bottom - mouth_top)
            mouth_aspect_ratio = mouth_height / max(mouth_width, 1)

            # Calculate smile ratio (curvature of the mouth)
            mouth_corners = [landmarks.part(48), landmarks.part(54)]
            mouth_center = landmarks.part(51)

            corner_y_avg = (mouth_corners[0].y + mouth_corners[1].y) / 2
            smile_curve = (corner_y_avg - mouth_center.y) / face_height

            # Improved expression detection
            if mouth_aspect_ratio > 0.5:  # Open mouth
                expression = "surprised"
            elif smile_curve > 0.01:  # Curved up mouth - happy
                expression = "happy"
            elif eyebrow_eye_ratio < 0.12:  # Low eyebrows - sad or angry
                expression = "sad"
            else:
                expression = "neutral"

            # Convert dlib rectangle to OpenCV format for visualization
            (x, y, w, h) = (face.left(), face.top(), face.width(), face.height())
            face_rect = (x, y, w, h)

            return expression, face_rect
        else:
            # Simplified detection without landmarks
            (x, y, w, h) = (face.left(), face.top(), face.width(), face.height())
            face_rect = (x, y, w, h)

            # Just return neutral as we can't detect expression without landmarks
            return "neutral", face_rect
    
    def record_video(self):
        """Record video frames and analyze in real-time"""
        while self.recording:
            # Use the EyeContactDetector to get frame and eye contact status
            frame, eye_contact = self.eye_detector.detect_eye_contact()
            if frame is None:
                print("Error: Could not get frame from eye contact detector")
                break
                    
            self.total_frames += 1
            
            # Store frame for later processing
            self.frames.append(frame.copy())
            
            # Update eye contact counter using the detector's result
            if eye_contact:
                self.eye_contact_frames += 1
                cv2.putText(frame, "Eye Contact ✓", (10, 60), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # Real-time face analysis for emotions
            expression, face_rect = self.detect_facial_expression(frame)
            if expression != "none":
                self.emotions.append(expression)
                
                if face_rect:
                    # Display the detected face
                    x, y, w, h = face_rect
                    cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                    
                    # Display expression
                    cv2.putText(frame, f"Expression: {expression}", (10, 90), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # Display recording indicator
            cv2.putText(frame, "Recording...", (10, 30), 
                      cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            
            cv2.imshow('Presentation Recording', frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                self.recording = False
        
        self.end_time = time.time()
        self.eye_detector.cap.stop()
        cv2.destroyAllWindows()
    
    def record_audio(self):
        """Record and process audio"""
        mic = sr.Microphone()
        
        with mic as source:
            self.recognizer.adjust_for_ambient_noise(source)
            
            while self.recording:
                try:
                    print("Listening...")
                    audio = self.recognizer.listen(source, timeout=2, phrase_time_limit=10)
                    self.audio_frames.append(audio)
                    print("Audio chunk recorded")
                except sr.WaitTimeoutError:
                    print("Listening timeout, continuing...")
                    continue
                except Exception as e:
                    print(f"Audio recording error: {e}")
                    continue
    
    def analyze_recording(self):
        """Analyze the recorded presentation"""
        print("\nAnalyzing presentation...")
        
        # Process audio for transcript and filler words
        self.process_audio()
        
        # Calculate presentation duration
        self.duration = self.end_time - self.start_time
        
        # Calculate eye contact percentage
        self.eye_contact_percent = (self.eye_contact_frames / max(1, self.total_frames)) * 100
        
        # Count emotions
        emotion_counter = Counter(self.emotions)
        self.dominant_emotion = emotion_counter.most_common(1)[0][0] if emotion_counter else "neutral"
        
        # Get emotion percentages
        self.emotion_percentages = {}
        for emotion, count in emotion_counter.items():
            self.emotion_percentages[emotion] = (count / max(1, len(self.emotions))) * 100
            
        # Detect presentation context
        if "happy" in self.emotion_percentages and self.emotion_percentages["happy"] > 50:
            self.presentation_context = "light"
        elif "sad" in self.emotion_percentages and self.emotion_percentages.get("sad", 0) > 30:
            self.presentation_context = "serious"
        elif "neutral" in self.emotion_percentages and self.emotion_percentages["neutral"] > 40:
            self.presentation_context = "neutral"
        else:
            self.presentation_context = "mixed"
            
        # Count total filler words
        self.total_fillers = sum(self.filler_words.values())
        self.filler_rate = self.total_fillers / max(1, self.duration / 60)  # Fillers per minute
    
    def process_audio(self):
        """Process audio to get transcript and count filler words"""
        all_text = []

        print(f"Processing {len(self.audio_frames)} audio chunks...")

        # Process each audio segment
        for i, audio in enumerate(self.audio_frames):
            try:
                print(f"Transcribing chunk {i+1}/{len(self.audio_frames)}...")

                # Use a different approach for filler word detection
                # 1. First get the normal transcript
                text = self.recognizer.recognize_google(audio).lower()
                print(f"Transcribed: {text}")
                all_text.append(text)

                # 2. Do a separate processing specifically for filler sounds
                # This is an approximation as speech recognition usually filters these out
                # We'll need to use a different model or approach to detect these more accurately

                # Count words that might be transcribed versions of fillers
                for word in ["um", "uh", "er", "ah"]:
                    if f" {word} " in f" {text} ":
                        self.filler_words[word] += text.count(f" {word} ")

                # Count "like" and "you know" only when used as fillers (approximation)
                if " like " in text:
                    # Count only instances not part of phrases like "I like it" or "looks like"
                    for pattern in [" like, ", " like "]:
                        self.filler_words["like"] += text.count(pattern)

                if " you know " in text:
                    self.filler_words["you know"] += text.count(" you know ")

                # Check for "so" at the beginning of sentences (common filler)
                if text.startswith("so ") or " so " in text:
                    self.filler_words["so"] += 1

            except sr.UnknownValueError:
                print("Could not understand audio")
                continue
            except sr.RequestError as e:
                print(f"Could not request results; {e}")
                continue
            except Exception as e:
                print(f"Error processing audio: {e}")
                continue

        self.transcript = " ".join(all_text)
        print(f"Full transcript: {self.transcript}")
    
    def generate_feedback(self):
        """Generate presentation feedback based on analysis"""
        print("\n==== Presentation Feedback Report ====")
        print(f"Duration: {self.duration:.1f} seconds ({self.duration/60:.1f} minutes)")
        
        # Basic statistics
        print("\n--- Basic Statistics ---")
        print(f"Eye contact maintained: {self.eye_contact_percent:.1f}% of the time")
        print(f"Dominant facial expression: {self.dominant_emotion}")
        print("Facial expressions breakdown:")
        for emotion, percentage in self.emotion_percentages.items():
            print(f"  - {emotion}: {percentage:.1f}%")
        
        print("\n--- Verbal Analysis ---")
        print(f"Total filler words detected: {self.total_fillers}")
        print(f"Rate: {self.filler_rate:.1f} filler words per minute")
        print("Filler word breakdown:")
        for word, count in self.filler_words.items():
            if count > 0:
                print(f"  - '{word}': {count} times")
        
        # Generate visualization
        self.generate_visualizations()
        
        # Overall feedback
        print("\n--- Overall Feedback ---")
        feedback = []
        
        # Eye contact feedback
        if self.eye_contact_percent < 30:
            feedback.append("Work on maintaining more eye contact with your audience")
        elif self.eye_contact_percent < 60:
            feedback.append("Your eye contact is acceptable but could be improved")
        else:
            feedback.append("You maintain excellent eye contact")
        
        # Filler words feedback
        if self.filler_rate > 10:
            feedback.append("Try to significantly reduce filler words (um, uh, like, etc.)")
        elif self.filler_rate > 5:
            feedback.append("Work on reducing some of your filler words")
        else:
            feedback.append("You use minimal filler words, which is excellent")
        
        # Expression feedback
        # Expression feedback
        if self.presentation_context == "serious" and self.emotion_percentages.get("happy", 0) > 40:
            feedback.append("Your expressions appear too cheerful for what seems to be serious content")
        elif self.presentation_context == "light" and self.emotion_percentages.get("sad", 0) > 20:
            feedback.append("Try to appear more positive for this lighter presentation context")
        elif self.dominant_emotion == "neutral" and self.emotion_percentages.get("neutral", 0) > 80:
            feedback.append("Try to incorporate more expressive facial communication")
        
        # Print feedback
        for item in feedback:
            print(f"• {item}")

        print("\nPresentation analysis complete! Charts have been saved.")
        print("Keep practicing and refer to the feedback to improve your presentation skills.")
        
    def generate_visualizations(self):
        """Generate and save visualizations of the presentation analysis"""
        # Create a figure with subplots
        plt.figure(figsize=(15, 10))
        
        # Plot 1: Eye Contact Over Time
        plt.subplot(2, 2, 1)
        time_bins = min(20, self.total_frames)  # Divide recording into bins
        frames_per_bin = max(1, self.total_frames // time_bins)
        eye_contact_timeline = []
        
        for i in range(0, self.total_frames, frames_per_bin):
            end_idx = min(i + frames_per_bin, self.total_frames)
            segment_frames = end_idx - i
            if segment_frames > 0:
                # Check how many frames in this segment had eye contact
                eye_frames = sum(1 for j in range(i, end_idx) if j < len(self.frames) and 
                               self.eye_detector.eye_contact)  # This is an approximation
                eye_contact_timeline.append((eye_frames / segment_frames) * 100)
        
        plt.plot(eye_contact_timeline, color='blue', marker='o')
        plt.title('Eye Contact Over Time')
        plt.xlabel('Presentation Timeline')
        plt.ylabel('Eye Contact %')
        plt.ylim(0, 100)
        plt.grid(True, alpha=0.3)
        
        # Plot 2: Emotion Distribution
        plt.subplot(2, 2, 2)
        emotions = list(self.emotion_percentages.keys())
        percentages = list(self.emotion_percentages.values())
        plt.pie(percentages, labels=emotions, autopct='%1.1f%%', startangle=90, colors=['skyblue', 'yellowgreen', 'lightcoral', 'gold'])
        plt.axis('equal')
        plt.title('Emotional Expression Distribution')
        
        # Plot 3: Filler Words
        plt.subplot(2, 2, 3)
        filler_words = [word for word, count in self.filler_words.items() if count > 0]
        counts = [count for word, count in self.filler_words.items() if count > 0]
        if filler_words:
            plt.bar(filler_words, counts, color='maroon')
            plt.title('Filler Word Usage')
            plt.xlabel('Word')
            plt.ylabel('Count')
            plt.xticks(rotation=45)
            plt.grid(True, alpha=0.3, axis='y')
        else:
            plt.text(0.5, 0.5, "No filler words detected", ha='center', va='center', fontsize=12)
            plt.title('Filler Word Usage')
            plt.xticks([])
            plt.yticks([])
        
        # Plot 4: Overall Presentation Quality
        plt.subplot(2, 2, 4)
        metrics = ['Eye Contact', 'Expression', 'Few Fillers']
        
        # Calculate scores (0-100)
        eye_score = min(100, self.eye_contact_percent)
        
        expression_score = 0
        if self.presentation_context == "light" and self.emotion_percentages.get("happy", 0) > 40:
            expression_score = min(100, self.emotion_percentages.get("happy", 0) * 1.5)
        elif self.presentation_context == "serious" and self.emotion_percentages.get("sad", 0) > 20:
            expression_score = min(100, self.emotion_percentages.get("sad", 0) * 1.5)
        elif self.dominant_emotion != "neutral":
            expression_score = min(100, 100 - self.emotion_percentages.get("neutral", 0))
        else:
            expression_score = max(0, 50 - self.emotion_percentages.get("neutral", 0) / 2)
        
        # Score for low filler word usage (lower is better)
        filler_score = max(0, 100 - min(100, self.filler_rate * 10))
        
        scores = [eye_score, expression_score, filler_score]
        
        plt.bar(metrics, scores, color=['blue', 'green', 'purple'])
        plt.title('Presentation Quality Metrics')
        plt.ylabel('Score (0-100)')
        plt.ylim(0, 100)
        plt.grid(True, alpha=0.3, axis='y')
        
        # Adjust layout and save
        plt.tight_layout()
        plt.savefig('presentation_analysis.png')
        plt.close()

if __name__ == "__main__":
    feedback = PresentationFeedback()
    feedback.start_recording()
