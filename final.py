import cv2
import numpy as np
import time
import speech_recognition as sr
import threading
import matplotlib.pyplot as plt
from collections import Counter
import pyaudio
import wave
import struct
import librosa
from scipy.signal import find_peaks
import os
import json
from vosk import Model, KaldiRecognizer

# Import the EyeContactDetector class from the eye_contact_detector.py file
from eye_contact_detector import EyeContactDetector

class PresentationFeedback:
    def __init__(self):
        self.recording = False
        self.frames = []
        self.audio_frames = []
        self.start_time = None
        self.end_time = None
        self.eye_contact_frames = 0
        self.total_frames = 0
        self.filler_words = {
            # Um-like sounds
            "um": 0,
            # Uh-like sounds
            "uh": 0,
            # Er-like sounds
            "er": 0,
            # General hesitation words
            "like": 0,
            "you_know": 0,
            "basically": 0,
            "actually": 0,
            "sort_of": 0,
            "kind_of": 0,
            "i_mean": 0,
            "right": 0,
            "well": 0
        }
        self.transcript = ""
        
        # Add silence tracking
        self.silence_start_time = None
        self.awkward_pauses = 0
        self.SILENCE_THRESHOLD = 4.0  # seconds
        
        # Audio recording parameters
        self.CHUNK = 8000  # Smaller chunk size for better real-time processing
        self.FORMAT = pyaudio.paInt16  # Changed to 16-bit PCM
        self.CHANNELS = 1
        self.RATE = 16000  # Standard rate for speech recognition
        self.p = pyaudio.PyAudio()
        
        # Vocal analysis parameters
        self.note_history = []  # List of (note, duration) tuples
        self.current_note = None
        self.current_note_start = None
        self.min_note_duration = 0.5  # Minimum duration in seconds to count a note
        self.note_names = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
        self.silence_threshold = 0.01  # Threshold for silence detection
        
        # Initialize Vosk model
        print("Loading Vosk model...")
        model_path = "vosk-model-small-en-us-0.15"
        if not os.path.exists(model_path):
            print("Downloading Vosk model...")
            import urllib.request
            import zipfile
            url = "https://alphacephei.com/vosk/models/vosk-model-small-en-us-0.15.zip"
            urllib.request.urlretrieve(url, "vosk-model.zip")
            with zipfile.ZipFile("vosk-model.zip", 'r') as zip_ref:
                zip_ref.extractall(".")
            os.remove("vosk-model.zip")
        
        self.model = Model(model_path)
        self.recognizer = KaldiRecognizer(self.model, self.RATE)
        
        # Check microphone setup
        print("Checking microphone setup...")
        try:
            self.stream = self.p.open(
                format=self.FORMAT,
                channels=self.CHANNELS,
                rate=self.RATE,
                input=True,
                frames_per_buffer=self.CHUNK
            )
            print("Microphone found and working")
        except Exception as e:
            print(f"Warning: Microphone setup failed: {e}")
            print("Please check your microphone connection and permissions")
        
        # Initialize the EyeContactDetector from the imported class
        self.eye_detector = EyeContactDetector()
        
        # Add mapping for common misrecognitions
        self.filler_word_mapping = {
            # Um-like sounds
            "i'm": "um",
            "arm": "um",
            "hum": "um",
            "hmm": "um",
            "hem": "um",
            "um": "um",
            "umm": "um",
            
            # Uh-like sounds
            "uh": "uh",
            "ah": "uh",
            "a": "uh",
            "huh": "uh",
            "ha": "uh",
            
            # Er-like sounds
            "er": "er",
            "err": "er",
            "air": "er",
            "heir": "er",
            
            # Common filler phrases (keep as is)
            "like": "like",
            "you know": "you_know",
            "basically": "basically",
            "actually": "actually",
            "sort of": "sort_of",
            "kind of": "kind_of",
            "i mean": "i_mean",
            "right": "right",
            "well": "well"
        }
        
    def start_recording(self):
        """Start recording video and audio"""
        self.recording = True
        self.frames = []
        self.audio_frames = []
        
        # Start recording threads
        video_thread = threading.Thread(target=self.record_video)
        audio_thread = threading.Thread(target=self.record_audio)
        
        print("==== Starting Presentation Recording ====")
        print("Looking at the camera will be counted as eye contact.")
        print("Press 'q' to stop recording when finished.")
        
        # Start the timer only after webcam is ready
        self.start_time = time.time()
        
        video_thread.start()
        audio_thread.start()
        
        # Wait for threads to complete
        video_thread.join()
        audio_thread.join()
        
        # Process the recorded data
        self.analyze_recording()
        self.generate_feedback()
        
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
            
            # Calculate and display elapsed time
            elapsed_time = time.time() - self.start_time
            minutes = int(elapsed_time // 60)
            seconds = int(elapsed_time % 60)
            time_text = f"Time: {minutes:02d}:{seconds:02d}"
            
            # Display time overlay
            cv2.putText(frame, time_text, (10, 60), 
                      cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # Update eye contact counter using the detector's result
            if eye_contact:
                self.eye_contact_frames += 1
                cv2.putText(frame, "Eye Contact", (10, 30), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            else:
                cv2.putText(frame, "No Eye Contact", (10, 30), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
            # Display current note if available
            if hasattr(self, 'current_note') and self.current_note is not None:
                note_text = f"Current Note: {self.current_note}"
                cv2.putText(frame, note_text, 
                          (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # Display the frame with all overlays
            cv2.imshow('Presentation Recording', frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                self.recording = False
        
        self.end_time = time.time()
        self.eye_detector.cap.stop()
        cv2.destroyAllWindows()
    
    def frequency_to_note(self, frequency):
        """Convert frequency to musical note"""
        if frequency < 20:  # Below human hearing range
            return "Silence"
        
        # A4 = 440Hz
        note_number = 12 * np.log2(frequency/440) + 69
        note_number = round(note_number)
        
        if note_number < 0 or note_number > 127:  # Outside MIDI range
            return "Silence"
            
        octave = (note_number - 12) // 12 + 1
        note = self.note_names[note_number % 12]
        return f"{note}{octave}"

    def analyze_audio_chunk(self, audio_data):
        """Analyze a chunk of audio data for frequency content"""
        # Convert audio data to numpy array
        audio_array = np.frombuffer(audio_data, dtype=np.int16)
        
        # Normalize audio
        audio_float = audio_array.astype(float) / 32768.0
        
        # Check for silence
        if np.max(np.abs(audio_float)) < self.silence_threshold:
            return "Silence"
        
        # Perform FFT
        fft_data = np.fft.fft(audio_float)
        frequencies = np.fft.fftfreq(len(audio_float), 1/self.RATE)
        
        # Get magnitude spectrum
        magnitude = np.abs(fft_data)
        
        # Find peaks in the frequency spectrum
        peaks, _ = find_peaks(magnitude[:len(magnitude)//2])
        
        if len(peaks) == 0:
            return "Silence"
            
        # Get the frequency with highest magnitude
        dominant_freq = frequencies[peaks[np.argmax(magnitude[peaks])]]
        
        # Convert to note
        return self.frequency_to_note(dominant_freq)

    def record_audio(self):
        """Record and process audio using Vosk"""
        print("Starting audio recording...")
        
        while self.recording:
            try:
                # Read audio data
                data = self.stream.read(self.CHUNK, exception_on_overflow=False)
                
                # Process with Vosk
                if self.recognizer.AcceptWaveform(data):
                    result = json.loads(self.recognizer.Result())
                    text = result.get("text", "").lower()
                    
                    if text:
                        print(f"Real-time transcription: {text}")
                        self.audio_frames.append(text)
                        
                        # Process the text for filler words
                        self.detect_filler_words(text)
                
                # Analyze audio for vocal variety
                current_note = self.analyze_audio_chunk(data)
                
                # Track silence duration
                if current_note == "Silence":
                    if self.silence_start_time is None:
                        self.silence_start_time = time.time()
                    else:
                        silence_duration = time.time() - self.silence_start_time
                        if silence_duration >= self.SILENCE_THRESHOLD:
                            self.awkward_pauses += 1
                            self.silence_start_time = time.time()  # Reset to avoid counting the same pause multiple times
                else:
                    self.silence_start_time = None
                
                # Update note history
                current_time = time.time()
                if current_note != self.current_note:
                    if self.current_note is not None:
                        duration = current_time - self.current_note_start
                        if duration >= self.min_note_duration:
                            self.note_history.append((self.current_note, duration))
                    self.current_note = current_note
                    self.current_note_start = current_time
                
            except Exception as e:
                print(f"Audio recording error: {e}")
                continue

    def detect_filler_words(self, text):
        """Enhanced filler word detection with misrecognition handling"""
        # Add spaces around text for better word boundary detection
        text = f" {text} "
        
        # First, check for multi-word phrases
        multi_word_fillers = [k for k in self.filler_word_mapping.keys() if " " in k]
        for phrase in multi_word_fillers:
            if f" {phrase} " in text:
                category = self.filler_word_mapping[phrase]
                count = text.count(f" {phrase} ")
                self.filler_words[category] += count
                print(f"Found {count} instance(s) of '{phrase}' (classified as {category})")
        
        # Then check for single words
        words = text.split()
        for word in words:
            word = word.strip(".,!?")  # Remove punctuation
            if word in self.filler_word_mapping:
                category = self.filler_word_mapping[word]
                self.filler_words[category] += 1
                print(f"Found '{word}' (classified as {category})")
        
        # Context-based detection for potential filler words
        for i, word in enumerate(words):
            word = word.strip(".,!?")
            # Check for standalone "I'm", "a", etc. that are likely fillers
            if word in ["i'm", "arm", "a", "ah"] and i > 0:
                # Check if preceded by pause indicators or common setup words
                prev_word = words[i-1].strip(".,!?")
                if prev_word in ["and", "but", "so", "then", "like", "just"]:
                    category = self.filler_word_mapping.get(word, "um")  # Default to "um" category
                    self.filler_words[category] += 1
                    print(f"Found likely filler '{word}' based on context (classified as {category})")

    def process_audio(self):
        """Process the recorded audio"""
        print("\nProcessing recorded audio...")
        
        # Combine all transcribed text
        self.transcript = " ".join(self.audio_frames)
        print(f"Full transcript: {self.transcript}")
        
        # Get final result from Vosk
        final_result = json.loads(self.recognizer.FinalResult())
        final_text = final_result.get("text", "").lower()
        if final_text:
            print(f"Final transcription: {final_text}")
            self.transcript += " " + final_text

    def __del__(self):
        """Cleanup resources"""
        if hasattr(self, 'stream'):
            self.stream.stop_stream()
            self.stream.close()
        if hasattr(self, 'p'):
            self.p.terminate()

    def analyze_recording(self):
        """Analyze the recorded presentation"""
        print("\nAnalyzing presentation...")
        
        # Process audio for transcript and filler words
        self.process_audio()
        
        # Calculate presentation duration
        self.duration = self.end_time - self.start_time
        
        # Calculate eye contact percentage
        self.eye_contact_percent = (self.eye_contact_frames / max(1, self.total_frames)) * 100
        
        # Count total filler words
        self.total_fillers = sum(self.filler_words.values())
        self.filler_rate = self.total_fillers / max(1, self.duration / 60)  # Fillers per minute
    
    def analyze_vocal_variety(self):
        """Analyze vocal variety and generate a score"""
        if not self.note_history:
            return 0, "No vocal variety detected"
        
        # Filter out silence and short notes
        valid_notes = [(note, duration) for note, duration in self.note_history 
                      if note != "Silence" and duration >= self.min_note_duration]
        
        if not valid_notes:
            return 0, "No sustained vocal notes detected"
        
        # Extract unique notes and their total durations
        note_durations = {}
        for note, duration in valid_notes:
            if note in note_durations:
                note_durations[note] += duration
            else:
                note_durations[note] = duration
        
        # Calculate total duration of valid notes
        total_duration = sum(duration for _, duration in valid_notes)
        
        # Calculate note range (in semitones)
        notes = list(note_durations.keys())
        if not notes:
            return 0, "No valid notes detected"
            
        # Convert notes to MIDI numbers for range calculation
        def note_to_midi(note):
            note_name = note[:-1]  # Remove octave number
            octave = int(note[-1])
            note_index = self.note_names.index(note_name)
            return note_index + (octave * 12)
        
        midi_notes = [note_to_midi(note) for note in notes]
        note_range = max(midi_notes) - min(midi_notes)
        
        # Calculate variety score (0-100)
        # Factors:
        # 1. Note range (40% of score)
        # 2. Number of unique notes (30% of score)
        # 3. Distribution of notes (30% of score)
        
        range_score = min(40, note_range * 2)  # 20 semitones = 40 points
        unique_notes_score = min(30, len(notes) * 2)  # 15 unique notes = 30 points
        
        # Calculate distribution score
        durations = np.array(list(note_durations.values()))
        distribution = durations / np.sum(durations)
        distribution_score = 30 * (1 - np.std(distribution))  # More uniform = higher score
        
        total_score = range_score + unique_notes_score + distribution_score
        
        # Generate feedback
        feedback = []
        if note_range < 5:
            feedback.append("Your vocal range is very limited")
        elif note_range < 10:
            feedback.append("Your vocal range is moderate")
        else:
            feedback.append("You have good vocal range")
            
        if distribution_score < 15:
            feedback.append("You tend to favor certain notes")
        else:
            feedback.append("You distribute your notes well")
            
        return total_score, ", ".join(feedback)

    def generate_feedback(self):
        """Generate presentation feedback based on analysis"""
        print("\n==== Presentation Feedback Report ====")
        print(f"Duration: {self.duration:.1f} seconds ({self.duration/60:.1f} minutes)")
        
        # Basic statistics
        print("\n--- Basic Statistics ---")
        print(f"Eye contact maintained: {self.eye_contact_percent:.1f}% of the time")
        
        # Vocal variety analysis
        vocal_score, vocal_feedback = self.analyze_vocal_variety()
        print("\n--- Vocal Variety Analysis ---")
        print(f"Vocal Variety Score: {vocal_score:.1f}/100")
        print(f"Feedback: {vocal_feedback}")
        
        print("\n--- Verbal Analysis ---")
        print(f"Total filler words detected: {self.total_fillers}")
        print(f"Rate: {self.filler_rate:.1f} filler words per minute")
        print(f"Awkward pauses (>4s): {self.awkward_pauses}")
        print("\n--- Filler Word Analysis ---")
        print("Filler word categories:")
        
        # Group filler words by category
        category_totals = {
            "Um-like sounds": sum(self.filler_words[cat] for cat in ["um"]),
            "Uh-like sounds": sum(self.filler_words[cat] for cat in ["uh"]),
            "Er-like sounds": sum(self.filler_words[cat] for cat in ["er"]),
            "Other filler phrases": sum(self.filler_words[cat] for cat in 
                ["like", "you_know", "basically", "actually", "sort_of", "kind_of", "i_mean", "right", "well"])
        }
        
        for category, count in category_totals.items():
            print(f"  {category}: {count} instances")
        
        print("\nDetailed breakdown:")
        for word, count in self.filler_words.items():
            if count > 0:
                print(f"  - {word.replace('_', ' ')}: {count} times")
        
        # Overall feedback
        print("\n--- Overall Feedback ---")
        feedback = []
        
        # Eye contact feedback
        if self.eye_contact_percent < 30:
            feedback.append("Work on maintaining more eye contact with your audience")
        elif self.eye_contact_percent < 70:
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
            
        # Awkward pauses feedback
        if self.awkward_pauses > 3:
            feedback.append("You had several awkward pauses (>4s). Try to maintain a more consistent speaking pace")
        elif self.awkward_pauses > 0:
            feedback.append("You had a few awkward pauses.")
        else:
            feedback.append("You maintained a good speaking pace with no awkward pauses")
        
        # Print feedback
        for item in feedback:
            print(f"• {item}")

        print("\nPresentation analysis complete!")
        print("Keep practicing and refer to the feedback to improve your presentation skills.")

if __name__ == "__main__":
    feedback = PresentationFeedback()
    feedback.start_recording()