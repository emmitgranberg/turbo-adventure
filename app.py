from flask import Flask, render_template, Response, jsonify, request
import cv2
import numpy as np
import mediapipe as mp
import time
import threading
import pyaudio
import json
import os
from scipy.signal import find_peaks
import numpy as np
from collections import Counter
from eye_contact_detector import EyeContactDetector

# For speech recognition
try:
    from vosk import Model, KaldiRecognizer
    VOSK_AVAILABLE = True
except ImportError:
    print("Warning: Vosk not available. Speech recognition will be disabled.")
    VOSK_AVAILABLE = False

app = Flask(__name__)

# Global variables
detector = None
recording = False
eye_contact_stats = {
    "total_frames": 0,
    "eye_contact_frames": 0,
    "percentage": 0
}

# Audio recording globals
audio_thread = None
audio_recording = False
transcript = []
filler_words = {
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
note_history = []
current_note = None
current_note_start = None
silence_start_time = None
awkward_pauses = 0

# Audio parameters
CHUNK = 8000
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000
silence_threshold = 0.01
SILENCE_THRESHOLD = 4.0  # seconds for awkward pause detection
note_names = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
min_note_duration = 0.5

# Initialize Vosk model if available
recognizer = None
if VOSK_AVAILABLE:
    try:
        model_path = "vosk-model-small-en-us-0.15"
        if not os.path.exists(model_path):
            print("Vosk model not found. Please download it manually.")
            print("Visit: https://alphacephei.com/vosk/models")
        else:
            model = Model(model_path)
            recognizer = KaldiRecognizer(model, RATE)
            print("Vosk model loaded successfully")
    except Exception as e:
        print(f"Error loading Vosk model: {e}")

# Filler word mapping for recognition
filler_word_mapping = {
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
    
    # Common filler phrases
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

def frequency_to_note(frequency):
    """Convert frequency to musical note"""
    if frequency < 20:  # Below human hearing range
        return "Silence"
    
    # A4 = 440Hz
    note_number = 12 * np.log2(frequency/440) + 69
    note_number = round(note_number)
    
    if note_number < 0 or note_number > 127:  # Outside MIDI range
        return "Silence"
        
    octave = (note_number - 12) // 12 + 1
    note = note_names[note_number % 12]
    return f"{note}{octave}"

def analyze_audio_chunk(audio_data):
    """Analyze a chunk of audio data for frequency content"""
    # Convert audio data to numpy array
    audio_array = np.frombuffer(audio_data, dtype=np.int16)
    
    # Normalize audio
    audio_float = audio_array.astype(float) / 32768.0
    
    # Check for silence
    if np.max(np.abs(audio_float)) < silence_threshold:
        return "Silence"
    
    # Perform FFT
    fft_data = np.fft.fft(audio_float)
    frequencies = np.fft.fftfreq(len(audio_float), 1/RATE)
    
    # Get magnitude spectrum
    magnitude = np.abs(fft_data)
    
    # Find peaks in the frequency spectrum
    peaks, _ = find_peaks(magnitude[:len(magnitude)//2])
    
    if len(peaks) == 0:
        return "Silence"
        
    # Get the frequency with highest magnitude
    dominant_freq = frequencies[peaks[np.argmax(magnitude[peaks])]]
    
    # Convert to note
    return frequency_to_note(dominant_freq)

def detect_filler_words(text):
    """Enhanced filler word detection with misrecognition handling"""
    global filler_words
    
    # Add spaces around text for better word boundary detection
    text = f" {text} "
    
    # First, check for multi-word phrases
    multi_word_fillers = [k for k in filler_word_mapping.keys() if " " in k]
    for phrase in multi_word_fillers:
        if f" {phrase} " in text:
            category = filler_word_mapping[phrase]
            count = text.count(f" {phrase} ")
            filler_words[category] += count
            print(f"Found {count} instance(s) of '{phrase}' (classified as {category})")
    
    # Then check for single words
    words = text.split()
    for word in words:
        word = word.strip(".,!?")  # Remove punctuation
        if word in filler_word_mapping:
            category = filler_word_mapping[word]
            filler_words[category] += 1
            print(f"Found '{word}' (classified as {category})")
    
    # Context-based detection for potential filler words
    for i, word in enumerate(words):
        word = word.strip(".,!?")
        # Check for standalone "I'm", "a", etc. that are likely fillers
        if word in ["i'm", "arm", "a", "ah"] and i > 0:
            # Check if preceded by pause indicators or common setup words
            prev_word = words[i-1].strip(".,!?")
            if prev_word in ["and", "but", "so", "then", "like", "just"]:
                category = filler_word_mapping.get(word, "um")  # Default to "um" category
                filler_words[category] += 1
                print(f"Found likely filler '{word}' based on context (classified as {category})")

def record_audio():
    """Record and process audio using Vosk"""
    global audio_recording, transcript, current_note, current_note_start, silence_start_time, awkward_pauses
    
    if not VOSK_AVAILABLE or recognizer is None:
        print("Warning: Speech recognition not available")
        return
        
    p = pyaudio.PyAudio()
    stream = p.open(
        format=FORMAT,
        channels=CHANNELS,
        rate=RATE,
        input=True,
        frames_per_buffer=CHUNK
    )
    
    print("Starting audio recording...")
    
    while audio_recording:
        try:
            # Read audio data
            data = stream.read(CHUNK, exception_on_overflow=False)
            
            # Process with Vosk
            if recognizer.AcceptWaveform(data):
                result = json.loads(recognizer.Result())
                text = result.get("text", "").lower()
                
                if text:
                    print(f"Real-time transcription: {text}")
                    transcript.append(text)
                    
                    # Process the text for filler words
                    detect_filler_words(text)
            
            # Analyze audio for vocal variety
            note = analyze_audio_chunk(data)
            
            # Track silence duration
            if note == "Silence":
                if silence_start_time is None:
                    silence_start_time = time.time()
                else:
                    silence_duration = time.time() - silence_start_time
                    if silence_duration >= SILENCE_THRESHOLD:
                        awkward_pauses += 1
                        silence_start_time = time.time()  # Reset to avoid counting the same pause multiple times
            else:
                silence_start_time = None
            
            # Update note history
            current_time = time.time()
            if note != current_note:
                if current_note is not None:
                    duration = current_time - current_note_start
                    if duration >= min_note_duration:
                        note_history.append((current_note, duration))
                current_note = note
                current_note_start = current_time
            
        except Exception as e:
            print(f"Audio recording error: {e}")
            continue
    
    # Get final result from Vosk
    if recognizer:
        final_result = json.loads(recognizer.FinalResult())
        final_text = final_result.get("text", "").lower()
        if final_text:
            print(f"Final transcription: {final_text}")
            transcript.append(final_text)
            detect_filler_words(final_text)
    
    # Close audio resources
    stream.stop_stream()
    stream.close()
    p.terminate()
    print("Audio recording stopped")

def generate_frames():
    global detector, recording, eye_contact_stats, current_note
    
    if detector is None:
        detector = EyeContactDetector()
    
    while True:
        frame, eye_contact = detector.detect_eye_contact()
        
        if recording:
            eye_contact_stats["total_frames"] += 1
            if eye_contact:
                eye_contact_stats["eye_contact_frames"] += 1
            eye_contact_stats["percentage"] = (eye_contact_stats["eye_contact_frames"] / 
                                             max(1, eye_contact_stats["total_frames"]) * 100)
        
        # Add overlay information to the frame
        elapsed_time = time.time() - start_time if recording else 0
        minutes = int(elapsed_time // 60)
        seconds = int(elapsed_time % 60)
        time_text = f"Time: {minutes:02d}:{seconds:02d}"
        
        # Display time overlay
        cv2.putText(frame, time_text, (10, 60), 
                  cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Display eye contact status
        if eye_contact:
            cv2.putText(frame, "Eye Contact", (10, 30), 
                      cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        else:
            cv2.putText(frame, "No Eye Contact", (10, 30), 
                      cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        # Display current note if available
        if current_note is not None:
            note_text = f"Current Note: {current_note}"
            cv2.putText(frame, note_text, 
                      (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

def analyze_vocal_variety():
    """Analyze vocal variety and generate a score"""
    if not note_history:
        return 0, "No vocal variety detected"
    
    # Filter out silence and short notes
    valid_notes = [(note, duration) for note, duration in note_history 
                  if note != "Silence" and duration >= min_note_duration]
    
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
        note_index = note_names.index(note_name)
        return note_index + (octave * 12)
    
    try:
        midi_notes = [note_to_midi(note) for note in notes]
        note_range = max(midi_notes) - min(midi_notes)
    except (ValueError, IndexError):
        return 0, "Error analyzing vocal range"
    
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

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/start_recording')
def start_recording():
    global recording, audio_recording, eye_contact_stats, audio_thread
    global transcript, filler_words, note_history, awkward_pauses, start_time
    
    # Reset all stats
    recording = True
    audio_recording = True
    start_time = time.time()
    
    eye_contact_stats = {
        "total_frames": 0,
        "eye_contact_frames": 0,
        "percentage": 0
    }
    
    transcript = []
    filler_words = {key: 0 for key in filler_words}
    note_history = []
    awkward_pauses = 0
    
    # Start audio recording thread
    audio_thread = threading.Thread(target=record_audio)
    audio_thread.start()
    
    return jsonify({"status": "success"})

@app.route('/stop_recording')
def stop_recording():
    global recording, audio_recording, audio_thread
    
    recording = False
    audio_recording = False
    
    # Wait for audio thread to finish
    if audio_thread and audio_thread.is_alive():
        audio_thread.join(timeout=2)
    
    # Generate feedback report
    report = generate_feedback_report()
    
    return jsonify({
        "status": "success",
        "stats": eye_contact_stats,
        "report": report
    })

@app.route('/get_stats')
def get_stats():
    return jsonify(eye_contact_stats)

def generate_feedback_report():
    """Generate comprehensive feedback report based on all metrics"""
    # Calculate duration
    duration = time.time() - start_time
    
    # Get eye contact percentage
    eye_contact_percent = eye_contact_stats["percentage"]
    
    # Calculate filler word stats
    total_fillers = sum(filler_words.values())
    filler_rate = total_fillers / max(1, duration / 60)  # Fillers per minute
    
    # Get vocal variety score
    vocal_score, vocal_feedback = analyze_vocal_variety()
    
    # Group filler words by category
    category_totals = {
        "Um-like sounds": sum(filler_words[cat] for cat in ["um"]),
        "Uh-like sounds": sum(filler_words[cat] for cat in ["uh"]),
        "Er-like sounds": sum(filler_words[cat] for cat in ["er"]),
        "Other filler phrases": sum(filler_words[cat] for cat in 
            ["like", "you_know", "basically", "actually", "sort_of", "kind_of", "i_mean", "right", "well"])
    }
    
    # Generate overall feedback
    feedback = []
    
    # Eye contact feedback
    if eye_contact_percent < 30:
        feedback.append("Work on maintaining more eye contact with your audience")
    elif eye_contact_percent < 70:
        feedback.append("Your eye contact is acceptable but could be improved")
    else:
        feedback.append("You maintain excellent eye contact")
    
    # Filler words feedback
    if filler_rate > 10:
        feedback.append("Try to significantly reduce filler words (um, uh, like, etc.)")
    elif filler_rate > 5:
        feedback.append("Work on reducing some of your filler words")
    else:
        feedback.append("You use minimal filler words, which is excellent")
        
    # Awkward pauses feedback
    if awkward_pauses > 3:
        feedback.append("You had several awkward pauses (>4s). Try to maintain a more consistent speaking pace")
    elif awkward_pauses > 0:
        feedback.append("You had a few awkward pauses.")
    else:
        feedback.append("You maintained a good speaking pace with no awkward pauses")
    
    # Add vocal variety feedback
    feedback.append(vocal_feedback)
    
    # Compile full report
    report = {
        "duration": {
            "seconds": duration,
            "minutes": duration / 60
        },
        "eye_contact": {
            "percentage": eye_contact_percent,
            "frames_with_contact": eye_contact_stats["eye_contact_frames"],
            "total_frames": eye_contact_stats["total_frames"]
        },
        "vocal_variety": {
            "score": vocal_score,
            "feedback": vocal_feedback
        },
        "verbal_analysis": {
            "total_fillers": total_fillers,
            "filler_rate": filler_rate,
            "awkward_pauses": awkward_pauses,
            "filler_categories": category_totals,
            "filler_details": {word: count for word, count in filler_words.items() if count > 0}
        },
        "transcript": " ".join(transcript),
        "feedback": feedback
    }
    
    return report

if __name__ == '__main__':
    app.run(debug=True)