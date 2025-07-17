#!/usr/bin/env python3
"""
Wake Word Detection System for Pi4 with I2S Microphone
Detects "Hey Moop" and triggers audio recording for further processing
"""

import numpy as np
import librosa
import tensorflow as tf
import pyaudio
import threading
import time
import queue
import os
from collections import deque

class WakeWordDetector:
    def __init__(self, model_path="models/wake_word_detector_test.tflite", 
                 sample_rate=16000, duration=1.0, hop_length=512, n_mfcc=13,
                 buffer_duration=2.0, detection_threshold=0.5):
        """
        Initialize the wake word detector
        
        Args:
            model_path: Path to the TensorFlow Lite model
            sample_rate: Audio sample rate (Hz)
            duration: Duration of audio chunks to analyze (seconds)
            hop_length: Hop length for MFCC calculation
            n_mfcc: Number of MFCC coefficients
            buffer_duration: Duration of audio buffer to keep (seconds)
            detection_threshold: Threshold for wake word detection (0-1)
        """
        self.sample_rate = sample_rate
        self.duration = duration
        self.hop_length = hop_length
        self.n_mfcc = n_mfcc
        self.samples_per_chunk = int(sample_rate * duration)
        self.buffer_samples = int(sample_rate * buffer_duration)
        self.detection_threshold = detection_threshold
        
        # Audio buffer (rolling buffer for pre-trigger audio)
        self.audio_buffer = deque(maxlen=self.buffer_samples)
        
        # Load TensorFlow Lite model
        self.interpreter = self.load_tflite_model(model_path)
        
        # Audio processing
        self.audio_queue = queue.Queue()
        self.is_listening = False
        self.is_processing = False
        
        # I2S Audio setup
        self.audio = pyaudio.PyAudio()
        self.stream = None
        
        print(f"Wake Word Detector initialized:")
        print(f"  Sample rate: {sample_rate} Hz")
        print(f"  Chunk duration: {duration} seconds")
        print(f"  Buffer duration: {buffer_duration} seconds")
        print(f"  Detection threshold: {detection_threshold}")
    
    def load_tflite_model(self, model_path):
        """Load TensorFlow Lite model"""
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model not found: {model_path}")
        
        interpreter = tf.lite.Interpreter(model_path=model_path)
        interpreter.allocate_tensors()
        
        # Get input and output tensors
        self.input_details = interpreter.get_input_details()
        self.output_details = interpreter.get_output_details()
        
        print(f"Model loaded: {model_path}")
        print(f"Input shape: {self.input_details[0]['shape']}")
        print(f"Output shape: {self.output_details[0]['shape']}")
        
        return interpreter
    
    def setup_i2s_audio(self, device_name=None):
        """Setup I2S microphone input"""
        # Find I2S device
        device_index = self.find_i2s_device(device_name)
        if device_index is None:
            print("Warning: I2S device not found, using default input device")
            device_index = self.audio.get_default_input_device_info()['index']
        
        print(f"Using audio device index: {device_index}")
        
        # Open audio stream
        self.stream = self.audio.open(
            format=pyaudio.paFloat32,
            channels=1,
            rate=self.sample_rate,
            input=True,
            input_device_index=int(device_index),
            frames_per_buffer=1024,
            stream_callback=self.audio_callback
        )
        
        return device_index
    
    def find_i2s_device(self, device_name=None):
        """Find I2S audio device"""
        for i in range(self.audio.get_device_count()):
            device_info = self.audio.get_device_info_by_index(i)
            device_name_str = str(device_info['name'])
            device_name_lower = device_name_str.lower()
            
            # Look for I2S devices
            if any(keyword in device_name_lower for keyword in ['i2s', 'mic', 'microphone']):
                print(f"Found I2S device: {device_info['name']} (index: {i})")
                return i
            
            # If specific device name provided, match it
            if device_name and device_name.lower() in device_name_lower:
                print(f"Found specified device: {device_info['name']} (index: {i})")
                return i
        
        return None
    
    def audio_callback(self, in_data, frame_count, time_info, status):
        """Callback for audio stream - called in separate thread"""
        if status:
            print(f"Audio callback status: {status}")
        
        # Convert bytes to numpy array
        audio_data = np.frombuffer(in_data, dtype=np.float32)
        
        # Add to rolling buffer
        for sample in audio_data:
            self.audio_buffer.append(sample)
        
        # Add to processing queue
        if not self.audio_queue.full():
            self.audio_queue.put(audio_data.copy())
        
        return (None, pyaudio.paContinue)
    
    def preprocess_audio(self, audio_chunk):
        """Preprocess audio chunk for model input"""
        # Ensure we have enough samples
        if len(audio_chunk) < self.samples_per_chunk:
            # Pad with zeros if too short
            audio_chunk = np.pad(audio_chunk, (0, self.samples_per_chunk - len(audio_chunk)))
        else:
            # Trim if too long
            audio_chunk = audio_chunk[:self.samples_per_chunk]
        
        # Extract MFCC features
        mfcc = librosa.feature.mfcc(
            y=audio_chunk, 
            sr=self.sample_rate, 
            n_mfcc=self.n_mfcc, 
            hop_length=self.hop_length
        )
        
        # Normalize MFCC
        mfcc = (mfcc - np.mean(mfcc)) / (np.std(mfcc) + 1e-8)
        
        # Reshape for model input (add batch and channel dimensions)
        mfcc = mfcc.reshape(1, mfcc.shape[0], mfcc.shape[1], 1)
        
        return mfcc
    
    def detect_wake_word(self, audio_chunk):
        """Detect wake word in audio chunk"""
        try:
            # Preprocess audio
            mfcc = self.preprocess_audio(audio_chunk)
            
            # Set input tensor
            self.interpreter.set_tensor(self.input_details[0]['index'], mfcc.astype(np.float32))
            
            # Run inference
            self.interpreter.invoke()
            
            # Get output
            prediction = self.interpreter.get_tensor(self.output_details[0]['index'])[0][0]
            
            return prediction
            
        except Exception as e:
            print(f"Error in wake word detection: {e}")
            return 0.0
    
    def save_triggered_audio(self, post_trigger_duration=3.0):
        """Save audio buffer + post-trigger audio when wake word is detected"""
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        filename = f"triggered_audio_{timestamp}.wav"
        
        # Get buffer audio
        buffer_audio = np.array(list(self.audio_buffer))
        
        # Record additional audio after trigger
        additional_samples = int(self.sample_rate * post_trigger_duration)
        additional_audio = []
        
        print(f"Recording {post_trigger_duration} seconds of additional audio...")
        start_time = time.time()
        
        while len(additional_audio) < additional_samples and (time.time() - start_time) < post_trigger_duration + 1:
            if not self.audio_queue.empty():
                chunk = self.audio_queue.get()
                additional_audio.extend(chunk)
            else:
                time.sleep(0.01)
        
        additional_audio = np.array(additional_audio[:additional_samples])
        
        # Combine buffer and additional audio
        full_audio = np.concatenate([buffer_audio, additional_audio])
        
        # Save as WAV file
        import soundfile as sf
        sf.write(filename, full_audio, self.sample_rate)
        
        print(f"Triggered audio saved: {filename}")
        print(f"Total duration: {len(full_audio) / self.sample_rate:.2f} seconds")
        
        return filename
    
    def process_audio_chunks(self):
        """Process audio chunks for wake word detection"""
        print("Starting audio processing...")
        
        while self.is_listening:
            try:
                # Get audio chunk from queue
                audio_chunk = self.audio_queue.get(timeout=1.0)
                
                # Detect wake word
                prediction = self.detect_wake_word(audio_chunk)
                
                # Check if wake word detected
                if prediction > self.detection_threshold:
                    print(f"\n🎯 WAKE WORD DETECTED! (confidence: {prediction:.3f})")
                    print("Hey Moop detected! Starting command recording...")
                    
                    # Save triggered audio
                    audio_file = self.save_triggered_audio()
                    
                    # Here you would trigger the next steps in your workflow:
                    # 1. Send audio to STT
                    # 2. Process command
                    # 3. Send to ESP32
                    # 4. TTS response
                    
                    print("Ready for next wake word...")
                    
                # Optional: Print confidence periodically (for debugging)
                elif prediction > 0.1:  # Only print if there's some activity
                    print(f"Activity detected (confidence: {prediction:.3f})", end='\r')
                
            except queue.Empty:
                continue
            except Exception as e:
                print(f"Error processing audio: {e}")
                continue
    
    def start_listening(self):
        """Start listening for wake word"""
        if self.is_listening:
            print("Already listening...")
            return
        
        print("Starting wake word detection...")
        print("Say 'Hey Moop' to trigger the system!")
        print("Press Ctrl+C to stop")
        
        self.is_listening = True
        
        # Start audio stream
        if self.stream:
            self.stream.start_stream()
        
        # Start processing thread
        self.processing_thread = threading.Thread(target=self.process_audio_chunks)
        self.processing_thread.daemon = True
        self.processing_thread.start()
        
        try:
            # Keep main thread alive
            while self.is_listening:
                time.sleep(0.1)
        except KeyboardInterrupt:
            print("\nStopping wake word detection...")
            self.stop_listening()
    
    def stop_listening(self):
        """Stop listening for wake word"""
        self.is_listening = False
        
        if self.stream:
            self.stream.stop_stream()
            self.stream.close()
        
        if self.audio:
            self.audio.terminate()
        
        print("Wake word detection stopped")
    
    def __del__(self):
        """Cleanup"""
        self.stop_listening()

def main():
    """Main function"""
    print("Pi4 Wake Word Detection System")
    print("=" * 40)
    
    try:
        # Initialize detector
        detector = WakeWordDetector()
        
        # Setup I2S audio
        detector.setup_i2s_audio()
        
        # Start listening
        detector.start_listening()
        
    except Exception as e:
        print(f"Error: {e}")
        print("\nTroubleshooting tips:")
        print("1. Make sure your I2S microphone is properly connected")
        print("2. Check if the model file exists: models/wake_word_detector_test.tflite")
        print("3. Install required packages: pip install pyaudio librosa tensorflow soundfile")
        print("4. For I2S setup, you may need to configure /boot/config.txt")

if __name__ == "__main__":
    main() 