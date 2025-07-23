#!/usr/bin/env python3
"""
Simple audio test script for USB microphone on Pi4
Use this to verify your USB microphone is working before running the wake word detector
"""

import pyaudio
import wave
import time
import numpy as np

def test_audio_devices():
    """List all available audio devices"""
    p = pyaudio.PyAudio()
    
    print("Available audio devices:")
    print("=" * 50)
    
    for i in range(p.get_device_count()):
        info = p.get_device_info_by_index(i)
        print(f"  {i}: {info['name']}")
        print(f"      Max inputs: {info['maxInputChannels']}")
        print(f"      Max outputs: {info['maxOutputChannels']}")
        print(f"      Default sample rate: {info['defaultSampleRate']}")
        print()
    
    p.terminate()

def find_usb_mic_device():
    p = pyaudio.PyAudio()
    usb_device = None
    for i in range(p.get_device_count()):
        info = p.get_device_info_by_index(i)
        device_name = str(info['name']).lower()
        if 'usb' in device_name or 'mic' in device_name:
            usb_device = i
            print(f"Found USB mic: {info['name']} (index: {i})")
            break
    if usb_device is None:
        print("No USB mic found, using default input device")
        usb_device = p.get_default_input_device_info()['index']
    p.terminate()
    return usb_device

def record_audio_test(device_index, duration=5):
    """Record audio for testing"""
    # Audio parameters
    CHUNK = 1024
    FORMAT = pyaudio.paFloat32
    CHANNELS = 1
    RATE = 16000
    
    p = pyaudio.PyAudio()
    
    print(f"Recording {duration} seconds of audio...")
    print("Speak into the microphone...")
    
    # Open audio stream
    stream = p.open(format=FORMAT,
                    channels=CHANNELS,
                    rate=RATE,
                    input=True,
                    input_device_index=device_index,
                    frames_per_buffer=CHUNK)
    
    frames = []
    
    # Record audio
    for i in range(0, int(RATE / CHUNK * duration)):
        data = stream.read(CHUNK)
        frames.append(data)
        
        # Show progress
        if i % 10 == 0:
            print(f"Recording... {i * CHUNK / RATE:.1f}s / {duration}s")
    
    print("Recording complete!")
    
    stream.stop_stream()
    stream.close()
    p.terminate()
    
    return frames, RATE, CHANNELS, FORMAT

def save_audio(frames, rate, channels, format_type, filename="test_recording.wav"):
    """Save recorded audio to WAV file"""
    p = pyaudio.PyAudio()
    
    with wave.open(filename, 'wb') as wf:
        wf.setnchannels(channels)
        wf.setsampwidth(p.get_sample_size(format_type))
        wf.setframerate(rate)
        wf.writeframes(b''.join(frames))
    
    p.terminate()
    print(f"Audio saved as: {filename}")

def analyze_audio(frames, rate):
    """Basic audio analysis"""
    # Convert frames to numpy array
    audio_data = np.frombuffer(b''.join(frames), dtype=np.float32)
    
    print("\nAudio Analysis:")
    print("=" * 30)
    print(f"Duration: {len(audio_data) / rate:.2f} seconds")
    print(f"Sample rate: {rate} Hz")
    print(f"Number of samples: {len(audio_data)}")
    print(f"Min amplitude: {np.min(audio_data):.4f}")
    print(f"Max amplitude: {np.max(audio_data):.4f}")
    print(f"RMS amplitude: {np.sqrt(np.mean(audio_data**2)):.4f}")
    
    # Check for silence
    silence_threshold = 0.01
    silent_samples = np.sum(np.abs(audio_data) < silence_threshold)
    silence_percentage = (silent_samples / len(audio_data)) * 100
    
    print(f"Silence percentage: {silence_percentage:.1f}%")
    
    if silence_percentage > 80:
        print("⚠️  Warning: Audio appears to be mostly silent!")
        print("   Check microphone connections and volume levels.")
    elif silence_percentage < 20:
        print("✅ Audio levels look good!")
    else:
        print("ℹ️  Audio levels are moderate.")

def main():
    """Main test function"""
    print("Pi4 I2S Microphone Test")
    print("=" * 40)
    
    try:
        # Step 1: List all audio devices
        print("Step 1: Checking audio devices...")
        test_audio_devices()
        
        # Step 2: Find I2S device
        print("Step 2: Finding I2S device...")
        device_index = find_usb_mic_device()
        
        # Step 3: Record test audio
        print(f"\nStep 3: Recording test audio...")
        print("Press Enter to start recording...")
        input()
        
        frames, rate, channels, format_type = record_audio_test(device_index, duration=5)
        
        # Step 4: Analyze audio
        print("\nStep 4: Analyzing audio...")
        analyze_audio(frames, rate)
        
        # Step 5: Save audio
        print("\nStep 5: Saving audio...")
        save_audio(frames, rate, channels, format_type)
        
        print("\n✅ Test completed successfully!")
        print("\nNext steps:")
        print("1. Play back the test_recording.wav file to verify audio quality")
        print("2. If audio quality is good, you can run the wake word detector")
        print("3. If there are issues, check the troubleshooting guide")
        
    except Exception as e:
        print(f"\n❌ Error during test: {e}")
        print("\nTroubleshooting tips:")
        print("1. Check I2S microphone connections")
        print("2. Verify /boot/config.txt settings")
        print("3. Reboot the Pi4")
        print("4. Check if I2S device appears in 'arecord -l'")

if __name__ == "__main__":
    main() 
