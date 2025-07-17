# Pi4 Wake Word Detection Setup Guide

This guide will help you set up the wake word detection system on your Raspberry Pi 4 with an I2S microphone.

## Prerequisites

- Raspberry Pi 4 (8GB RAM recommended)
- I2S microphone (e.g., INMP441, SPH0645LM4H-B, etc.)
- MicroSD card with Raspberry Pi OS (Bullseye or newer)
- Speaker or audio output device

## Step 1: Hardware Setup

### I2S Microphone Connection

Connect your I2S microphone to the Pi4 GPIO pins:

| I2S Pin | Pi4 GPIO | Description |
|---------|----------|-------------|
| VCC     | 3.3V     | Power       |
| GND     | GND      | Ground      |
| BCLK    | GPIO 18  | Bit Clock   |
| LRCLK   | GPIO 19  | Left/Right Clock |
| SD      | GPIO 20  | Data        |

**Note:** Pin numbers may vary depending on your I2S microphone. Check your microphone's datasheet.

## Step 2: Enable I2S Audio

### Edit Boot Configuration

1. Open the boot configuration file:
   ```bash
   sudo nano /boot/config.txt
   ```

2. Add these lines at the end of the file:
   ```
   # Enable I2S
   dtoverlay=i2s-mmap
   
   # Enable I2S audio interface
   dtoverlay=googlevoicehat-soundcard
   
   # Alternative: For generic I2S microphone
   # dtoverlay=i2s-mmap
   # dtparam=i2s=on
   ```

3. Save and reboot:
   ```bash
   sudo reboot
   ```

### Verify I2S Setup

After reboot, check if I2S is enabled:
```bash
# Check if I2S device is recognized
ls /proc/asound/card*

# List audio devices
aplay -l
arecord -l
```

You should see your I2S microphone listed as an audio device.

## Step 3: Install System Dependencies

```bash
# Update system
sudo apt update && sudo apt upgrade -y

# Install audio dependencies
sudo apt install -y portaudio19-dev python3-pyaudio
sudo apt install -y libasound2-dev
sudo apt install -y libportaudio2 libportaudiocpp0

# Install Python dependencies
sudo apt install -y python3-pip python3-venv
sudo apt install -y python3-dev

# Install additional libraries for audio processing
sudo apt install -y libfftw3-dev libavcodec-dev libavformat-dev libavutil-dev
sudo apt install -y libswresample-dev libavfilter-dev
```

## Step 4: Create Python Virtual Environment

```bash
# Create virtual environment
python3 -m venv wake_word_env

# Activate virtual environment
source wake_word_env/bin/activate

# Upgrade pip
pip install --upgrade pip
```

## Step 5: Install Python Dependencies

```bash
# Install requirements
pip install -r requirements.txt
```

**Note:** If you encounter issues with TensorFlow installation, you may need to install the ARM64 version:
```bash
pip install tensorflow-aarch64
```

## Step 6: Test Audio Setup

### Test I2S Microphone

Create a simple test script to verify your I2S microphone is working:

```python
# test_audio.py
import pyaudio
import wave
import time

# Audio parameters
CHUNK = 1024
FORMAT = pyaudio.paFloat32
CHANNELS = 1
RATE = 16000
RECORD_SECONDS = 5

p = pyaudio.PyAudio()

# List available devices
print("Available audio devices:")
for i in range(p.get_device_count()):
    info = p.get_device_info_by_index(i)
    print(f"  {i}: {info['name']}")

# Find I2S device
i2s_device = None
for i in range(p.get_device_count()):
    info = p.get_device_info_by_index(i)
    if 'i2s' in info['name'].lower() or 'mic' in info['name'].lower():
        i2s_device = i
        break

if i2s_device is None:
    print("No I2S device found, using default input")
    i2s_device = p.get_default_input_device_info()['index']

print(f"Using device: {p.get_device_info_by_index(i2s_device)['name']}")

# Record audio
stream = p.open(format=FORMAT,
                channels=CHANNELS,
                rate=RATE,
                input=True,
                input_device_index=i2s_device,
                frames_per_buffer=CHUNK)

print("Recording...")
frames = []

for i in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
    data = stream.read(CHUNK)
    frames.append(data)

print("Done recording")

stream.stop_stream()
stream.close()
p.terminate()

# Save recording
with wave.open("test_recording.wav", 'wb') as wf:
    wf.setnchannels(CHANNELS)
    wf.setsampwidth(p.get_sample_size(FORMAT))
    wf.setframerate(RATE)
    wf.writeframes(b''.join(frames))

print("Test recording saved as test_recording.wav")
```

Run the test:
```bash
python test_audio.py
```

## Step 7: Test Wake Word Detection

### Basic Test

1. Make sure your model file exists:
   ```bash
   ls models/wake_word_detector_test.tflite
   ```

2. Run the wake word detector:
   ```bash
   python pi4_wake_word_detector.py
   ```

3. Say "Hey Moop" clearly into the microphone

4. Check if the system detects the wake word and saves audio files

### Troubleshooting

If you encounter issues:

1. **Audio device not found:**
   - Check I2S connections
   - Verify `/boot/config.txt` settings
   - Reboot and check `aplay -l` and `arecord -l`

2. **Model loading errors:**
   - Ensure the .tflite file exists in the models directory
   - Check file permissions

3. **Poor detection accuracy:**
   - The test model was trained with only 20 samples
   - Consider retraining with more data
   - Adjust the detection threshold in the code

4. **Audio quality issues:**
   - Check microphone positioning
   - Ensure proper power supply
   - Verify sample rate settings

## Step 8: Performance Optimization

### For Better Performance

1. **Overclock Pi4 (optional):**
   ```bash
   # Add to /boot/config.txt
   over_voltage=2
   arm_freq=2000
   ```

2. **Use SSD instead of SD card:**
   - Improves model loading speed
   - Better for audio file I/O

3. **Disable unnecessary services:**
   ```bash
   sudo systemctl disable bluetooth
   sudo systemctl disable wifi  # if not needed
   ```

## Next Steps

Once the wake word detection is working:

1. **Improve the model:** Collect more training data and retrain
2. **Add STT:** Integrate Vosk or Whisper for speech-to-text
3. **Add TTS:** Integrate eSpeak for text-to-speech
4. **Add ESP32 communication:** Set up serial communication
5. **Add cloud processing:** Integrate with OpenAI or other services

## Useful Commands

```bash
# Check CPU usage
htop

# Check memory usage
free -h

# Monitor audio devices
watch -n 1 'arecord -l'

# Check system temperature
vcgencmd measure_temp

# Monitor disk usage
df -h
```

## Support

If you encounter issues:

1. Check the troubleshooting section above
2. Verify all connections and configurations
3. Test with a simple audio recording first
4. Check system logs: `dmesg | grep -i audio`
5. Verify Python environment and dependencies 