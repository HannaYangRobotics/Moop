# Pi4 Wake Word Detection Setup Guide (USB Microphone)

This guide will help you set up the wake word detection system on your Raspberry Pi 4 with a USB microphone.

## Prerequisites

- Raspberry Pi 4 (8GB RAM recommended)
- USB microphone (plug-and-play)
- MicroSD card with Raspberry Pi OS (Bullseye or newer)
- Speaker or audio output device

## Step 1: Hardware Setup

### USB Microphone Connection

1. Plug your USB microphone into any available USB port on the Pi4.
2. No wiring or special configuration is needed.

## Step 2: Verify USB Microphone Detection

1. Open a terminal and run:
   ```bash
   arecord -l
   ```
2. You should see your USB microphone listed, e.g.:
   ```
   card 1: USB [USB Audio Device], device 0: USB Audio [USB Audio]
   ```

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

## Step 6: Test USB Microphone Setup

### Test USB Microphone

Run the provided test script to verify your USB microphone is working:

```bash
python test_audio.py
```

- The script will list all audio devices and attempt to auto-select your USB mic.
- It will record a short audio clip and save it as `test_recording.wav`.
- Play back the file to verify audio quality.

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

1. **USB mic not found:**
   - Check USB connection
   - Run `arecord -l` to verify detection
   - Try a different USB port
   - Reboot the Pi4

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
