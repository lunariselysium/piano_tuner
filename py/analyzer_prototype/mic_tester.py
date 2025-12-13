import pyaudio
import numpy as np

# --- CONFIGURATION ---
FORMAT = pyaudio.paInt16  # 16-bit resolution
CHANNELS = 1              # Mono input
RATE = 44100              # Sample rate
CHUNK = 1024              # Buffer size
RMS_SENSITIVITY = 1000    # Threshold for "Noise Detected" (adjust as needed)

def list_audio_devices():
    """Lists all available PyAudio input devices."""
    p = pyaudio.PyAudio()
    info = p.get_host_api_info_by_index(0)
    numdevices = info.get('deviceCount')
    
    print("--- Available Audio Devices ---")
    for i in range(0, numdevices):
        device_info = p.get_device_info_by_host_api_device_index(0, i)
        
        # We are only interested in input devices (microphones)
        if device_info.get('maxInputChannels') > 0:
            print(f"Device ID {i}: {device_info.get('name')}")
            print(f"  Max Input Channels: {device_info.get('maxInputChannels')}")
            
    print("-----------------------------\n")
    p.terminate()


def volume_checker(device_index=None):
    """
    Runs a simple volume checker (VU Meter) on a selected device.
    
    :param device_index: The ID of the input device to use. If None, PyAudio default is used.
    """
    p = pyaudio.PyAudio()

    # If device_index is not provided, PyAudio will use the system's default device.
    if device_index is not None:
        try:
            device_info = p.get_device_info_by_index(device_index)
            print(f"Attempting to use device: {device_info['name']}")
        except ValueError:
            print(f"Error: Device ID {device_index} not found. Using default device instead.")
            device_index = None

    stream = p.open(format=FORMAT,
                    channels=CHANNELS,
                    rate=RATE,
                    input=True,
                    frames_per_buffer=CHUNK,
                    input_device_index=device_index) # Use the specific device ID

    print(f"\n--- Volume Level Check (RMS > {RMS_SENSITIVITY} means activity) ---")
    print("Speak or make noise into the mic. Press Ctrl+C to stop.\n")

    try:
        while True:
            # Read audio data from the stream
            data = stream.read(CHUNK, exception_on_overflow=False)
            
            # Convert audio data from bytes to a numpy array of integers
            data_int = np.frombuffer(data, dtype=np.int16)
            
            # Calculate RMS (Root Mean Square) amplitude
            rms = np.sqrt(np.mean(data_int**2))
            
            # Simple volume bar and status
            bar_length = int(rms / 100) # Scale RMS value for a visual bar
            bar = '#' * min(bar_length, 40) # Max length of 40 characters
            
            if rms > RMS_SENSITIVITY:
                status = "NOISE DETECTED! 🔊"
                # Use ANSI color for emphasis (Green)
                output = f"\r[ {bar:<40} ] RMS: {rms:6.0f} | \033[92m{status}\033[0m"
            else:
                status = "Silence"
                output = f"\r[ {bar:<40} ] RMS: {rms:6.0f} | {status}"
            
            print(output, end='')
            
    except KeyboardInterrupt:
        print("\n\nStopping volume check...")
    finally:
        stream.stop_stream()
        stream.close()
        p.terminate()


if __name__ == "__main__":
    # 1. List all devices first
    list_audio_devices()
    
    # 2. Get user input for the device
    device_id_input = input("Enter the Device ID you want to test (or press Enter to use the default mic): ")
    
    if device_id_input.isdigit():
        device_id = int(device_id_input)
    else:
        device_id = None # Use default device
    
    # 3. Start the volume check
    volume_checker(device_index=device_id)