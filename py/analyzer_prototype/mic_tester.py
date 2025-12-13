import sounddevice as sd
import numpy as np

# --- CONFIGURATION ---
DEVICE_ID = None      # Let the user choose the device
CHANNELS = 2          # Your mic is a 2-channel device
RATE = 48000          # Your mic's native sample rate
CHUNK = 1024          # Buffer size

# This is a starting point for 24-bit audio packed in a 32-bit int.
# You will need to speak into the mic and adjust this threshold.
RMS_SENSITIVITY = 2000000 

def audio_callback(indata, frames, time, status):
    """This function is called for each audio chunk."""
    if status:
        print(f"Status: {status}", flush=True)
    
    # indata is a NumPy array of 32-bit integers.
    # The actual audio data occupies the most significant 24 bits.
    
    # Calculate RMS on the first channel (index 0).
    # We must use a larger data type for the calculation (float64) to prevent overflow
    # when squaring the large int32 values.
    rms = np.sqrt(np.mean(indata[:, 0].astype(np.float64)**2))
    
    # --- Visual Feedback ---
    # Adjust the scaling factor (e.g., 500000) based on the RMS values you see.
    bar_length = int(rms / 500000) 
    bar = '#' * min(bar_length, 40)
    
    if rms > RMS_SENSITIVITY:
        status_text = "NOISE DETECTED! 🔊"
        output = f"\r[ {bar:<40} ] RMS: {rms:10.0f} | \033[92m{status_text}\033[0m"
    else:
        status_text = "Silence"
        output = f"\r[ {bar:<40} ] RMS: {rms:10.0f} | {status_text}"
    
    print(output, end='')


if __name__ == "__main__":
    print("\n--- Available Audio Devices ---")
    print(sd.query_devices())
    print("-----------------------------\n")

    device_id_input = input("Enter the Device ID for your USB Mic (e.g., the DJI MIC): ")
    if device_id_input.isdigit():
        DEVICE_ID = int(device_id_input)
    
    print(f"\n--- Volume Level Check (RMS > {RMS_SENSITIVITY} means activity) ---")
    print("Speak or make noise into the mic. Press Ctrl+C to stop.\n")

    try:
        # Use a 'with' block to ensure the stream is always closed correctly.
        with sd.InputStream(
            device=DEVICE_ID,
            channels=CHANNELS,
            samplerate=RATE,
            blocksize=CHUNK,
            # THIS IS THE CORRECTED, CRITICAL LINE:
            dtype='int32',  # Request 32-bit integers to hold the 24-bit data.
            callback=audio_callback
        ):
            # The callback runs in the background. We just wait here.
            # You can press Ctrl+C at any time to exit.
            while True:
                sd.sleep(1000) # Sleep in the main thread to keep the script alive.
                    
    except KeyboardInterrupt:
        print("\nStopping...")
    except Exception as e:
        print(f"\nAn error occurred: {e}")
        