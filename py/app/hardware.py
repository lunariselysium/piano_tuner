import sounddevice as sd
import numpy as np
from scipy.optimize import curve_fit
import threading
import time

# --- MOCK BLUETOOTH DEVICE ---
class TunerDevice:
    def __init__(self):
        self.connected = False

    def connect(self, device_name):
        print(f"[BT] Connecting to {device_name}...")
        time.sleep(1) # Fake delay
        self.connected = True
        print(f"[BT] Connected!")
        return True

    def send_command(self, command, value=None):
        if not self.connected:
            print("[BT Error] Not connected")
            return
        
        # Command logic
        print(f"[BT SENT] CMD: {command} | VAL: {value}")

# --- AUDIO ENGINE ---
class AudioEngine:
    def __init__(self):
        self.stream = None
        self.rms = 0
        self.detected_freq = 0
        self.note_detected = False
        self.is_listening = False
        self.sensitivity = 5000000
        
        # Result containers
        self.last_analysis_result = None
        self.ready_for_analysis = False

    def get_devices(self):
        return sd.query_devices()

    def start_stream(self, device_id, rate=48000):
        if self.stream:
            self.stop_stream()
        
        try:
            self.stream = sd.InputStream(
                device=device_id,
                channels=1,
                samplerate=rate,
                blocksize=4096,
                callback=self._audio_callback
            )
            self.stream.start()
            self.is_listening = True
        except Exception as e:
            print(f"Audio Error: {e}")

    def stop_stream(self):
        if self.stream:
            self.stream.stop()
            self.stream.close()
            self.stream = None
            self.is_listening = False

    def _audio_callback(self, indata, frames, time, status):
        # Calculate RMS for UI visualization
        self.rms = np.sqrt(np.mean(indata[:, 0].astype(np.float64)**2))

        # Simple trigger logic
        if self.rms > self.sensitivity and not self.note_detected:
            self.note_detected = True
            # Launch analysis in a separate thread so audio doesn't stutter
            threading.Thread(target=self._analyze_worker, args=(indata.copy(), 48000)).start()
        
        if self.rms < (self.sensitivity / 2):
            self.note_detected = False

    def _analyze_worker(self, indata, rate):
        # Your original logic adapted here
        window = np.hanning(len(indata))
        indata_windowed = indata[:, 0] * window
        fft_spectrum = np.fft.rfft(indata_windowed)
        fft_freq = np.fft.rfftfreq(len(indata_windowed), 1.0 / rate)
        fft_magnitude = np.abs(fft_spectrum)
        
        peak_index = np.argmax(fft_magnitude)
        fundamental_freq = fft_freq[peak_index]
        
        # (Simplified Inharmonicity Calculation for UI Demo Speed)
        # In the real version, paste your full `analyze_note` logic here
        # and store B value.
        
        self.detected_freq = fundamental_freq
        
        # Store result to be picked up by UI
        self.last_analysis_result = {
            "freq": fundamental_freq,
            "partials": [], 
            "B": 0.0001 # Mock B
        }
        self.ready_for_analysis = True