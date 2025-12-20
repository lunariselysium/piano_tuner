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

# --- ANALYSIS HELPER FUNCTIONS ---
def get_note_name(frequency):
    if frequency is None or frequency == 0: return "N/A"
    A4 = 440
    C0 = A4 * pow(2, -4.75)
    note_names = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
    h = round(12 * np.log2(frequency / C0))
    octave = h // 12
    note_index = h % 12
    return f"{note_names[note_index]}{octave}"

def inharmonicity_func(n, B):
    return np.sqrt(1 + B * n**2)

# --- AUDIO ENGINE ---
class AudioEngine:
    def __init__(self):
        self.stream = None
        self.rms = 0
        self.is_listening = False
        
        # Default high sensitivity for 32-bit int mics
        self.sensitivity = 2000000 
        
        # State management
        self.analyzing = False
        self.last_analysis_result = None
        self.ready_for_analysis = False

    def get_devices(self):
        return sd.query_devices()

    def start_stream(self, device_id, rate=48000):
        if self.stream: self.stop_stream()
        try:
            # Note: dtype='int32' is standard for many USB interfaces
            self.stream = sd.InputStream(
                device=device_id, channels=1, samplerate=rate,
                blocksize=4096, dtype='int32',
                callback=self._audio_callback
            )
            self.stream.start()
            self.is_listening = True
        except Exception as e:
            print(f"Audio Start Error: {e}")

    def stop_stream(self):
        if self.stream:
            self.stream.stop()
            self.stream.close()
            self.stream = None
            self.is_listening = False

    def _audio_callback(self, indata, frames, time, status):
        if status: print(status)
        
        # Calculate RMS (Root Mean Square) for volume
        # We cast to float64 to avoid overflow during squaring
        self.rms = np.sqrt(np.mean(indata[:, 0].astype(np.float64)**2))

        # Trigger logic: 
        # 1. Must be louder than threshold
        # 2. Must not currently be analyzing (debouncing)
        if self.rms > self.sensitivity and not self.analyzing:
            self.analyzing = True
            # Copy data for processing so we don't block the audio thread
            data_copy = indata.copy()
            threading.Thread(target=self._analyze_worker, args=(data_copy, 48000)).start()

    def _analyze_worker(self, indata, rate):
        try:
            # --- 1. FFT Analysis ---
            window = np.hanning(len(indata))
            indata_windowed = indata[:, 0] * window
            
            fft_spectrum = np.fft.rfft(indata_windowed)
            fft_freq = np.fft.rfftfreq(len(indata_windowed), 1.0 / rate)
            fft_magnitude = np.abs(fft_spectrum)

            # Find Fundamental Frequency (Peak)
            peak_index = np.argmax(fft_magnitude)
            fundamental_freq = fft_freq[peak_index]

            # --- 2. Find Partials (Simplified for speed) ---
            partials_freqs = [fundamental_freq]
            NUM_PARTIALS = 5
            
            for n in range(2, NUM_PARTIALS + 1):
                expected = fundamental_freq * n
                search_range = 50 # Hz window
                
                # Mask for search range
                mask = (fft_freq >= expected - search_range) & (fft_freq <= expected + search_range)
                if np.any(mask):
                    # Find peak within this specific window
                    segment_mag = fft_magnitude[mask]
                    segment_freq = fft_freq[mask]
                    if len(segment_mag) > 0:
                        local_peak_idx = np.argmax(segment_mag)
                        partials_freqs.append(segment_freq[local_peak_idx])

            # --- 3. Calculate Inharmonicity (B) ---
            inharmonicity_B = 0.0001 # Default fallback
            
            if len(partials_freqs) > 2:
                n_vals = np.arange(1, len(partials_freqs) + 1)
                measured_ratios = []
                valid_n = []
                f1 = partials_freqs[0]

                if f1 > 10: # Basic noise filter
                    for i, fn in enumerate(partials_freqs):
                        n = i + 1
                        measured_ratios.append(fn / (n * f1))
                        valid_n.append(n)
                    
                    try:
                        params, _ = curve_fit(inharmonicity_func, valid_n, measured_ratios, p0=[0.0001])
                        inharmonicity_B = abs(params[0])
                    except:
                        pass # Keep default

            # --- 4. Post Results ---
            self.last_analysis_result = {
                "freq": fundamental_freq,
                "B": inharmonicity_B
            }
            self.ready_for_analysis = True
            
            # Short cooldown to prevent double-triggering on the same note resonance
            time.sleep(0.5) 
            
        except Exception as e:
            print(f"Analysis Error: {e}")
        finally:
            self.analyzing = False