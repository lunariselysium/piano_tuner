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
        
        # Lower sensitivity slightly for complex waveforms
        self.sensitivity = 2000000 
        
        self.analyzing = False
        self.last_analysis_result = None
        self.ready_for_analysis = False

    def get_devices(self):
        return sd.query_devices()

    def start_stream(self, device_id, rate=48000):
        if self.stream: self.stop_stream()
        try:
            self.stream = sd.InputStream(
                device=device_id, channels=1, samplerate=rate,
                blocksize=8192, # Increased buffer for better bass resolution
                dtype='int32',
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
        
        # RMS Calculation
        self.rms = np.sqrt(np.mean(indata[:, 0].astype(np.float64)**2))

        if self.rms > self.sensitivity and not self.analyzing:
            self.analyzing = True
            # Wait a tiny bit (50ms) to skip the hammer attack noise
            # This makes the "sustained" tone clearer
            threading.Timer(0.05, self._trigger_worker, [indata.copy(), 48000]).start()

    def _trigger_worker(self, data, rate):
        threading.Thread(target=self._analyze_worker, args=(data, rate)).start()

    def _analyze_worker(self, indata, rate):
        try:
            # Prepare Data: Hanning Window
            data = indata[:, 0]
            window = np.hanning(len(data))
            data = data * window
            
            # FFT
            fft_spectrum = np.abs(np.fft.rfft(data))
            freqs = np.fft.rfftfreq(len(data), 1.0 / rate)
            
            # --- IMPROVED: Harmonic Product Spectrum (HPS) ---
            # This multiplies the spectrum by downsampled versions of itself.
            # Harmonics line up at the fundamental.
            hps_spec = list(fft_spectrum)
            num_hps = 4 # Number of harmonics to consider
            
            for h in range(2, num_hps + 1):
                # Downsample
                decimated = fft_spectrum[::h]
                # Multiply (pad with zeros to match length)
                hps_spec[:len(decimated)] *= decimated
            
            # Find the peak in the HPS spectrum (not the raw spectrum)
            # We ignore very low frequencies (< 20Hz) to avoid DC offset/rumble
            start_index = int(20 / (rate / len(data))) 
            peak_index = np.argmax(hps_spec[start_index:]) + start_index
            
            # --- PARABOLIC INTERPOLATION ---
            # Refine the peak estimate to be more precise than just the bin width
            if 0 < peak_index < len(hps_spec) - 1:
                y0, y1, y2 = hps_spec[peak_index-1], hps_spec[peak_index], hps_spec[peak_index+1]
                denom = (y0 - 2 * y1 + y2)
                if denom != 0:
                    offset = (y0 - y2) / (2 * denom)
                    peak_freq = freqs[peak_index] + (offset * (freqs[1] - freqs[0]))
                else:
                    peak_freq = freqs[peak_index]
            else:
                peak_freq = freqs[peak_index]

            # --- PARTIALS ANALYSIS (For B Value) ---
            # We look for partials based on our sophisticated peak_freq
            partials_freqs = [peak_freq]
            
            # Scan for partials in the RAW spectrum (not HPS)
            for n in range(2, 6):
                expected = peak_freq * n
                search_width = 30 # Hz
                
                # Find indices in raw spectrum near expected harmonic
                idx_min = int((expected - search_width) / (rate / len(data)))
                idx_max = int((expected + search_width) / (rate / len(data)))
                
                if idx_max < len(fft_spectrum):
                    local_peak = np.argmax(fft_spectrum[idx_min:idx_max]) + idx_min
                    partials_freqs.append(freqs[local_peak])

            # --- CALCULATE B ---
            inharmonicity_B = 0.0001
            if len(partials_freqs) > 2:
                n_vals = np.arange(1, len(partials_freqs) + 1)
                measured_ratios = []
                valid_n = []
                f1 = partials_freqs[0]

                if f1 > 20: 
                    for i, fn in enumerate(partials_freqs):
                        n = i + 1
                        measured_ratios.append(fn / (n * f1))
                        valid_n.append(n)
                    try:
                        params, _ = curve_fit(inharmonicity_func, valid_n, measured_ratios, p0=[0.0001])
                        inharmonicity_B = abs(params[0])
                    except:
                        pass

            # Output
            self.last_analysis_result = {
                "freq": peak_freq,
                "B": inharmonicity_B
            }
            self.ready_for_analysis = True
            
            # Cooldown
            time.sleep(0.4) 
            
        except Exception as e:
            print(f"Analysis Error: {e}")
        finally:
            self.analyzing = False