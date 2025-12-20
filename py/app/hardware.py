import sounddevice as sd
import numpy as np
from scipy.optimize import curve_fit
import threading
import time
import asyncio
from bleak import BleakScanner, BleakClient

# --- BLUETOOTH DEVICE (BLEAK + MOCK) ---
SERVICE_UUID = "4fafc201-1fb5-459e-8fcc-c5c9c331914b"
CHAR_UUID = "beb5483e-36e1-4688-b7f5-ea07361b26a8"
MOCK_ADDR = "00:00:00:00:00:00"

class TunerDevice:
    def __init__(self):
        self.client = None
        self.connected = False
        self.is_mock = False
        self.loop = asyncio.new_event_loop()
        # Start asyncio loop in a separate thread to not block Kivy
        self._thread = threading.Thread(target=self._start_loop, daemon=True)
        self._thread.start()

    def _start_loop(self):
        asyncio.set_event_loop(self.loop)
        self.loop.run_forever()

    def _run_async(self, coro):
        """Helper to submit async tasks to the background loop"""
        return asyncio.run_coroutine_threadsafe(coro, self.loop)

    def scan_devices(self, callback):
        """Scans for devices and returns list to callback"""
        async def _scan():
            print("[BT] Scanning started...")
            results = []
            
            # 1. Always add the Mock Device first (for debugging)
            results.append(("DEBUG: Mock Tuner", MOCK_ADDR))

            # 2. Scan Real Devices
            try:
                devices = await BleakScanner.discover(timeout=5.0)
                
                # Robust sort: Prioritize device with our Service UUID
                def sort_key(d):
                    # Safety check: metadata might be missing on some OS/versions
                    uuids = d.metadata.get("uuids", []) if hasattr(d, "metadata") else []
                    return 0 if SERVICE_UUID in uuids else 1
                
                sorted_devs = sorted(devices, key=sort_key)
                
                for d in sorted_devs:
                    name = d.name or "Unknown"
                    # Avoid duplicates if the mock address somehow appears in real life
                    if d.address != MOCK_ADDR: 
                        results.append((name, d.address))
                        
            except Exception as e:
                print(f"[BT Scan Error] {e}")

            print(f"[BT] Scan complete. Found {len(results)} devices.")
            # 3. Return results to UI (Even if scan failed, we have the Mock device)
            callback(results)

        self._run_async(_scan())

    def connect(self, address, callback):
        # Reset state
        self.connected = False
        self.is_mock = False
        
        # --- MOCK CONNECTION PATH ---
        if address == MOCK_ADDR:
            print("[BT] Connecting to MOCK device...")
            # Simulate a small network delay
            def _mock_connect():
                import time
                time.sleep(0.5) 
                self.connected = True
                self.is_mock = True
                print("[BT] MOCK Connected!")
                callback(True)
            
            # Run in a thread so we don't block the UI thread
            threading.Thread(target=_mock_connect).start()
            return True

        # --- REAL CONNECTION PATH ---
        async def _connect():
            print(f"[BT] Connecting to {address}...")
            await asyncio.sleep(1.0)
            try:
                if self.client:
                    await self.client.disconnect()
                
                self.client = BleakClient(address, timeout=10.0)
                
                await self.client.connect()

                if self.client.is_connected:
                    self.connected = True
                    print("[BT] Connected successfully!")
                    
                    print("\n" + "="*40)
                    print("CONNECTED: Showing Device Map")
                    print("="*40)
                    
                    # Iterate through services
                    for service in self.client.services:
                        print(f"\n[Service] {service.description} (UUID: {service.uuid})")
                        
                        # Iterate through characteristics in this service
                        for char in service.characteristics:
                            props = ", ".join(char.properties)
                            print(f"  └── [Char] {char.description} (UUID: {char.uuid})")
                            print(f"             Properties: {props}")
                    
                    callback(True)
                else:
                    print("[BT] Connect call finished but is_connected is False")
                    callback(False)
            except Exception as e:
                print(f"[BT Error] {e}")
                callback(False)

        self._run_async(_connect())
        return True

    def send_command(self, command, value=None):
        if not self.connected:
            print("[BT Error] Not connected")
            return
        
        # --- MOCK SEND ---
        if self.is_mock:
            print(f"[BT MOCK SENT] CMD: {command}, 10 (Derived from Val: {value})")
            return

        # --- REAL SEND ---
        async def _send():
            try:
                if self.client and self.client.is_connected:
                    # Request: "CMD,10"
                    payload_str = f"{command},10"
                    payload = payload_str.encode('utf-8')
                    
                    print(f"[BT SENDING] {payload_str}")
                    await self.client.write_gatt_char(CHAR_UUID, payload, response=True)
                else:
                    print("[BT Error] Client disconnected unexpectedly.")
            except Exception as e:
                print(f"[BT Write Error] {e}")

        self._run_async(_send())

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