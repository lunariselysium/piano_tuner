import pyaudio
import numpy as np
import time
from scipy.signal import find_peaks

# --- CONFIGURATION ---
RATE = 44100          # Sample rate (Hz)
CHUNK = 4096          # Buffer size (higher = more lag but better freq resolution)
THRESHOLD = 1000      # Amplitude threshold for "Activity Detection"
PEAK_HEIGHT = 50      # Min spectral magnitude to consider a partial
NUM_PARTIALS = 5      # How many partials/harmonics to display

# Note names mapping
NOTE_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]

def freq_to_note_name(freq):
    """
    Converts a frequency (Hz) to a Note Name (e.g., A4) and cents deviation.
    """
    if freq == 0: return "Unknown"
    
    # Formula to get MIDI number from freq: n = 12 * log2(f / 440) + 69
    h = 12 * np.log2(freq / 440)
    n = h + 69
    
    n_rounded = int(round(n))
    note_name = NOTE_NAMES[n_rounded % 12]
    octave = (n_rounded // 12) - 1
    
    # Calculate deviation in cents
    deviation = n - n_rounded
    cents = int(deviation * 100)
    
    return f"{note_name}{octave} ({cents:+}c)"

def parabolic_interpolation(magnitude_spectrum, peak_index):
    """
    Refines the peak frequency estimate.
    Standard FFT returns integer 'bins'. This estimates the float position 
    of the true peak between bins.
    """
    # Handle edge cases
    if peak_index <= 0 or peak_index >= len(magnitude_spectrum) - 1:
        return peak_index
        
    alpha = magnitude_spectrum[peak_index - 1]
    beta = magnitude_spectrum[peak_index]
    gamma = magnitude_spectrum[peak_index + 1]
    
    p = 0.5 * (alpha - gamma) / (alpha - 2 * beta + gamma)
    return peak_index + p

def main():
    # Initialize PyAudio
    p = pyaudio.PyAudio()

    # Open Stream
    stream = p.open(format=pyaudio.paInt16,
                    channels=1,
                    rate=RATE,
                    input=True,
                    frames_per_buffer=CHUNK)

    print("🎹 Listening for Piano... (Press Ctrl+C to stop)")
    print("-" * 60)

    # Hanning window to smooth the audio chunk (reduces spectral leakage)
    window = np.hanning(CHUNK)

    try:
        while True:
            # 1. Get Audio Data
            raw_data = stream.read(CHUNK, exception_on_overflow=False)
            data_int = np.frombuffer(raw_data, dtype=np.int16)
            
            # 2. Activity Detection (Gate)
            # Calculate RMS (Root Mean Square) amplitude
            rms = np.sqrt(np.mean(data_int**2))
            
            if rms > THRESHOLD:
                # 3. FFT (Fast Fourier Transform)
                # Apply window function and compute FFT
                fft_data = np.fft.rfft(data_int * window)
                mag_spectrum = np.abs(fft_data)
                
                # 4. Find Peaks (Partials)
                # find_peaks returns indices of peaks. 
                # distance=50 prevents picking peaks too close to each other
                peaks, properties = find_peaks(mag_spectrum, height=PEAK_HEIGHT, distance=50)
                
                if len(peaks) > 0:
                    # Sort peaks by magnitude (loudest first)
                    sorted_indices = np.argsort(properties['peak_heights'])[::-1]
                    strongest_peaks = peaks[sorted_indices]
                    
                    # 5. Identify the Fundamental (The Note)
                    # NOTE: On a piano, the fundamental is usually the loudest, 
                    # but for very low notes, the first harmonic might be louder.
                    # Here we assume the Loudest Peak is the fundamental for simplicity.
                    
                    fundamental_idx = strongest_peaks[0]
                    
                    # Refine frequency using parabolic interpolation
                    refined_bin = parabolic_interpolation(mag_spectrum, fundamental_idx)
                    fundamental_freq = refined_bin * RATE / CHUNK
                    
                    note_name = freq_to_note_name(fundamental_freq)

                    # 6. Identify Actual Partial Frequencies
                    # We take the top N loudest peaks found in the signal
                    partials = []
                    for i in range(min(len(strongest_peaks), NUM_PARTIALS)):
                        idx = strongest_peaks[i]
                        r_bin = parabolic_interpolation(mag_spectrum, idx)
                        freq = r_bin * RATE / CHUNK
                        partials.append(freq)
                    
                    # Sort partials by frequency for display (low to high)
                    partials.sort()

                    # 7. Output
                    print(f"Detected: \033[92m{note_name}\033[0m | Freq: {fundamental_freq:.2f}Hz")
                    print(f"   -> Top Partials: {[f'{p:.1f}Hz' for p in partials]}")
                    print("-" * 40)
            
            else:
                # Optional: small sleep to save CPU when silent
                time.sleep(0.01)

    except KeyboardInterrupt:
        print("\nStopping...")
    finally:
        stream.stop_stream()
        stream.close()
        p.terminate()

if __name__ == "__main__":
    main()