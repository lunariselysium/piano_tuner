import sounddevice as sd
import numpy as np
from scipy.optimize import curve_fit

# --- CONFIGURATION ---
DEVICE_ID = 3         # Let the user choose the device
CHANNELS = 1          # Mono channel is sufficient for frequency analysis
RATE = 48000          # Your mic's native sample rate
CHUNK = 4096          # Increased buffer size for better frequency resolution

# This is a starting point for 24-bit audio packed in a 32-bit int.
# You will need to speak into the mic and adjust this threshold.
RMS_SENSITIVITY = 8000000 

# --- New Configuration for Note Analysis ---
NUM_PARTIALS = 5  # How many partials to find
NOTE_DETECTED = False # Flag to control analysis

def get_note_name(frequency):
    """Converts a frequency in Hz to its corresponding musical note name."""
    if frequency is None or frequency == 0:
        return "N/A"
    
    A4 = 440
    C0 = A4 * pow(2, -4.75)
    note_names = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
    
    h = round(12 * np.log2(frequency / C0))
    octave = h // 12
    note_index = h % 12
    return f"{note_names[note_index]}{octave}"


def inharmonicity_func(n, B):
    """
    Theoretical model for the frequency of the nth partial.
    This is used for the curve_fit function.
    f_n = n * f_1 * sqrt(1 + B * n^2)
    We will fit for 'B' so the function is simplified to just the inharmonicity part.
    """
    # We are fitting the ratio f_n / (n * f_1)
    return np.sqrt(1 + B * n**2)

def analyze_note(indata, rate):
    """
    Analyzes the audio data to find the fundamental frequency, its partials,
    and calculates the inharmonicity coefficient.
    """
    global NOTE_DETECTED
    NOTE_DETECTED = True # Set flag to prevent re-triggering while analyzing

    # Apply a window function to the data to reduce spectral leakage
    window = np.hanning(len(indata))
    indata_windowed = indata[:, 0] * window

    # --- Frequency Analysis (FFT) ---
    # Perform the FFT
    fft_spectrum = np.fft.rfft(indata_windowed)
    fft_freq = np.fft.rfftfreq(len(indata_windowed), 1.0 / rate)
    fft_magnitude = np.abs(fft_spectrum)

    # Find the peak in the FFT, which corresponds to the fundamental frequency
    peak_index = np.argmax(fft_magnitude)
    fundamental_freq = fft_freq[peak_index]
    
    # --- Find Partials ---
    partials_freqs = [fundamental_freq]
    partial_indices = [peak_index]

    for n in range(2, NUM_PARTIALS + 1):
        # Estimate the frequency of the next partial
        expected_freq = fundamental_freq * n
        
        # Find the closest frequency bin in the FFT
        search_range = 50 # Hz, search within this range of the expected frequency
        min_freq = expected_freq - search_range
        max_freq = expected_freq + search_range
        
        # Find indices within the search range
        indices_in_range = np.where((fft_freq >= min_freq) & (fft_freq <= max_freq))
        
        if indices_in_range[0].size > 0:
            # Find the peak magnitude within the search range
            peak_partial_index = indices_in_range[0][np.argmax(fft_magnitude[indices_in_range])]
            
            if peak_partial_index not in partial_indices:
                partials_freqs.append(fft_freq[peak_partial_index])
                partial_indices.append(peak_partial_index)

    # --- Calculate Inharmonicity Coefficient 'B' ---
    inharmonicity_coeff_B = None
    if len(partials_freqs) > 2:
        n_values = np.arange(1, len(partials_freqs) + 1)
        
        # We need to match the partial number to the correct frequency
        measured_ratios = []
        valid_n = []
        
        # Use the first detected frequency as the fundamental
        f1 = partials_freqs[0] 
        
        for i, fn in enumerate(partials_freqs):
            n = i + 1
            if fn > 0 and f1 > 0:
                 measured_ratios.append(fn / (n * f1))
                 valid_n.append(n)

        measured_ratios = np.array(measured_ratios)
        valid_n = np.array(valid_n)

        try:
            # Use curve_fit to find the best 'B'
            params, _ = curve_fit(inharmonicity_func, valid_n, measured_ratios, p0=[0.0001])
            inharmonicity_coeff_B = params[0]
        except RuntimeError:
            inharmonicity_coeff_B = "Calculation failed"


    # --- Display Results ---
    print("\n\n--- Note Analysis ---")
    print(f"Fundamental Frequency: {fundamental_freq:.2f} Hz ({get_note_name(fundamental_freq)})")
    
    print("\nPartials Found:")
    for i, freq in enumerate(partials_freqs):
        print(f"  Partial {i+1}: {freq:.2f} Hz")
    
    print(f"\nInharmonicity Coefficient (B): {inharmonicity_coeff_B}")
    print("\nListening for next note...")


def audio_callback(indata, frames, time, status):
    """This function is called for each audio chunk."""
    global NOTE_DETECTED
    
    if status:
        print(f"Status: {status}", flush=True)
    
    # Calculate RMS on the first channel to detect activity
    rms = np.sqrt(np.mean(indata[:, 0].astype(np.float64)**2))
    
    bar_length = int(rms / 500000) 
    bar = '#' * min(bar_length, 40)
    
    if rms > RMS_SENSITIVITY and not NOTE_DETECTED:
        status_text = "NOTE DETECTED! Analyzing... 🎹"
        output = f"\r[ {bar:<40} ] RMS: {rms:10.0f} | \033[92m{status_text}\033[0m"
        print(output)
        analyze_note(indata, RATE)
    elif not NOTE_DETECTED:
        status_text = "Silence"
        output = f"\r[ {bar:<40} ] RMS: {rms:10.0f} | {status_text}"
        print(output, end='')

    if rms < (RMS_SENSITIVITY / 2):
        NOTE_DETECTED = False # Reset the flag when it's quiet


if __name__ == "__main__":
    print("\n--- Available Audio Devices ---")
    print(sd.query_devices())
    print("-----------------------------\n")

    device_id_input = input("Enter the Device ID for your USB Mic (e.g., the DJI MIC): ")
    if device_id_input.isdigit():
        DEVICE_ID = int(device_id_input)
    
    print(f"\n--- Piano Note Analyzer (RMS > {RMS_SENSITIVITY} triggers analysis) ---")
    print("Play a single, clear piano note. Press Ctrl+C to stop.\n")

    try:
        with sd.InputStream(
            device=DEVICE_ID,
            channels=CHANNELS,
            samplerate=RATE,
            blocksize=CHUNK,
            dtype='int32',
            callback=audio_callback
        ):
            while True:
                sd.sleep(1000)
                    
    except KeyboardInterrupt:
        print("\nStopping...")
    except Exception as e:
        print(f"\nAn error occurred: {e}")