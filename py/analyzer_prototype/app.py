import numpy as np
import scipy.fft
from flask import Flask, render_template
from flask_socketio import SocketIO, emit

# --- Configuration ---
SAMPLE_RATE = 44100  # This must match the sample rate from the browser's AudioContext
NOTE_NAMES = ['A', 'A#', 'B', 'C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#']
A4_PITCH = 440.0
NUM_PARTIALS = 6

# --- Flask App Setup ---
app = Flask(__name__)
app.config['SECRET_KEY'] = 'secret!'
socketio = SocketIO(app)

# --- Core Analysis Functions ---

def freq_to_note(freq):
    """Converts a frequency in Hz to the nearest note name and its deviation in cents."""
    if freq is None or freq <= 0:
        return "...", 0, 0

    midi_note = 12 * (np.log2(freq / A4_PITCH)) + 69
    note_index = int(round(midi_note))
    standard_freq = A4_PITCH * 2**((note_index - 69) / 12)
    cents_deviation = 1200 * np.log2(freq / standard_freq)
    note_name = NOTE_NAMES[note_index % 12]
    octave = (note_index // 12) - 1
    
    return f"{note_name}{octave}", freq, cents_deviation

def find_fundamental_frequency(data):
    """Finds the fundamental frequency and calculates theoretical partials."""
    if np.max(np.abs(data)) < 0.01:  # Silence threshold
        return None, None

    window = np.hanning(len(data))
    data = data * window
    
    fft_spectrum = scipy.fft.fft(data)
    fft_freqs = scipy.fft.fftfreq(len(data), 1.0 / SAMPLE_RATE)

    positive_mask = fft_freqs > 0
    freqs = fft_freqs[positive_mask]
    mags = np.abs(fft_spectrum[positive_mask])

    min_freq_idx = np.searchsorted(freqs, 20)
    max_freq_idx = np.searchsorted(freqs, 5000)
    
    if len(mags[min_freq_idx:max_freq_idx]) == 0:
        return None, None

    peak_index = np.argmax(mags[min_freq_idx:max_freq_idx]) + min_freq_idx
    fundamental_freq = freqs[peak_index]
    partials = [fundamental_freq * i for i in range(1, NUM_PARTIALS + 1)]

    return (fundamental_freq, freqs.tolist(), mags.tolist()), partials

# --- Flask Routes and Socket.IO Events ---

@app.route('/')
def index():
    """Serve the main HTML file."""
    return render_template('index.html')

@socketio.on('connect')
def on_connect():
    print('Client connected')

@socketio.on('disconnect')
def on_disconnect():
    print('Client disconnected')

@socketio.on('process_audio')
def process_audio_chunk(data):
    """Receives an audio chunk from the browser, processes it, and sends back the results."""
    # Convert the received list of floats into a NumPy array for processing
    audio_data = np.array(data, dtype=np.float32)
    
    analysis_result, partials = find_fundamental_frequency(audio_data)
    
    payload = {'note': '...', 'frequency': '...', 'cents': 0, 'partials': []}
    if analysis_result:
        fundamental_freq, freqs, mags = analysis_result
        note, freq, cents = freq_to_note(fundamental_freq)
        
        # Limit data sent to frontend for performance
        index_limit = np.searchsorted(freqs, 5000)
        
        payload.update({
            'note': note,
            'frequency': f"{freq:.2f} Hz",
            'cents': cents,
            'waveform': audio_data.tolist(),
            'fft_freqs': freqs[:index_limit],
            'fft_mags': mags[:index_limit],
            'partials': partials
        })
    # Emit the analysis result back to the client on a different event name
    emit('analysis_result', payload)

if __name__ == '__main__':
    print("Starting server. Please open http://127.0.0.1:5000 in your browser.")
    # Use allow_unsafe_werkzeug for compatibility with Flask-SocketIO's threading model
    socketio.run(app, host='127.0.0.1', port=5000, allow_unsafe_werkzeug=True)