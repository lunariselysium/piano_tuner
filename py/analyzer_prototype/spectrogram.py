import sounddevice as sd
import numpy as np
from scipy.fft import fft, fftfreq
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Parameters
duration = 0.1  # seconds for each chunk
sample_rate = 44100  # Hz
channels = 1
device_id = 3  # Default input device

# Figure setup
fig, ax = plt.subplots()
ax.set_xlim(0, sample_rate / 2)
ax.set_ylim(0, 1)
ax.set_xlabel('Frequency (Hz)')
ax.set_ylabel('Amplitude')
line, = ax.plot([], [], lw=2)

# Buffer for audio data
audio_buffer = np.zeros(int(duration * sample_rate * channels))

def audio_callback(indata, frames, time, status):
    global audio_buffer
    audio_buffer = indata[:, 0]  # Assuming mono

def init():
    line.set_data([], [])
    return line,

def update(frame):
    # Compute FFT
    N = len(audio_buffer)
    yf = fft(audio_buffer)
    xf = fftfreq(N, 1 / sample_rate)[:N//2]
    amplitudes = 2.0 / N * np.abs(yf[:N//2])
    
    # Normalize for display
    amplitudes = amplitudes / np.max(amplitudes) if np.max(amplitudes) > 0 else amplitudes
    
    line.set_data(xf, amplitudes)
    return line,

print(sd.query_devices())
# Start audio stream
stream = sd.InputStream(samplerate=sample_rate, channels=channels, device=device_id,
                        callback=audio_callback, blocksize=int(duration * sample_rate))
stream.start()

# Animate
ani = FuncAnimation(fig, update, init_func=init, blit=True, interval=50)

plt.show()

# Stop stream when done (though plt.show() blocks)
stream.stop()
stream.close()