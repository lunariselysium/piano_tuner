import sounddevice as sd

fs=44100
duration = 10.5

sd.default.device = 33
sd.default.samplerate = fs
sd.default.channels = 2



print(sd.query_devices())

myrecording = sd.rec(int(duration * fs))
sd.wait()
print(myrecording)