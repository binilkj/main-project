from __future__ import print_function
import scipy.io.wavfile as wavfile
import scipy
import scipy.fftpack
import numpy as np
from matplotlib import pyplot as plt

rate, signal = wavfile.read("test.wav")
audio_length = len(signal.shape)
if audio_length == 2:
    signal = signal.sum(axis=1) / 2
N = signal.shape[0]
secs = N / float(rate)
Ts = 1.0/rate # sampling interval in time
time_vector = scipy.arange(0, secs, Ts) # time vector as scipy arange field / numpy.ndarray
FFT = abs(scipy.fft(signal))
FFT_side = FFT[range(N//2)] # one side FFT range
freqs = scipy.fftpack.fftfreq(signal.size, time_vector[1]-time_vector[0])
fft_freqs = np.array(freqs)
freqs_side = freqs[range(N//2)] # one side frequency range
fft_freqs_side = np.array(freqs_side)
plt.subplot(311)


plot_graph1 = plt.plot(time_vector, signal, "g") # plotting the signal
plt.xlabel('Time')
plt.ylabel('Amplitude')
plt.subplot(312)
plot_graph2 = plt.plot(freqs, FFT, "r") # plotting the complete fft spectrum
plt.xlabel('Frequency (Hz)')
plt.ylabel('Count dbl-sided')
#plt.figure(figsize=(20,20))
plt.show()
