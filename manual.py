import matplotlib.pyplot as plt
import numpy
from scipy.fftpack import fft, fftfreq
import scipy
import wave


def readwav(filename):
    spf = wave.open('/Users/vijayanthtummala/PycharmProjects/AITest/small/normal__107_1305654946865_C1.wav', 'r')
   # assert (spf.getnchannels() == 1)  # mono
  #  assert (spf.getsampwidth() == 2)  # 16bit
    sample_rate = spf.getframerate()
    num_samples = spf.getnframes()
    spf.rewind()
    signal = spf.readframes(spf.getnframes())  # (-1)
    #print(signal)
    signal = numpy.fromstring(signal, 'Int16')
    print(signal)

    plt.plot(signal)
    plt.show()
    #signal = signal / 32768.0  # why not /= ??
    timeline = numpy.linspace(0, num_samples / sample_rate, num=num_samples)
    return (signal, timeline, num_samples, sample_rate)


def calc_normfft(signal, timeline, num_samples):
    FFT = scipy.absolute(fft(signal[:num_samples]))
    FFT_max = numpy.max(FFT)
    FFT /= FFT_max # normalize
    freqs = fftfreq(num_samples, timeline[1] - timeline[0])
    #print(freqs)
    return (FFT, freqs)


def plot_fft(fft, freqs, num_samples, tonic_freq, max_freq, title):
    plt.figure(figsize=(12, 6))
    plt.plot(freqs[:num_samples], FFT, marker='.')
    plt.xlim([0, max_freq])
    plt.title(title)
    # tonic_freq=123 # rather sharp!
    for i in range(60):
        plt.axvline(x=i * tonic_freq, color="g")
    plt.show()


# http://stackoverflow.com/questions/2459295/stft-and-istft-in-python
def stft(x, fs, framesz, hop):
    "short-term frequency transform"
    framesamp = int(framesz * fs)
    hopsamp = int(hop * fs)
    w = scipy.hamming(framesamp)
    X = scipy.array([scipy.fft(w * x[i:i + framesamp]) for i in range(0, len(x) - framesamp, hopsamp)])
    # freqs = fftfreq(framesamp, hop)
    return X  # ,freqs


def plot_stft(X, tonic, title):
    plt.figure(figsize=(12, 6))
    # X.T is transpose!
    plt.imshow(scipy.absolute(X.T), origin='lower', interpolation='nearest')
    # extent=[timeline[0], timeline[1],Xfreqs[0],Xfreqs[-1]])
    # plt.imshow(scipy.absolute(X.T), origin='lower', aspect='auto')
    # plt.imshow(scipy.absolute(X.T), origin='lower', aspect='auto', interpolation='nearest')
    plt.xlabel('Time')
    plt.ylabel('Frequency')
    plt.ylim([0, 50])
    plt.xlim([0,80])
    for n in range(1, 25):
        f = n * tonic
        plt.axhline(y=f, color="w")
    plt.title(title)
    plt.show()


files = ['/Users/vijayanthtummala/PycharmProjects/AITest/small/normal__107_1305654946865_C1.wav']
ROOT = ''

for f in files:
    cur_filename = ROOT + f
    (signal, timeline, num_samples, sample_rate) = readwav(cur_filename)
    (FFT, freqs) = calc_normfft(signal, timeline, num_samples)
    f = open("out.txt", "w")
    for line in [str(y) + ";" + str(x) + "\n" for (x, y) in zip(FFT.tolist(), freqs.tolist())]:
        f.write(line)
    f.close()
    plot_fft(FFT, freqs, num_samples, 110 + 13, 2000, f)  # 220+56
    plt.show()

framesz =0.22
hop = 0.11

for f in files:
    cur_filename = ROOT + f
    (signal, timeline, num_samples, sample_rate) = readwav(cur_filename)
    STFT = stft(signal, sample_rate, framesz, hop)
    #print(STFT)
    plot_stft(STFT,100,f)
    plt.plot(STFT)
    plt.title('.2')
    plt.show()


