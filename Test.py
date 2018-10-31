import matplotlib.pyplot as plt
import numpy as np
import wave
import sys
import scipy
import scipy.io.wavfile
import os
#import matplotlib.pyplot as plt2
#spf = wave.open('/Users/vijayanthtummala/PycharmProjects/AITest/training-a/a0006.wav','r')

spf = wave.open('normal.wav','r')

#Extract Raw Audio from Wav File
signal = spf.readframes(-1)
signal = np.fromstring(signal, 'Int16')

print (signal)
#If Stereo
#if spf.getnchannels() == 2:
#    print ('Just mono files')
#    sys.exit(0)

plt.figure(1)
plt.title('Signal Wave...')
plt.plot(signal)
plt.show()


#sample_rate, song_array = scipy.io.wavfile.read("/Users/vijayanthtummala/PycharmProjects/AITest/training-a/a0006.wav")
#fft_features = abs(scipy.fft(song_array[:10000]))
#base_fn, ext = os.path.splitext("normal.wav")
#data_fn = base_fn + ".fft"
#np.save(data_fn, fft_features)

#print(fft_features)


#plt.plot(fft_features)
#plt.show()

#[Fs, x] = audioBasicIO.readAudioFile("normal.wav");
#F, f_names = audioFeatureExtraction.stFeatureExtraction(x[:,0], Fs, 0.050*Fs, 0.025*Fs)

#plt.plot(F)
#plt.show()

#plt.subplot(2,1,1); plt.plot(F[0,:]); plt.xlabel('Frame no'); plt.ylabel(f_names[0]);
#plt.subplot(2,1,2); plt.plot(F[1,:]); plt.xlabel('Frame no'); plt.ylabel(f_names[1]); plt.show()