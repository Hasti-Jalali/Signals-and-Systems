from os.path import dirname, join as pjoin
from scipy.io import wavfile
import scipy.io
import numpy as np
from matplotlib import pyplot as plt

def wavefile_to_waveform(wave_name, sample):
    data_dir = 'waves/'
    wav_fname = pjoin(data_dir, wave_name)
    samplerate, data = wavfile.read(wav_fname)
    length = data.shape[0] / samplerate
    print(f"length = {length}s")
    # return length,data
    return data[0:sample]

def calc_ak(x,T, k):
    w0 = 2*np.pi / T
    b = []
    c = []
    a = []
    for k in range(0,k):
        b1 = 0
        c1 = 0
        for t in range(0,int(T-1)):
            b1 = b1 + x[t] * np.cos(k*w0*t)
            c1 = c1 + x[t] * np.sin(k*w0*t)
        b.append(b1/T)
        c.append(c1/T)
        a.append(abs(complex(b[k], -c[k])))

    return a


# fig = plt.figure()
# ax1 = fig.add_subplot(5,2,1)
# ax2 = fig.add_subplot(5,2,3)
# ax3 = fig.add_subplot(5,2,5)
# ax4 = fig.add_subplot(5,2,7)
# ax5 = fig.add_subplot(5,2,9)
#
# ax6 = fig.add_subplot(5,2,2)
# ax7 = fig.add_subplot(5,2,4)
# ax8 = fig.add_subplot(5,2,6)
# ax9 = fig.add_subplot(5,2,8)
# ax10 = fig.add_subplot(5,2,10)

sample = 500
# length1,y1 = wavefile_to_waveform('wave1.wav')
# x1 = np.linspace(0., length1, y1.shape[0])
x1 = np.linspace(0,499,500)
y1 = wavefile_to_waveform('wave1.wav',sample)
# ax1.plot(x1,y1)
plt.plot(x1,y1)
plt.title('WAVE1')
plt.show()

# length2, y2 = wavefile_to_waveform('wave2.wav')
# x2 = np.linspace(0., length2, y2.shape[0])
x2 = np.linspace(0,499,500)
y2 = wavefile_to_waveform('wave2.wav',sample)
# ax2.plot(x2,y2)
plt.plot(x2,y2)
plt.title('WAVE2')
plt.show()

# length3, y3 = wavefile_to_waveform('wave3.wav')
# x3 = np.linspace(0., length3, y3.shape[0])
x3 = np.linspace(0,499,500)
y3 = wavefile_to_waveform('wave3.wav',sample)
# ax3.plot(x3,y3)
plt.plot(x3,y3)
plt.title('WAVE3')
plt.show()

# length4, y4 = wavefile_to_waveform('wave4.wav')
# x4 = np.linspace(0., length4, y4.shape[0])
x4 = np.linspace(0,499,500)
y4 = wavefile_to_waveform('wave4.wav',sample)
# ax4.plot(x4,y4)
plt.plot(x4,y4)
plt.title('WAVE4')
plt.show()

x5 = np.linspace(0,303398,30399)
y5 = wavefile_to_waveform('bonus.wav',30399)
# ax5.plot(x5,y5)
plt.plot(x5,y5)
plt.title('BONUS')
plt.show()


T = 100
K = 11
X = np.linspace(0, 44100, K)

a1 = calc_ak(y1,T,K)
# ax6.stem(X,a1)
plt.stem(X,a1)
plt.title('|ak| WAVE1')
plt.xlabel('Frequncy [F]')
plt.show()
#
a2 = calc_ak(y2,T,K)
# ax7.stem(X,a2)
plt.stem(X,a2)
plt.title('|ak| WAVE2')
plt.xlabel('Frequncy [F]')
plt.show()

a3 = calc_ak(y3,T,K)
# ax8.stem(X,a3)
plt.stem(X,a3)
plt.title('|ak| WAVE3')
plt.xlabel('Frequncy [F]')
plt.show()

a4 = calc_ak(y4,T,K)
# ax9.stem(X,a4)
plt.stem(X,a4)
plt.title('|ak| WAVE4')
plt.xlabel('Frequncy [F]')
plt.show()

T1 = 10113
F1 = 1/T1
X1 = np.linspace(0, 176400, 100)
a5 = calc_ak(y5,T1,100)
# ax10.stem(X1,a5)
plt.stem(X1,a5)
plt.title('|ak| BONUS')
plt.xlabel('Frequncy [F]')
plt.show()
#
# plt.show()