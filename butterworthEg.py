# -*- coding: utf-8 -*-
"""
Created on Fri May 28 12:29:13 2021

File to test the design of 3rd order Butterworth filter

@author: Malachi
"""


from scipy.signal import butter, lfilter

def butter_bandpass(lowcut, highcut, fs, order):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='bandpass')
    return b, a


def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = lfilter(b, a, data)
    return y


def run():
    import numpy as np
    import matplotlib.pyplot as plt
    from scipy.signal import freqz

    # Sample rate and desired cutoff frequencies (in Hz).
    fs = 5000
    lowcut = 500
    highcut = 1250

    # Plot the frequency response for a few different orders.
    plt.figure(1)
    plt.clf()
    b, a = butter_bandpass(lowcut, highcut, fs, 3)
    w, h = freqz(b, a, worN=5000)
    plt.plot(w/np.pi, np.abs(h))
    plt.plot([0, 1], [np.sqrt(0.5), np.sqrt(0.5)],
             '--', label='sqrt(0.5)')
    plt.xlabel('Normalized Frequency (x Pi rad/sample) \n(1=Nyquist freq)')    #1/m ?
    plt.ylabel('Gain')
   
    plt.title("3rd Order Butterworth Frequency Response \n0.5km - 2km cutoff")
    plt.grid(True)
    plt.legend(loc='best')
    
    # Filter a noisy signal.
    T = 0.05
    nsamples = int(T * fs)
    t = np.linspace(0, T, nsamples, endpoint=False)
    a = 0.02
    f0 = .015
    x = 0.1 * np.sin(2 * np.pi * 1.2 * np.sqrt(t))
    x += 0.01 * np.cos(2 * np.pi * 312 * t + 0.1)
    x += a * np.cos(2 * np.pi * f0 * t + .11)
    x += 0.03 * np.cos(2 * np.pi * 2000 * t)
    plt.figure(2)
    plt.clf()
    plt.plot(t, x, label='Noisy signal')

    y = butter_bandpass_filter(x, lowcut, highcut, fs, order=6)
    plt.plot(t, y, label='Filtered signal (%g Hz)' % f0)
    plt.xlabel('time (seconds)')
    plt.hlines([-a, a], 0, T, linestyles='--')
    plt.grid(True)
    plt.axis('tight')
    plt.legend(loc='upper left')

    

    plt.show()


run()