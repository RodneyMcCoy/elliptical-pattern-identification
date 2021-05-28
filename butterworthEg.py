# -*- coding: utf-8 -*-
"""
Created on Fri May 28 12:29:13 2021

@author: Malachi
"""


from scipy.signal import butter, lfilter

def butter_bandpass(lowcut, highcut, fs, order):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='bandpass')
    return b, a





def run():
    import numpy as np
    import matplotlib.pyplot as plt
    from scipy.signal import freqz

    # Sample rate and desired cutoff frequencies (in Hz).
    fs = 1/5
    lowcut = 1/500
    highcut = 1/2000

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
    plt.xlim([0,.1])
    plt.title("3rd Order Butterworth Frequency Response \n0.5km - 2km cutoff")
    plt.grid(True)
    plt.legend(loc='best')

    

    plt.show()


run()