#
# reference: https://github.com/bkvogel/griffin_lim
#

import math
import sys
import time
import numpy as np
import wave
import scipy
import scipy.signal
from pylab import *
import array
import os
import scipy.io.wavfile
import soundfile as sf

# Author: Brian K. Vogel
# brian.vogel@gmail.com


def fft_bin_to_hz(n_bin, sample_rate_hz, fft_size):
    """Convert FFT bin index to frequency in Hz.

    Args:
        n_bin (int or float): The FFT bin index.
        sample_rate_hz (int or float): The sample rate in Hz.
        fft_size (int or float): The FFT size.

    Returns:
        The value in Hz.
    """
    n_bin = float(n_bin)
    sample_rate_hz = float(sample_rate_hz)
    fft_size = float(fft_size)
    return n_bin*sample_rate_hz/(2.0*fft_size)


def hz_to_fft_bin(f_hz, sample_rate_hz, fft_size):
    """Convert frequency in Hz to FFT bin index.

    Args:
        f_hz (int or float): The frequency in Hz.
        sample_rate_hz (int or float): The sample rate in Hz.
        fft_size (int or float): The FFT size.

    Returns:
        The FFT bin index as an int.
    """
    f_hz = float(f_hz)
    sample_rate_hz = float(sample_rate_hz)
    fft_size = float(fft_size)
    fft_bin = int(np.round((f_hz*2.0*fft_size/sample_rate_hz)))
    if fft_bin >= fft_size:
        fft_bin = fft_size-1
    return fft_bin


def stft_for_recon(x, fft_size, hopsamp):
    """Compute and return the STFT of the time domain signal x.

    Args:
        x (1-dim Numpy array): A time domain signal.
        fft_size (int): FFT size. Should be a power of 2, otherwise DFT will be used.
        hopsamp (int):

    Returns:
        The STFT(complex-valued). The rows are the time slices and columns are the frequency bins.
    """
    fft_size = int(fft_size)
    window = np.hanning(fft_size)
    hopsamp = int(hopsamp)
    # print("x len is:" , len(x)) 166960
    return np.array([np.fft.rfft(window*x[i:i+fft_size], fft_size)
                     for i in range(0, len(x)-fft_size, hopsamp)])


def istft_for_recon(X, fft_size, hopsamp):
    """Invert a STFT into a time domain signal.

    Args:
        X (2-dim Numpy array): Input spectrogram. The rows are the time slices and columns are the frequency bins.
        fft_size (int):
        hopsamp (int): The hop size, in samples.

    Returns:
        The inverse STFT.
    """
    fft_size = int(fft_size)
    hopsamp = int(hopsamp)
    window = np.hanning(fft_size)
    time_slices = X.shape[0]
    len_samples = int(time_slices*hopsamp + fft_size)
    x = np.zeros(len_samples)
    for n,i in enumerate(range(0, len(x)-fft_size, hopsamp)):
        x[i:i+fft_size] += window*np.real(np.fft.irfft(X[n]))
    return x


def get_signal(in_file, expected_fs=16000):
    """Load a wav file.

    If the file contains more than one channel, return a mono file by taking
    the mean of all channels.

    If the sample rate differs from the expected sample rate,
    raise an exception.

    Args:
        in_file: The input wav file, which should have a sample rate of `expected_fs`.
        expected_fs (int): The expected sample rate of the input wav file.

    Returns:
        The audio siganl as a 1-dim Numpy array. The values will be in the range [-1.0, 1.0].
    """
    fs, y = scipy.io.wavfile.read(in_file)
    num_type = y[0].dtype
    if num_type == 'int16':
        y = y*(1.0/32768)
    elif num_type == 'int32':
        y = y*(1.0/2147483648)
    elif num_type == 'float32':
        # Nothing to do
        pass
    elif num_type == 'uint8':
        raise Exception('8-bit PCM is not supported.')
    else:
        raise Exception('Unknown format.')
    if fs != expected_fs:
        raise Exception('Invalid sample rate.')
    if y.ndim == 1:
        return y
    else:
        return y.mean(axis=1)


def reconstruct_signal_griffin_lim(magnitude_spectrogram, fft_size, hopsamp, iterations):
    """Reconstruct an audio signal from a magnitude spectrogram.

    Given a magnitude spectrogram as input, reconstruct
    the audio signal and return it using the Griffin-Lim algorithm from the paper:
    "Signal estimation from modified short-time fourier transform" by Griffin and Lim,
    in IEEE transactions on Acoustics, Speech, and Signal Processing. Vol ASSP-32, No. 2, April 1984.

    Args:
        magnitude_spectrogram (2-dim Numpy array): The magnitude spectrogram. The rows correspond to the time slices
            and the columns correspond to frequency bins.
        fft_size (int): The FFT size, which should be a power of 2.
        hopsamp (int): The hope size in samples.
        iterations (int): Number of iterations for the Griffin-Lim algorithm. Typically a few hundred
            is sufficient.

    Returns:
        The reconstructed time domain signal as a 1-dim Numpy array.
    """
    time_slices = magnitude_spectrogram.shape[0]
    len_samples = int(time_slices*hopsamp + fft_size)
    # Initialize the reconstructed signal to noise.
    x_reconstruct = np.random.randn(len_samples)
    n = iterations  # number of iterations of Griffin-Lim algorithm.
    while n > 0:
        n -= 1
        reconstruction_spectrogram = stft_for_recon(x_reconstruct, fft_size, hopsamp)  # 通过时域信号计算频域信号
        reconstruction_angle = np.angle(reconstruction_spectrogram)
        # Discard magnitude part of the reconstruction and use the supplied magnitude spectrogram instead.
        proposal_spectrogram = magnitude_spectrogram*np.exp(1.0j*reconstruction_angle)
        prev_x = x_reconstruct
        x_reconstruct = istft_for_recon(proposal_spectrogram, fft_size, hopsamp)  # 获取stft逆变换的时域信号
        diff = sqrt(sum((x_reconstruct - prev_x)**2)/x_reconstruct.size)
        print('Reconstruction iteration: {}/{} RMSE: {} '.format(iterations - n, iterations, diff))
    return x_reconstruct


def save_audio_to_wav(x, sample_rate, outfile='out.wav'):
    print(outfile)
    """Save a mono signal to a file.

    Args:
        x (1-dim Numpy array): The audio signal to save. The signal values should be in the range [-1.0, 1.0].
        sample_rate (int): The sample rate of the signal, in Hz.
        outfile: Name of the file to save.

    """
    x_max = np.max(abs(x))
    assert x_max <= 1.0, 'Input audio value is out of range. Should be in the range [-1.0, 1.0].'
    x = x*32767.0
    data = array.array('h')
    for i in range(len(x)):
        cur_samp = int(round(x[i]))
        data.append(cur_samp)
    f = wave.open(outfile, 'w')
    f.setparams((1, 2, sample_rate, 0, "NONE", "Uncompressed"))
    f.writeframes(data.tostring())
    f.close()


def save_audio_to_flac(x, sample_rate, outfile='out.flac'):
    print(outfile)
    """Save a mono signal to a file.
    Args:
        x (1-dim Numpy array): The audio signal to save. The signal values should be in the range [-1.0, 1.0].
        sample_rate (int): The sample rate of the signal, in Hz.
        outfile: Name of the file to save.
    """
    x_max = np.max(abs(x))
    assert x_max <= 1.0, 'Input audio value is out of range. Should be in the range [-1.0, 1.0].'
    data = array.array('d')  # double
    for i in range(len(x)):
        cur_samp = x[i]
        data.append(cur_samp)

    sf.write(outfile, data, sample_rate)