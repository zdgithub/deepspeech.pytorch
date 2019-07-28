#
# reference: https://github.com/bkvogel/griffin_lim
#

import argparse
from pylab import *
import os
import numpy as np
import matplotlib.pyplot as plt
from reconToAudio import audio_utilities
from reconToAudio import Fftreader


def getM(mdir):
    '''
    draw M feature map
    :param mdir: m.txt in K x T shape
    :return:
    '''
    norm = np.loadtxt(mdir)
    print('m shape is:', norm.shape)  # K x T

    dimMeans = np.mean(norm, axis=1)
    frameMeans = np.mean(norm, axis=0)

    plt.figure()
    plt.hist(norm.flatten(), bins=100, range=(-100, 100))  # use quantity
    plt.title('hist for m')
    plt.text(-100, 2e3, 'μ=%.3f, δ=%.3f' % (np.mean(norm), np.var(norm)))
    plt.savefig(os.path.dirname(mdir) + '/m_hist.png')
    plt.clf()

    return dimMeans, frameMeans, norm


def mBlarInput(fft, m):
    '''
    blar the input fft matrix with m
    :param fft input: K x T
    :param m: K x T
    :return: m blared input: K x T
    '''
    K = fft.shape[0]
    T = fft.shape[1]
    range1 = np.array(list(range(K)) * K * T).reshape((K, T, K))
    range2 = np.array(list(range(K)) * K * T).reshape((K, T, K)).transpose()

    abs_m = np.abs(m).reshape((K, T, 1))
    m_tile = np.tile(abs_m, (1, 1, K))
    out = np.maximum((m_tile - np.abs(range1 - range2)) / (np.square(m_tile)), 0)
    blar = (np.multiply(out, (m_tile > 1)) + np.multiply((m_tile <= 1), (range1 == range2)))
    norm_index = np.tile(np.sum(blar, axis=2).reshape([K, T, 1]), (1, 1, K))
    blar = blar / norm_index
    inputtile = np.tile(fft.reshape([K, T, 1]), (1, 1, K))
    inputs = np.sum(np.multiply(blar, inputtile), axis=0).transpose().reshape([K, T])

    return inputs


def saveMusic(stft_mag, outdir):
    '''
    from the magnitude spectrogram to reconstruct an audio and save it
    :param stft_mag: T x K(real-valued) stft (magnitude only) spectrogram
    :param outdir: save reconstructed music as wav file outdir
    :return:
    '''
    parser = argparse.ArgumentParser()
    parser.add_argument('--in_file', type=str, default="../train_only_one/train.wav",
                        help='Input WAV file')
    parser.add_argument('--sample_rate', default=16000, type=int,
                        help='Sample rate in Hz')
    parser.add_argument('--window_size', default=0.02, type=float,
                        help='Window size for spectrogram in seconds')
    parser.add_argument('--window_stride', default=0.01, type=float,
                        help='Window stride for spectrogram in seconds')
    parser.add_argument('--cutoff_freq', type=int, default=1000,
                        help='If filter is enable, the low-pass cutoff frequency in Hz')
    parser.add_argument('--iterations', default=300, type=int,
                        help='Number of iterations to run')
    args = parser.parse_args()

    # using 512 instead of 320 will make reconstructed audio more clear
    # but deepspeech2 is 320 and wav2letter is 512
    fft_size = int(args.sample_rate * args.window_size)
    hopsamp = int(args.sample_rate * args.window_stride)

    # Use the Griffin&Lim algorithm to reconstruct an audio signal from the
    # magnitude spectrogram.
    x_reconstruct = audio_utilities.reconstruct_signal_griffin_lim(stft_mag,
                                                                   fft_size, hopsamp,
                                                                   args.iterations)

    # The output signal must be in the range [-1, 1], otherwise we need to clip or normalize.
    max_sample = np.max(abs(x_reconstruct))
    if max_sample > 1.0:
        x_reconstruct = x_reconstruct / max_sample

    # Save the reconstructed signal to a WAV file.
    audio_utilities.save_audio_to_wav(x_reconstruct, args.sample_rate, outfile=outdir)


def makeVoice(prefftdir, dimMeans, frameMeans, m):
    '''

    :param prefftdir: preFft.txt
    :param dimMeans: got from getM()
    :param frameMeans: got from getM()
    :param m: got from getM() K x T
    :param dimThresh: list
    :param frameThresh: list
    :return:
    '''
    fft = np.loadtxt(prefftdir)  # K x T
    print('fft shape is:', fft.shape)
    fftT = fft.T  # T x K
    saveMusic(fftT, os.path.dirname(prefftdir) + '/original.wav')
    Fftreader.FftDrawer(fftT, "/original", os.path.dirname(prefftdir))

    # prefft with m blar noise
    noise_fft = mBlarInput(fft, m).T  # T x K
    saveMusic(noise_fft, os.path.dirname(prefftdir) + '/noise_original.wav')
    Fftreader.FftDrawer(noise_fft, "/noise_original", os.path.dirname(prefftdir))

    dimhist = np.sort(dimMeans)
    framehist = np.sort(frameMeans)
    norm = m  # K x T
    normhist = np.sort(norm.flatten())

    normThresh = [0.2]
    for threshPercent in normThresh:
        thresh = normhist[-1 * int(len(normhist) * threshPercent)]
        outIndexs = [(i, j) for i in range(norm.shape[0]) for j in range(norm.shape[1]) if norm[i][j] > thresh]
        outIndex_rev = [(i, j) for i in range(norm.shape[0]) for j in range(norm.shape[1]) if norm[i][j] <= thresh]
        print(len(outIndexs))
        print(len(outIndex_rev))
        # K x T
        newnorm = np.copy(fft)
        newnorm_rev = np.copy(fft)

        for ind in outIndexs:
            newnorm[ind] = 0
        for ind in outIndex_rev:
            newnorm_rev[ind] = 0

        Fftreader.FftDrawer(newnorm.T, '/throw%d' % (100 * threshPercent) + 'high', os.path.dirname(prefftdir))
        Fftreader.FftDrawer(newnorm_rev.T, '/throw%d' % (100 - 100 * threshPercent) + 'low', os.path.dirname(prefftdir))

        outdir = os.path.dirname(prefftdir) + '/throw%d' % (100 * threshPercent) + 'high.wav'
        outdir_rev = os.path.dirname(prefftdir) + '/throw%d' % (100 - 100 * threshPercent) + 'low.wav'

        saveMusic(newnorm.T, outdir)
        saveMusic(newnorm_rev.T, outdir_rev)


if __name__ == '__main__':
    wholeDir = './07-23-14-14/'
    mdir = wholeDir + 'm.txt'
    prefftdir = wholeDir + 'fft.txt'
    lossdir = wholeDir + 'loss.txt'

    # draw pics about m
    dimMeans, frameMeans, m = getM(mdir)  # draw hist m
    Fftreader.mcolorDrawer(m, wholeDir)  # draw hotmap m

    # make zero points then reconstruct to audios
    makeVoice(prefftdir, dimMeans, frameMeans, m)

    # draw loss curve
    y = np.loadtxt(lossdir)
    x = np.arange(len(y))
    plt.figure()
    plt.plot(x, y, color='r', linewidth=1)
    plt.xlabel('iters')
    plt.ylabel('loss')
    plt.title('loss curve')
    plt.savefig(wholeDir + '/loss.png', dpi=None)














