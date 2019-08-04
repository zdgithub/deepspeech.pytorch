import os
import wave
import numpy as np
import audiotools
import random

path1 = 'dev-clean-noise'
path2 = 'music'

fileIn1 = os.listdir(path1)
fileIn2 = os.listdir(path2)
musicNum = len(fileIn2)
print('music num is:', musicNum)

def main():

    for file1 in fileIn1:

        if not ('.flac' in file1):
            continue

        adopath_flac = os.path.join(path1, file1)
        adopath_wav = adopath_flac.replace('flac', 'wav')
        audiotools.open(adopath_flac).convert(adopath_wav, audiotools.WaveAudio)   # sampleRate 16000Hz

        with wave.open(adopath_wav, 'rb') as f1:
            params1 = f1.getparams()
            nchannels1, sampwidth1, framerate1, nframes1, comptype1, compname1 = params1[:6]
            f1_str_data = f1.readframes(nframes1)

        f1_wave_data = np.fromstring(f1_str_data, dtype=np.int16)

        musicID = random.randint(0, musicNum - 1)  # [a, b]
        music_wav = os.path.join(path2, fileIn2[musicID])  # sampleRate 44100Hz
        with wave.open(music_wav, 'rb') as f2:
            params2 = f2.getparams()
            nchannels2, sampwidth2, framerate2, nframes2, comptype2, compname2 = params2[:6]
            f2_str_data = f2.readframes(nframes2)

        f2_wave_data = np.fromstring(f2_str_data, dtype=np.int16)
        f2_wave_data = f2_wave_data[::3]  # decrease sampleRate
        nframes2 = len(f2_wave_data)

        if nframes1 < nframes2:
            length = abs(nframes2 - nframes1)
            startp = random.randint(0, length)
            rf1_wave_data = f1_wave_data
            rf2_wave_data = f2_wave_data[startp:startp+nframes1]
        elif nframes1 > nframes2:
            length = abs(nframes1 - nframes2)
            temp_array = np.zeros(length, dtype=np.int16)
            rf2_wave_data = np.concatenate((f2_wave_data, temp_array))
            rf1_wave_data = f1_wave_data
        else:
            rf1_wave_data = f1_wave_data
            rf2_wave_data = f2_wave_data

        new_wave_data = rf1_wave_data + rf2_wave_data
        new_wave = new_wave_data.tostring()

        out_wav = adopath_wav
        with wave.open(out_wav, 'wb') as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(16000)
            wf.writeframes(new_wave)
    

        out_flac = adopath_flac
        audiotools.open(out_wav).convert(out_flac, audiotools.FlacAudio)



if __name__ == '__main__':
    main()

