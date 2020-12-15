#import librosa
#import soundfile as sf
from pydub import AudioSegment
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal

def read_signal(directory):
    # 末尾の拡張子によって
    if directory.endswith("wav"):
        data=AudioSegment.from_wav(directory)
        rate=data.frame_rate
        data=np.array(data.get_array_of_samples(), dtype=float)
        return rate, data
    elif directory.endswith("mp3"):
        data=AudioSegment.from_mp3(directory)
        rate=data.frame_rate
        data=np.array(data.get_array_of_samples(), dtype=float)
        return rate, data
    else:
        print("invalid file")
        exit()

def psd(data, fs):
    return signal.welch(data, fs=fs)
