import librosa
#import soundfile as sf
from pydub import AudioSegment
import numpy as np
import matplotlib.pyplot as plt

def read_signal(directory):
    # 末尾の拡張子によって
    if directory.endswith("wav"):
        data=AudioSegment.from_wav(directory)
        data=np.array(data.get_array_of_samples(), dtype=float)
        return data
    elif directory.endswith("mp3"):
        data=AudioSegment.from_mp3(directory)
        data=np.array(data.get_array_of_samples(), dtype=float)
        return data
    else:
        print("invalid file")
        exit()

