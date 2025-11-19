import librosa
import torch
from pycochleagram import cochleagram
import torch.nn.functional as F
import numpy as np
import os
import pyloudnorm as pyln
import warnings

#can ignore divide by zero warnings, as some frequencies may be 0
warnings.filterwarnings("ignore", message="divide by zero encountered in log10")
                        
home = os.path.dirname(os.path.abspath(__file__))


def extract(path, startInterval, endInterval, window=30):
    #args in miliseconds
    
    audio, sr = librosa.load(path, sr=None)
 
    #root mean squared normalization
    meter = pyln.Meter(sr)
    loudness = meter.integrated_loudness(audio)
    loud_norm_audio = pyln.normalize.loudness(audio, loudness, -12.0)

    #slice
    #get the duration in MS
    duration = (endInterval-startInterval) / 1000

    #figure out, in ms, where the 1/3 point of the vowel is
    startMS = startInterval + (duration/3)

    #figure out how many frames will be in a 30 ms window based on the sr
    framesInSlice = sr * (window/1000)

    #convert the starting point to a frame index
    startFrame = int(startMS * sr)

    #find the ending index
    endFrame = int(startFrame+framesInSlice)

    #convert the endFrame back into MS, check that the slice didn't exceed the vowel length
    if endInterval < (endFrame * (1000/sr)):
        raise ValueError(f"Attemped to slice. Vowel not long enough")

    segment = loud_norm_audio[startFrame:endFrame]

    #cochleogram
    cg = cochleagram.cochleagram(
        segment,
        sr,
        n=24,
        low_lim=50,
        hi_lim=8000,
        sample_factor=2,
        strict=False,
        )
    
    print(cg)
    print(cg.shape)
    return
    
    #average over the time dimension
    cg_tensor = torch.tensor(cg, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
    cg_column = F.adaptive_avg_pool2d(cg_tensor, (80, 1)).squeeze().numpy()

    return(cg_column)

if __name__ == "__main__":
    sin70 = os.path.join(home, "testAudio", "sin_70Hz.wav")
    sin440 = os.path.join(home, "testAudio", "sin_440Hz.wav")

    extract(sin70, 0, 3000)