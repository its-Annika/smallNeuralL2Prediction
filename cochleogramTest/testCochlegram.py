import librosa
import torch
from pycochleagram import cochleagram
import torch.nn.functional as F
import numpy as np
import os
import pyloudnorm as pyln
import warnings

#check what pycochleagram you're using
os.path.abspath(cochleagram.__file__)
                        
home = os.path.dirname(os.path.abspath(__file__))


#slicing works

def extract(path, startInterval, endInterval, window=30):
    #args in miliseconds
    
    audio, sr = librosa.load(path, sr=None)
 
    #loudness normalization
    meter = pyln.Meter(sr)
    loudness = meter.integrated_loudness(audio)
    loud_norm_audio = pyln.normalize.loudness(audio, loudness, -25.0)

    #slice
    #get the duration in MS
    duration = (endInterval-startInterval)
    print(f'duration:{duration} ms')

    #get the midpoint in ms
    #and find in it frames
    midpoint = startInterval + duration/2
    midpointFrame = int(midpoint * (sr/1000))
    print(f'midpoint:{midpoint} ms')
    print(f'midpointFrame: {midpointFrame}')

    #figure out how many frames will be in half the window
    framesInSlice = sr * ((window/2)/1000)
    print(f'framesInSlice:{framesInSlice} frames')

    #find the start and the end frame
    #15 ms off the mid point in either direction
    startFrame = int(midpointFrame-framesInSlice)
    print(f"startFrame:{startFrame}")
    endFrame = int(midpointFrame+framesInSlice)
    print(f"endFrame: {endFrame}")

    #find the end and the start in ms
    endMS = endFrame/(sr/1000)
    startMS = startFrame/(sr/1000)
    print(f"startMS:{startMS}")
    print(f"endMS:{endMS}")

    # check that the slice didn't exceed the vowel length
    if endMS > endInterval or startInterval > startMS:
        raise ValueError(f"Attemped to slice. Vowel not long enough")

    #from old version
    #figure out, in ms, where the 1/3 point of the vowel is
    # startMS = startInterval + (duration/3)
    # print(f'startMs:{startMS}')

    # #figure out how many frames will be in a 30 ms window based on the sr
    # framesInSlice = sr * (window/1000)
    # print(f'framesInSlice:{framesInSlice}')

    # #convert the starting point to a frame index
    # startFrame = int(startMS * (sr/1000))
    # print(f'startFrame:{startFrame}')

    #find the ending index
    # endFrame = int(startFrame+framesInSlice)
    # print(f'endFrame:{endFrame}')

    #find the ending time in ms
    # endMS = endFrame/(sr/1000)
    # print(f'endMS:{endMS}')

    # check that the slice didn't exceed the vowel length
    #if endInterval < endMS:
        #raise ValueError(f"Attemped to slice. Vowel not long enough")


    segment = loud_norm_audio[startFrame:endFrame]

    #cochleogram
    cg = cochleagram.cochleagram(
        segment,
        sr,
        n=24,
        low_lim=50,
        hi_lim=8000,
        sample_factor=1,
        strict=False,
        )

    #average over the time dimension
    cg_column = np.mean(cg, axis=1)
    #np.savetxt('440Hz.csv', cg_column, delimiter=',')
    return(cg_column)

if __name__ == "__main__":
    sin440Sil = os.path.join(home, "testAudio", "440HZWithSilence.wav")
    sin2000Sil = os.path.join(home, "testAudio", "2000HzWithSilence.wav")
    sin100Sil = os.path.join(home, "testAudio", "100HZWithSilence.wav")
    sin70Sil = os.path.join(home, "testAudio", "70HzWithSilence.wav")
    sin6500Sil = os.path.join(home, "testAudio", "6500HzWithSilence.wav")

    extract(sin440Sil, 2000, 3000)