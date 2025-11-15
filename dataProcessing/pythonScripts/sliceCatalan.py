from praatio import textgrid
from collections import defaultdict
import random
import librosa
import torch
from pycochleagram import cochleagram
import torch.nn.functional as F
import numpy as np
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)

random.seed(42)

#set-up
audio = "/fs/nexus-scratch/ashankwi/phonProjectF25/catalan_slr69/all_audio/"
textGrids = "/fs/nexus-scratch/ashankwi/phonProjectF25/catalan_slr69_textGrids/all_grids/"
transcript = "/fs/nexus-scratch/ashankwi/phonProjectF25/catalan_slr69/allTranscripts.tsv"
output = "/fs/nexus-scratch/ashankwi/phonProjectF25/catalan_vowelSlices/allSlices.tsv"
vowels = ['a', 'e', 'i', 'o', 'u', 'ɔ', 'ɛ']


#functions 
#finds all the vowels in a textGrid
def readTextGrid(textGrid):
    try:
        tg = textgrid.openTextgrid(textGrid, False)
        phone_tier = tg.getTier("phones")
        vowelIntervals = []

        for entry in phone_tier.entries:
            if entry.label.lower() in vowels:
                vowelIntervals.append((entry.label.lower(), entry.start, entry.end))
        
        return vowelIntervals
    except FileNotFoundError:
        #print(f'{textGrid} not found.')
        return []

#makes a dictionary with speaker metaInfo
def makeInfoDict():
    speakerInfoDict = {}
    with open(transcript) as f:
        for line in f:
            speaker = line.split("\t")[0].split("_")[1]
            gender = line.split("\t")[0][2]
            if speaker not in speakerInfoDict.keys():
                speakerInfoDict[speaker] = gender
    
    return speakerInfoDict


#loads the transcript
def loadTranscrip():
    mapping = defaultdict(list)
    with open(transcript) as f:
        for line in f:
            fileName = line.split()[0]
            speaker = fileName.split("_")[1]
            mapping[speaker].append(fileName)

    return mapping

#makes a dictionary of one speaker's vowels
def makeSpeakerDict(speaker, map):

    speakerDict = defaultdict(list)

    for file in map[speaker]:
        textGridPath = f"{textGrids}{file}.TextGrid"
        vowelIntervals = readTextGrid(textGridPath)
        if vowelIntervals != None:
            for vowel, start, end, in vowelIntervals:
                speakerDict[vowel].append([start, end, file])
    
    return speakerDict

#get the cochleogram
#representation: a column vector which colapses time and keeps frequency
def extract(path, startInterval, endInterval, window=30):
    
    audio, sr = librosa.load(path, sr=None)

    #normalize the audio
    norm = max(abs(audio))
    if norm > 0:
        audio = audio / norm

    #slice
    duration = endInterval-startInterval
    startSlice = startInterval + (duration/3)
    endSlice  = startSlice + (window / 1000)

    if endSlice > endInterval:
        raise ValueError(f"Slice endpoint {endSlice:.4f}s exceeds vowel end {endInterval:.4f}s")

    segment = audio[int(startSlice*sr):int(endSlice*sr)]
    segment_t = torch.tensor(segment, dtype=torch.float32).unsqueeze(0)

    #cochleogram
    cg = cochleagram.cochleagram(
        segment_t,
        sr,
        n=80,
        low_lim=50,
        hi_lim=8000,
        sample_factor=4,
        strict=False,
        )
    
    #average over the time dimension
    cg_tensor = torch.tensor(cg, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
    cg_column = F.adaptive_avg_pool2d(cg_tensor, (80, 1)).squeeze().numpy()

    return(cg_column)

 
if __name__ == "__main__":
    #clear the output file
    open(output, "w").close()

    infoDict = makeInfoDict()
    speakerMap = loadTranscrip()

    print('speakers read in')

    notEnough = []

    with open(output, "w") as out:
        out.write("speaker ID\tGender\tVowel\tCochleogram\tfile\n")

        speakerCount = 1
        for speaker, gender in infoDict.items():

            if speaker in ['08106', '06042','06008', '04247', '09796', '08001', '00762', '06582', '04484']:
                continue
    
            print(f'working on speaker {speaker}')

            speakerDict = makeSpeakerDict(speaker, speakerMap)
        
            for vowel, items in speakerDict.items():
                #shuffle all a speaker's productions of a vowel
                random.shuffle(items)

                #and we want to sample 30 of them
                #30 per vowel per speaker
                validVowels = 0

                for start, end, file in items:
                    if validVowels >= 20: 
                        break

                    audioPath = f"{audio}{file}.wav"

                    try:
                        cg = extract(audioPath, start, end)
                    except ValueError:
                        #print(f"Skipping problematic slice: {file}, {vowel}")
                        continue

                    cg_flat = cg.flatten()
                    cg_formated = ",".join(map(str,cg_flat))
                    out.write(f"{speaker}\t{gender}\t{vowel}\t[{cg_formated}]\t{file}\n")
                    out.flush()                    
                    validVowels += 1

                if validVowels < 20:
                    notEnough.append(speaker)
                    #raise ValueError(f"only found {validVowels} for {vowel} from {speaker}")


            speakerCount += 1
        
        print(notEnough)
                    





