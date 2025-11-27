import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import ast 
from collections import defaultdict

#awk '{print $3, $4}' all.tsv > cgs.tsv 

def readFile(file):
    vowels = defaultdict(list)

    with open(file) as f:
        #skip header
        next(f)
        for line in f:
            #make sure line isn't empty
            if line.strip():
            #parts[2] is the vowel, parts[3] is the cg
                parts = line.split("\t")
                vowels[parts[2]].append(ast.literal_eval(parts[3]))
    
    return vowels


def plot(vowel, lists, lang):
    
    data = np.array(lists)

    #scale it up so we might be able to see something...
    data = data * 1000
    dataByIndex = [data[:, i] for i in range(data.shape[1])]

    #make the plot
    plt.figure(figsize=(14, 6))
    plt.boxplot(dataByIndex)
    plt.xlabel("Filter Bank (1~50Hz, 26~8000Hz)")
    plt.ylabel("Amplitude (scaled by x1000)")
    plt.title(f"Boxplot per Filter Bank for {vowel}")
    plt.savefig(f"{lang}_{vowel}.png")



if __name__ == "__main__":

    lang = "Spanish"

    vowels = readFile('/Users/annikashankwitz/Desktop/smallNeuralL2Prediction/vowelSlices/Spanish/all.tsv')
    
    for key in vowels.keys():
        plot(key, vowels[key], lang)