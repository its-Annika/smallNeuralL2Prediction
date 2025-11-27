import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


dataFile = "/Users/annikashankwitz/Desktop/smallNeuralL2Prediction/models/CatalanPreds.tsv"
data = pd.read_csv(dataFile, sep="\t").iloc[:, :-2]

trueCol = "goldLabel"
predCol = "predictedLabel"
vowelColums = data.columns[2:]

counts = {}
averageProbs = {}

for trueVowel, group in data.groupby(trueCol):
    countSeries = group[predCol].value_counts()
    counts[trueVowel] = {v: countSeries.get(v,0) for v in vowelColums}

    avgProbSeries = group[vowelColums].mean()
    averageProbs[trueVowel] = avgProbSeries.to_dict()

countsDF = pd.DataFrame.from_dict(counts, orient='index').fillna(0).astype(int)
avgProbDF = pd.DataFrame.from_dict(averageProbs, orient='index').fillna(0.0)

print("Counts:")
print(countsDF)

print("Probs:")
print(avgProbDF)

plt.figure(figsize=(8, 6)) # Adjust figure size as needed
sns.heatmap(countsDF, annot=True, cmap='viridis', fmt=".0f")
plt.title('L2 Catalan Vowel Predictions')
plt.xlabel("Predicted Vowels")
plt.ylabel("True Catalan Vowels")
plt.show()