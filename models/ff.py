import torch
import torch.nn as nn
import torch.nn.functional as F
import tensorflow as tf
import ast
from torch import optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np

#model
class feedForward(nn.Module):

    def __init__(self, inputDim, modelDim, numVowels):
        
        super().__init__()

        #layers
        self.ff1 = nn.Linear(inputDim, modelDim)
        self.ff2 = nn.Linear(modelDim, modelDim)
        self.ff3= nn.Linear(modelDim, numVowels)

        #non-linear activation activation
        self.relu = nn.ReLU()
    
    def forward(self, input):

        #put the input through the layer
        #non-linear activation after 1st and 2nd layer
        #soft-max after 3rd layer
        firstLayer = self.ff1(input)
        firstActivation = self.relu(firstLayer)
        
        secondLayer = self.ff2(firstActivation)
        secondActivation = self.relu(secondLayer)
        
        thirdLayer = self.ff3(secondActivation)

        return thirdLayer


#training loop
def trainModel(modelArgs, trainData, devData):
    
    #model args = lr, num epochs, modeDim 

    #these should be fixed
    inputDim = 26
    numVowels = 5

    #initalize the model
    model = feedForward(inputDim, modelArgs[2], numVowels)
    model = model.double()

    #optimizer
    optimizer = optim.Adam(model.parameters(), lr = modelArgs[0])

    #loss 
    lossFN = nn.CrossEntropyLoss()

    trainLoader = DataLoader(trainData, modelArgs[1], shuffle=True)

    for epoch in range(modelArgs[1]):

        epochLoss = 0

        for batchX, batchY in trainLoader:

            optimizer.zero_grad()

            outputs = model(batchX)

            loss = lossFN(outputs, batchY)

            loss.backward()

            optimizer.step()

            epochLoss += loss.item()
        

        avgLoss = epochLoss/len(trainData)
        print(f"Epoch {epoch+1}, Avg. Train Loss {avgLoss:.4f}")
        trainAccuracy = findAccuracy(model, trainData)
        print(f"Epoch {epoch+1}, Avg. Train Accuracy {trainAccuracy:.4f}")        

        devLoss = evalModel(model, devData)
        print(f"Epoch {epoch+1}, Avg. Dev Loss {devLoss:.4f}")
        devAccuracy = findAccuracy(model, devData)
        print(f"Epoch {epoch+1}, Avg. Dev Accuracy {devAccuracy:.4f}")    
    
    model.eval()
    return model


#find the loss of another set
def evalModel(model, data):
    model.eval()
    totalLoss = 0
    lossFN = nn.CrossEntropyLoss()

    dataLoader = DataLoader(data, modelArgs[1], shuffle=True)
    with torch.no_grad():
        for batchX, batchY in dataLoader:
            outputs = model(batchX)
            loss = lossFN(outputs, batchY)
            totalLoss += loss.item()
    
    avgLoss = totalLoss / len(dataLoader)
    model.train()
    return avgLoss


#compute accuracy
def findAccuracy(model, data):
    model.eval()
    corectVowels = 0
    totalVowels = 0

    dataLoader = DataLoader(data, modelArgs[1], shuffle=True)
    with torch.no_grad():
        for batchX, batchY in dataLoader:
            outputs = model(batchX)
            _, predicted = torch.max(outputs,1)
            corectVowels += (predicted == batchY).sum().item()
            totalVowels  += batchY.size(0)
    
    model.train()
    return corectVowels/totalVowels
        

#read in the data, and covert to tensors
def readData(path):
    
    y = []
    x = []

    with open(path) as f:
        #skip header
        next(f)
        for line in f:
            if line.strip():
                parts = line.split("\t")
                #two is the vowel, three is the cg
                y.append(parts[2])
                x.append(ast.literal_eval(parts[3]))
    
    vowelEncoderMap = {'a':0,
                       'e':1,
                       'i':2,
                       'o':3,
                       'u':4,
                       'ɛ':5,
                       'ɔ':6 }
    
    #encode the vowel
    encoded = [vowelEncoderMap[vowel] for vowel in y]

    y = torch.tensor(encoded)
    x = torch.tensor(x, dtype=torch.float64)
    return TensorDataset(x,y)


if __name__ == "__main__":
    train = "/Users/annikashankwitz/Desktop/smallNeuralL2Prediction/vowelSlices/Spanish/train.tsv"
    dev = "/Users/annikashankwitz/Desktop/smallNeuralL2Prediction/vowelSlices/Spanish/dev.tsv"
    test = "/Users/annikashankwitz/Desktop/smallNeuralL2Prediction/vowelSlices/Spanish/test.tsv"
    train = readData(train)
    dev = readData(dev)
    test = readData(test)

    #set your model args here
    #gridSearch to come
    #learning rate, numEpochs, dmodel
    modelArgs = [0.0001, 10, 52]

    model = trainModel(modelArgs, train, dev)
    


    
        
    