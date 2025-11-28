import torch
import torch.nn as nn
import torch.nn.functional as F
import tensorflow as tf
import ast
from torch import optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import csv
from itertools import product

#model
class feedForward(nn.Module):

    def __init__(self, modelDim, inputDim=26, numVowels=5):
        
        super(feedForward, self).__init__()

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

    #initalize the model
    model = feedForward(modelArgs[2])
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
        trainAccuracy = findAccuracy(model, modelArgs, trainData)
        # print(f"Epoch {epoch+1}, Avg. Train Loss {avgLoss:.4f}")
        # print(f"Epoch {epoch+1}, Avg. Train Accuracy {trainAccuracy:.4f}")        

        devLoss = evalModel(model, modelArgs, devData)
        devAccuracy = findAccuracy(model, modelArgs, devData)
        # print(f"Epoch {epoch+1}, Avg. Dev Loss {devLoss:.4f}")
        # print(f"Epoch {epoch+1}, Avg. Dev Accuracy {devAccuracy:.4f}")    
    
    model.eval()
    return model


#find the loss of another set
def evalModel(model, modelArgs, data):
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
def findAccuracy(model, modelArgs, data):
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


#eval on Catalan
def EvalCatalan(model, catalanData, outPutPath):

    outputRows = []

    vowelDecoderMap = {0:'a',
                       1: 'e',
                       2: 'i',
                       3: 'o',
                       4: 'u',
                       5:'ɛ',
                       6:'ɔ'}
    
    #order of the vowels
    vowelList = [vowelDecoderMap[i] for i in range(len(vowelDecoderMap))]

    dataLoader = DataLoader(catalanData, 32)

    with torch.no_grad():
        for batchX, batchY in dataLoader:
            outputs = model(batchX)
            probs = torch.softmax(outputs, dim=1)
            _, predicted = torch.max(outputs,1)

            for i in range(batchX.size(0)):
                goldLabel = vowelDecoderMap[batchY[i].item()]
                predLabel = vowelDecoderMap[predicted[i].item()]
                row = [goldLabel, predLabel] + probs[i].tolist()
                outputRows.append(row)
    
    with open(outPutPath, "w", newline="") as f:
        writer = csv.writer(f, delimiter="\t")
        header = ["goldLabel", "predictedLabel"] + vowelList
        writer.writerow(header)
        writer.writerows(outputRows)


if __name__ == "__main__":
    #run this section if you're training ---------------------------------------------------------
    train = "/Users/annikashankwitz/Desktop/smallNeuralL2Prediction/vowelSlices/Spanish/train.tsv"
    dev = "/Users/annikashankwitz/Desktop/smallNeuralL2Prediction/vowelSlices/Spanish/dev.tsv"
    test = "/Users/annikashankwitz/Desktop/smallNeuralL2Prediction/vowelSlices/Spanish/test.tsv"

    modelPath = "/Users/annikashankwitz/Desktop/smallNeuralL2Prediction/models/"

    #process the data
    train = readData(train)
    dev = readData(dev)
    test = readData(test)

    #set your model args here
    #learning rate, numEpochs, dmodel

    #grid search
    models = []

    learningRate = [0.01, 0.001, 0.0001]
    modelDim = [5, 26, 52, 78, 104, 130]
    epochs = [5, 10, 20, 30, 40, 50]

    index = 1

    for lr, dMod, epoch in product(learningRate, modelDim, epochs):
        print(f"Round: {index} of {len(learningRate)*len(modelDim)*len(epochs)}")
        print(f"Trying: modelDim={dMod}, lr={lr}, epochs={epoch}")
        model = trainModel([lr, epoch, dMod], train, dev)

        accuracy = findAccuracy(model, [lr, epoch, dMod], dev)

        models.append((accuracy, [lr, epoch, dMod], model))
        
        index += 1
        
        
    modelsByAccuracy = sorted(models, key=lambda x:x[0])
    weakestModel = modelsByAccuracy[0]
    strongestModel = modelsByAccuracy[-1]
    moderateModel = modelsByAccuracy[int(len(modelsByAccuracy)/2)]


    print("gridSearch complete.")
    print(f'strongestModel: lr={strongestModel[1][0]}, epochs={strongestModel[1][1]}, modelDim={strongestModel[1][2]}')
    print(f"Dev accuracy: {strongestModel[0]}")
    torch.save(model.state_dict(), modelPath+"strongestModel.pth")    

    print(f'moderateModel: lr={moderateModel[1][0]}, epochs={moderateModel[1][1]}, modelDim={moderateModel[1][2]}')
    print(f"Dev accuracy: {moderateModel[0]}")
    torch.save(model.state_dict(), modelPath+"moderateModel.pth")    

    print(f'weakestModel: lr={weakestModel[1][0]}, epochs={weakestModel[1][1]}, modelDim={weakestModel[1][2]}')
    print(f"Dev accuracy: {weakestModel[0]}")
    torch.save(model.state_dict(), modelPath+"weakestModel.pth")   

    with open("gridSearchLog.tsv", "w+") as f:
        f.write("accuracy\tlr\tepochs\tmodeDim")
        for accuracy, params, model in modelsByAccuracy:
            f.write(f"{accuracy}\t{params[0]}\t{params[1]}\t{params[2]}\n")
        
        f.close()


    #run this section if you're evaluating on Catalan ---------------------------------------------------------
    #model args = [lr=0.01, modelDim=40, epochs=104]
    # model = feedForward(104).double()
    # stateDict = torch.load("GridSearchOptimized.pth")
    # model.load_state_dict(stateDict)
    # model.eval()

    # test = "/Users/annikashankwitz/Desktop/smallNeuralL2Prediction/vowelSlices/Catalan/allSlices.tsv"
    # test = readData(test)

    # EvalCatalan(model, test, "GSCatalanPreds.tsv")

