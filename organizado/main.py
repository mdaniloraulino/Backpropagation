import pandas as pd
import numpy as np
from numba import vectorize,jit

#leitura de csv e inicio das variaveis
inputCsv = pd.read_csv('mnist_train.csv')
inputData = []
inputNumber = []

import pandas as pd
import numpy as np

# Global variables
outputDictionary = {'0':[1.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0], '1':[0.0,1.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0],
                '2':[0.0,0.0,1.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0], '3':[0.0,0.0,0.0,1.0,0.0,0.0,0.0,0.0,0.0,0.0], '4':[0.0,0.0,0.0,0.0,1.0,0.0,0.0,0.0,0.0,0.0], 
                '5':[0.0,0.0,0.0,0.0,0.0,1.0,0.0,0.0,0.0,0.0], '6':[0.0,0.0,0.0,0.0,0.0,0.0,1.0,0.0,0.0,0.0], '7':[0.0,0.0,0.0,0.0,0.0,0.0,0.0,1.0,0.0,0.0], 
                '8':[0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,1.0,0.0], '9':[0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,1.0] }

learningRate = 0.2
middleLayerSize = 100
outputSize = 10
inputSize = 784

v = np.random.randn(28*28, middleLayerSize) / np.sqrt(28*28) # np.random.uniform(-1.00, 1.00,(inputSize, middleLayerSize)) # [linhas, middleLayerSize]
w = np.random.randn(middleLayerSize, inputSize) / np.sqrt(middleLayerSize) #np.random.uniform(-1.00, 1.00,(middleLayerSize, outputSize)) # [middleLayerSize, outputSize]
errors = []


# Functions
def prepareData():
    i = 1
    for row in inputCsv.itertuples(index=False):
        arrRow = list(row)
        inputNumber.append(arrRow.pop(0))
        arrRow = np.array(arrRow)
        inputData.append(arrRow)
    
def train(maxEpochs):
    global errors
    global graph
    global inputData

    for epoch in range(maxEpochs):
        errorCount = 0
        print('Period ' + str(epoch + 1))
        
        for i in range(len(inputData)):
            row = inputData[i]
            expectedNumber = inputNumber[i]
            expectedNumberObj = outputDictionary[str(expectedNumber)]
            zIn = calcZIn(row)
            zOutput = calcDelta(zIn, middleLayerSize)
            yIn = calcYIn(zOutput)
    
            yOutput = calcDelta(yIn, outputSize)
            validate = validadeOutput(expectedNumberObj, yOutput)
            
            if(validate == False):
                errorCount+= 1;
                propagateError(expectedNumberObj, row, yOutput, zOutput, zIn, yIn)
            if i % 1000 == 0:
                print(f'{i} linhas da epoca {epoch}, Erros até aqui {errorCount}')
        errors.append(errorCount)
        print('Error: ' + str(errorCount))

def calcZIn(row): 
    result = []
    
    for j in range(middleLayerSize):
        result.append(0)
        for i in range(inputSize):
            result[j] += v[i,j] * row[i]
    
    return result

def calcYIn(zOutput): 
    result = []
    for j in range(outputSize):
        result.append(0)
        for i in range(middleLayerSize):
            result[j] += w[i,j] * zOutput[i]
    
    return result


def calcDelta(arr, arrSize):
    deltas = []
    
    for i in range(arrSize):
        deltas.append(activationFunction(arr[i]))
        
    return deltas

def activationFunction(x):
    #r = (2 / 1 + (np.exp(-x))) -1 
    r = 1.0 / (1.0 + np.exp(-x))
    return r

def validadeOutput(targetObj, yOutput):
    for i in range(len(yOutput)):
        if(targetObj[i] != yOutput[i]):
            return False
    
    return True

def propagateError(expectedArr, row, yOutput, zOutput, zIn, yIn):
    errorY = calcError(expectedArr, yOutput, yIn, outputSize)
    errorW = calcWeightCorrection(errorY, zOutput, middleLayerSize, outputSize)
    
    sumError = sumDelta(errorY, w, middleLayerSize, outputSize)
    errorZ = calcErrorZ(sumError, zOutput, zIn, middleLayerSize)
    errorV = calcWeightCorrection(errorZ, row, inputSize, middleLayerSize)
    
    updateWeight(w, errorW, middleLayerSize, outputSize)
    updateWeight(v, errorV, inputSize, middleLayerSize)
    

def calcError(expectedArr, outputArr, inArr, size):
    error = []
    
    for i in range(size):
        error.append((expectedArr[i] - outputArr[i]) * (outputArr[i] * (1.0 - outputArr[i]))) # (0.5 * ((1 + inArr[i]) * (1 - inArr[i])))) 
    return error

def calcErrorZ(expectedArr, outputArr, inArr, size):
    error = []
    for i in range(size):
        error.append((expectedArr[i] * (1.0 - outputArr[i]))) # (0.5 * ((1 + inArr[i]) * (1 - inArr[i])))) ;
    return error

def calcWeightCorrection(error, output, lenght1, length2):
    delta = [];
    for i in range(lenght1):
        delta.append([])
        for j in range(length2):
            delta[i].append(learningRate * error[j] * output[i])
    
    return delta

def sumDelta(error, weights, lenght1, length2):
    delta = []
    
    for i in range(lenght1):
        deltaValue = 0.0
        for j in range(length2):
            deltaValue += error[j] * weights[i][j];
            
        delta.append(deltaValue)
    
    return delta

def updateWeight(weight, delta, lenght1, length2):
    # (lenght1)
    # print(length2)
    for i in range(lenght1):
        for j in range(length2):
            # print(str(i) + ' - ' + str(j))
            weight[i][j] += delta[i][j]
    

# Execution
prepareData()
train(200)