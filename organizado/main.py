import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#leitura de csv e inicio das variaveis
inputCsv = pd.read_csv('mnist_train.csv')
inputData = []
inputNumber = []



# Global variables
outputDictionary = {'0':[1.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0], '1':[0.0,1.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0],
                '2':[0.0,0.0,1.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0], '3':[0.0,0.0,0.0,1.0,0.0,0.0,0.0,0.0,0.0,0.0], '4':[0.0,0.0,0.0,0.0,1.0,0.0,0.0,0.0,0.0,0.0], 
                '5':[0.0,0.0,0.0,0.0,0.0,1.0,0.0,0.0,0.0,0.0], '6':[0.0,0.0,0.0,0.0,0.0,0.0,1.0,0.0,0.0,0.0], '7':[0.0,0.0,0.0,0.0,0.0,0.0,0.0,1.0,0.0,0.0], 
                '8':[0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,1.0,0.0], '9':[0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,1.0] }
#Grafico



learningRate = 0.2
middleLayerSize = 100
outputSize = 10
inputSize = 784
graphData = np.array([[0,100]])

v = np.random.randn(28*28, middleLayerSize) / np.sqrt(28*28) # np.random.uniform(-1.00, 1.00,(inputSize, middleLayerSize)) # [linhas, middleLayerSize]
w = np.random.randn(middleLayerSize, outputSize) / np.sqrt(middleLayerSize) #np.random.uniform(-1.00, 1.00,(middleLayerSize, outputSize)) # [middleLayerSize, outputSize]

errors = []

def prepareData():
    i = 1
    for row in inputCsv.itertuples(index=False):
        arrRow = list(row)
        inputNumber.append(arrRow.pop(0))
        arrRow = np.array(arrRow)
        inputData.append(arrRow)
        
def train(maxEpochs, train = False):
    global errors
    global graph
    global inputData
    global graphData
    for epoch in range(maxEpochs):
        errorCount = 0
        print('Period ' + str(epoch + 1))
        graphData = np.array([[0,100]])
        for i in range(len(inputData)):
            row = inputData[i]
            expectedNumber = inputNumber[i]
            expectedNumberObj = np.array(outputDictionary[str(expectedNumber)])
            zIn = calcZIn(np.array(row))
            zOutput = calcDelta(np.array(zIn), middleLayerSize)
            yIn = calcYIn(zOutput)  
            yOutput = calcDelta(yIn, outputSize)  
            validate = validadeOutput(expectedNumberObj, yOutput)
            if i != 0 and i % 1000 == 0:
                graphData = np.append(graphData,[[i,(errorCount / i * 100)]],axis=0)
            if i != 0 and i % 30000 == 0:   
                saveGraph(epoch,i)
                print(f'last Target: {expectedNumberObj} \nlast output: {yOutput}')
                print(f'{i} linhas da epoca {epoch}, Erros até aqui {errorCount}')
                print(f'pct Acerto {100 - (errorCount / i) * 100}')
                break
            if(validate == False):
                errorCount+= 1;
                if train:
                    propagateError(expectedNumberObj, row, yOutput, zOutput, zIn, yIn)
        errors.append(errorCount)
        saveGraphEpoch(epoch + 1)
        saveWeigthsToCsv()
        print('Error: ' + str(errorCount))


def calcZIn(row): 
    result = np.zeros(len(row)) 
    for j in range(middleLayerSize):
        result[j] = np.sum(np.dot(row , np.array(v[:,j])))
    return result

def calcYIn(zOutput): 
    result = np.zeros(len(zOutput))
    for j in range(outputSize):
        result[j] = np.sum(np.dot(zOutput, np.array(w[:,j])))
    return result

def calcDelta(arr, arrSize):
    deltas = np.zeros(arrSize)
    for i in range(arrSize):
        deltas[i] = (activationFunction(arr[i]))
    return deltas

def activationFunction(x):
    return 1.0 / (1.0 + np.exp(-x))

def validadeOutput(targetObj, yOutput):
    indTar = np.unravel_index(np.argmax(targetObj, axis=None), targetObj.shape)
    indOut = np.unravel_index(np.argmax(yOutput, axis=None), yOutput.shape)
    if (indTar == indOut):
        return True
    else :
        return False

def propagateError(expectedArr, row, yOutput, zOutput, zIn, yIn):
    global v
    global w
    errorY = calcError(expectedArr, yOutput, yIn, outputSize)
    errorW = calcWeightCorrection(errorY, zOutput, middleLayerSize, outputSize)
    
    sumError = sumDelta(errorY, w, middleLayerSize, outputSize)
    errorZ = calcErrorZ(sumError, zOutput, zIn, middleLayerSize)
    errorV = calcWeightCorrection(errorZ, row, inputSize, middleLayerSize)
    
    #Atualiza Peso
    w = np.add(w,errorW)
    v = np.add(v,errorV)
    

def calcError(expectedArr, outputArr, inArr, size):
    error = np.zeros(size)
    for i in range(size):
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

#leitura de csv e inicio das variaveis
inputCsv = pd.read_csv('mnist_train.csv')
inputData = []
inputNumber = []



# Global variables
outputDictionary = {'0':[1.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0], '1':[0.0,1.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0],
                '2':[0.0,0.0,1.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0], '3':[0.0,0.0,0.0,1.0,0.0,0.0,0.0,0.0,0.0,0.0], '4':[0.0,0.0,0.0,0.0,1.0,0.0,0.0,0.0,0.0,0.0], 
                '5':[0.0,0.0,0.0,0.0,0.0,1.0,0.0,0.0,0.0,0.0], '6':[0.0,0.0,0.0,0.0,0.0,0.0,1.0,0.0,0.0,0.0], '7':[0.0,0.0,0.0,0.0,0.0,0.0,0.0,1.0,0.0,0.0], 
                '8':[0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,1.0,0.0], '9':[0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,1.0] }
#Grafico


try:
    os.mkdir("epocas")
except:
    print("ja existe esta pasta")

learningRate = 0.2
middleLayerSize = 100
outputSize = 10
inputSize = 784
graphData = np.array([[0,100]])

v = np.random.randn(28*28, middleLayerSize) / np.sqrt(28*28) # np.random.uniform(-1.00, 1.00,(inputSize, middleLayerSize)) # [linhas, middleLayerSize]
w = np.random.randn(middleLayerSize, outputSize) / np.sqrt(middleLayerSize) #np.random.uniform(-1.00, 1.00,(middleLayerSize, outputSize)) # [middleLayerSize, outputSize]

errors = []

def prepareData():
    i = 1
    for row in inputCsv.itertuples(index=False):
        arrRow = list(row)
        inputNumber.append(arrRow.pop(0))
        arrRow = np.array(arrRow) / 255
        inputData.append(arrRow)
        
def iniciaGraficos():
    global fig,ax1,fig2,ax2
    
    fig = plt.figure(figsize=(20,4))
    ax1 = fig.add_subplot(1,1,1, title='Porcentagem de Erros por linha', xLabel='Linhas',yLabel='Porcentagem de Erro', yLim=[0,100])
    fig2 = plt.figure(figsize=(20,4))
    ax2 = fig2.add_subplot(1,1,1, title='Erro por épocas', xLabel='epocas',yLabel='porcentagem de erro', yLim=[0,100])
        
def train(maxEpochs, train = False):
    global errors
    global graph
    global inputData
    global graphData
    for epoch in range(maxEpochs):
        errorCount = 0
        print('Period ' + str(epoch + 1))
        graphData = np.array([[0,100]])
        for i in range(len(inputData)):
            row = inputData[i]
            expectedNumber = inputNumber[i]
            expectedNumberObj = np.array(outputDictionary[str(expectedNumber)])
            zIn = calcZIn(np.array(row))
            zOutput = calcDelta(np.array(zIn), middleLayerSize)
            yIn = calcYIn(zOutput)  
            yOutput = calcDelta(yIn, outputSize)  
            validate = validadeOutput(expectedNumberObj, yOutput)
            if i != 0 and i % 1000 == 0:
                graphData = np.append(graphData,[[i,(errorCount / i * 100)]],axis=0)
            if i != 0 and i % 30000 == 0:   
                saveGraph(epoch,i)
                print(f'last Target: {expectedNumberObj} \nlast output: {yOutput}')
                print(f'{i} linhas da epoca {epoch}, Erros até aqui {errorCount}')
                print(f'pct Acerto {100 - (errorCount / i) * 100}')
                break
            if(validate == False):
                errorCount+= 1;
                if train:
                    propagateError(expectedNumberObj, row, yOutput, zOutput, zIn, yIn)
        errors.append(errorCount)
        saveGraphEpoch(epoch + 1)
        saveWeigthsToCsv()
        print('Error: ' + str(errorCount))


def calcZIn(row): 
    result = np.zeros(len(row)) 
    for j in range(middleLayerSize):
        result[j] = np.sum(np.dot(row , np.array(v[:,j])))
    return result

def calcYIn(zOutput): 
    result = np.zeros(len(zOutput))
    for j in range(outputSize):
        result[j] = np.sum(np.dot(zOutput, np.array(w[:,j])))
    return result

def calcDelta(arr, arrSize):
    deltas = np.zeros(arrSize)
    for i in range(arrSize):
        deltas[i] = (activationFunction(arr[i]))
    return deltas

def activationFunction(x):
    return 1.0 / (1.0 + np.exp(-x))

def validadeOutput(targetObj, yOutput):
    indTar = np.unravel_index(np.argmax(targetObj, axis=None), targetObj.shape)
    indOut = np.unravel_index(np.argmax(yOutput, axis=None), yOutput.shape)
    if (indTar == indOut):
        return True
    else :
        return False

def propagateError(expectedArr, row, yOutput, zOutput, zIn, yIn):
    global v
    global w
    errorY = calcError(expectedArr, yOutput, yIn, outputSize)
    errorW = calcWeightCorrection(errorY, zOutput, middleLayerSize, outputSize)
    
    sumError = sumDelta(errorY, w, middleLayerSize, outputSize)
    errorZ = calcErrorZ(sumError, zOutput, zIn, middleLayerSize)
    errorV = calcWeightCorrection(errorZ, row, inputSize, middleLayerSize)
    
    #Atualiza Peso
    w = np.add(w,errorW)
    v = np.add(v,errorV)
    

def calcError(expectedArr, outputArr, inArr, size):
    error = np.zeros(size)
    for i in range(size):
        error[i] = (((expectedArr[i] - outputArr[i]) * (outputArr[i] * (1.0 - outputArr[i]))))
    return error

def calcErrorZ(expectedArr, outputArr, inArr, size):
    error = np.zeros(size)
    for i in range(size):
        error[i] = expectedArr[i] * (outputArr[i] * (1.0 - outputArr[i]))
    return error


def calcWeightCorrection(error, output, lenght1, length2):
    delta = np.array([[0]*lenght1])
    first = True
    for i in range(length2):
        if first:
            delta[i] = learningRate * (output * error[i])
            first = False
        else:
            delta = np.append(delta,np.array([(learningRate * output * error[i])]),axis=0)
    return delta.T

def sumDelta(error, weights, lenght1, length2):
    delta = np.zeros(lenght1);
    for i in range(lenght1):    
        for j in range(length2):
            delta[i] += error[j] * weights[i][j];
    
    return delta

def saveWeigthsToCsv():
    pd.DataFrame(w).to_csv(r"wWweigths.csv")
    pd.DataFrame(w).to_csv(r"vWeigths.csv")

#Funções Graficas
def saveGraph(epoca,linha):
    fig = plt.figure(figsize=(10,6))
    ax1 = fig.add_subplot(1,1,1, title='Porcentagem de Erros por linha', xLabel='Linhas',yLabel='Porcentagem de Erro', yLim=[0,100])
    ax1.plot(graphData[:,0],graphData[:,1])
    fig.savefig(r"epocas\epoca {0} linha {1}.png".format(epoca,linha))
    plt.close(fig)
    
def saveGraphEpoch(epoca):
    fig2 = plt.figure(figsize=(10,6))
    ax2 = fig2.add_subplot(1,1,1, title='Erro por épocas', xLabel='epocas',yLabel='porcentagem de erro')
    ax2.plot(np.arange(epoca),errors)
    fig2.savefig(r"epocas\geralEpocas.png")
    plt.close(fig2)

prepareData()
#(Epocas, treino = true)
train(200,True)
