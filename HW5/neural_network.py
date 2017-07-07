# Authors: Karthik Ravindra Rao - raokarth@usc.edu & Praneet Kalluri - pkalluri@usc.edu

import numpy as np
from PIL import Image
import random
from scipy import special
import time


inweight = [[random.uniform(-0.1, 0.1) for x in range(960)] for y in range(100)]
outweight = [random.uniform(-0.1, 0.1) for z in range(100)]
inputnodes = [0 for i in range(960)]
hiddennodes = [0 for j in range(100)]
predictioncount = 0


def sigmoid(x):
    return special.expit(x)


def dersig(x):
    y = sigmoid(x)
    checkval = y * (1 - y)
    return checkval


def feedforwardbackpropogation(label):
    global inputnodes
    global hiddennodes
    global inweight
    global outweight
    eta = 0.1
    opval = 0
    hiddendelta = []
    inputnodes = np.array(inputnodes)
    inweight = np.array(inweight)
    for i in range(100):
        tmp = 0
        tmp += np.dot(inputnodes, inweight[i])
        hiddennodes[i] = tmp
    for x in range(100):
        opval += (sigmoid(hiddennodes[x]) * outweight[x])
    errval = label - sigmoid(opval)
    delta = dersig(opval) * errval
    for x in range(100):
        temp = (delta * outweight[x] * dersig(hiddennodes[x]))
        hiddendelta.append(temp)
        outweight[x] += (eta * sigmoid(hiddennodes[x]) * delta)
    for y in range(100):
        val = eta * hiddendelta[y]
        newlst = np.dot(inputnodes, val)
        inweight[y] = inweight[y] + newlst


def prediction(label):
    global inputnodes
    global hiddennodes
    global predictioncount
    global inweight
    opval = 0
    for i in range(100):
        tmp = 0
        tmp += np.dot(inputnodes, inweight[i])
        hiddennodes[i] = tmp
    for x in range(100):
        opval += (sigmoid(hiddennodes[x]) * outweight[x])
    y = sigmoid(opval)
    if y >= 0.5 and label == 1:
        predictioncount += 1
        print('Correct classification')
    elif y < 0.5 and label == 0:
        predictioncount += 1
        print('Correct classification')
    else:
        print('Incorrect classification')


def main():
    localtime = time.asctime(time.localtime(time.time()))
    print(localtime)
    print('\n')
    global inputnodes
    global hiddennodes
    global inweight
    global predictioncount
    epochs = 1000
    filecount = 0
    count = 0
    inputfile = open("downgesture_train.list", "r")
    filecontent = inputfile.read().splitlines()
    val = filecontent[0]
    if 'down' in val:
        label = 1
    else:
        label = 0
    img = Image.open(val)
    lst = np.array(img)
    for var in lst:
        for x in var:
            inputnodes[count] = int(x)
            count += 1
    for i in range(100):
        tmp = 0
        for j in range(960):
            tmp += (inputnodes[j] * inweight[i][j])
        hiddennodes[i] = tmp
    feedforwardbackpropogation(label)
    imglst = []
    for val in range(0, len(filecontent)):
        img = Image.open(filecontent[val])
        imglst.append(np.array(img))
    while epochs:
        epochs -= 1
        print('Epoch: ',1000 - epochs)
        for val in range(0, len(filecontent)):
            if 'down' in filecontent[val]:
                label = 1
            else:
                label = 0
            count = 0
            for var in imglst[count]:
                for x in var:
                    inputnodes[count] = int(x)
                    count += 1
            feedforwardbackpropogation(label)
    # Prediction
    inputfile = open("downgesture_test.list", "r")
    filecontent = inputfile.read().splitlines()
    for val in range(0, len(filecontent)):
        filecount += 1
        if 'down' in filecontent[val]:
            label = 1
        else:
            label = 0
        print('File name: ',filecontent[val],'Label: ',label)
        img = Image.open(filecontent[val])
        lst = np.array(img)
        count = 0
        for var in lst:
            for x in var:
                inputnodes[count] = int(x)
                count += 1
        prediction(label)
    print('\n')
    print('Correctly classified: ',predictioncount)
    print('Total files: ',filecount)
    print('Accuracy: ',(predictioncount / filecount) * 100)
    localtime = time.asctime(time.localtime(time.time()))
    print('\n')
    print(localtime)


if __name__ == '__main__':
    main()
