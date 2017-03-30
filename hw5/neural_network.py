import numpy as np
from PIL import Image
import random
from scipy import special
import time


predictioncount = 0


def sigmoid(x):
    return special.expit(x)


def dersig(x):
    y = sigmoid(x)
    checkval = y * (1 - y)
    return checkval


class hiddenlayernode:
    def __init__(self, val, act):
        self.nodevalue = val
        self.finalval = act
        self.outweight = random.uniform(-0.1, 0.1)


class inputlayernode:
    def __init__(self, lst):
        self.val = lst
        self.weight = [random.uniform(-0.1, 0.1) for x in range(100)]


class opnode:
    def __init__(self, val, act):
        self.nodevalue = val
        self.finalval = act


def feedforwardbackpropogation(inputnodes, hiddennodes, label):
    eta = 0.1
    hiddendelta = []
    for i in range(100):
        tmp = 0
        for j in inputnodes:
            tmp += (j.val*j.weight[i])
        hiddennodes[i].nodevalue = tmp
        hiddennodes[i].finalval = sigmoid(tmp)
    opval = 0
    for x in hiddennodes:
        opval += (x.finalval*x.outweight)
    outputnode = opnode(opval, sigmoid(opval))
    errval = label - sigmoid(opval)
    delta = dersig(outputnode.nodevalue) * errval
    for x in hiddennodes:
        temp = (delta * x.outweight * dersig(x.nodevalue))
        hiddendelta.append(temp)
        x.outweight += (eta * x.finalval * delta)
    for x in inputnodes:
        for y in range(100):
            x.weight[y] += (eta * x.val * hiddendelta[y])
    return inputnodes, hiddennodes


def prediction(inputnodes, hiddennodes, label):
    global predictioncount
    for i in range(100):
        tmp = 0
        for j in inputnodes:
            tmp += (j.val * j.weight[i])
        hiddennodes[i].nodevalue = tmp
        hiddennodes[i].finalval = sigmoid(tmp)
    opval = 0
    for x in hiddennodes:
        opval += (x.finalval * x.outweight)
    y = sigmoid(opval)
    if y >= 0.5 and label == 1:
        predictioncount += 1
    if y < 0.5 and label == 0:
        predictioncount += 1
    return inputnodes, hiddennodes


def main():
    localtime = time.asctime(time.localtime(time.time()))
    print(localtime)
    global predictioncount
    predictionfilecount = 0
    inputnodes = []
    hiddennodes = []
    epochs = 1000
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
            tmpvar = inputlayernode(int(x))
            inputnodes.append(tmpvar)
    for i in range(100):
        tmp = 0
        for j in inputnodes:
            tmp += (j.val*j.weight[i])
        hiddentmp = hiddenlayernode(tmp, sigmoid(tmp))
        hiddennodes.append(hiddentmp)
    inputnodes, hiddennodes = feedforwardbackpropogation(inputnodes, hiddennodes, label)
    # tmplst = []
    # for x in range(1, len(filecontent)):
    #     tmplst.append(x)
    # random.shuffle(tmplst)
    imglst = []
    for val in range(0, len(filecontent)):
        img = Image.open(filecontent[val])
        imglst.append(np.array(img))
    while epochs:
        epochs -= 1
        print('Epoch: ', 1000 - epochs)
        for val in range(0, len(filecontent)):
            if 'down' in filecontent[val]:
                label = 1
            else:
                label = 0
            # img = Image.open(filecontent[val])
            # lst = np.array(img)
            count = 0
            for var in imglst[count]:
                for x in var:
                    inputnodes[count].val = int(x)
                    count += 1
            inputnodes, hiddennodes = feedforwardbackpropogation(inputnodes, hiddennodes, label)
    # Prediction
    inputfile = open("downgesture_test.list", "r")
    filecontent = inputfile.read().splitlines()
    for val in range(0, len(filecontent)):
        predictionfilecount += 1
        if 'down' in filecontent[val]:
            label = 1
        else:
            label = 0
        print('File name: ', filecontent[val], 'Label: ', label)
        img = Image.open(filecontent[val])
        lst = np.array(img)
        count = 0
        for var in lst:
            for x in var:
                inputnodes[count].val = int(x)
                count += 1
        inputnodes, hiddennodes = prediction(inputnodes, hiddennodes, label)
    print(predictioncount, predictionfilecount, (predictioncount / predictionfilecount) * 100)
    localtime = time.asctime(time.localtime(time.time()))
    print(localtime)


if __name__ == '__main__':
    main()
