import numpy as np
from PIL import Image
import random
from scipy import special


predictioncount = 0


def sigmoid(x):
    return special.expit(x)


def dersig(x):
    checkval = sigmoid(x) * (1 - sigmoid(x))
    return checkval
    # val = 1/(np.exp(x)+1)
    # newval = val * val
    # return val - newval
    # return exp(x)/((exp(x)+1)*(exp(x)+1))
    # return np.exp(x)/((1+np.exp(x))*(1+np.exp(x)))


class hiddenlayernode:
    def __init__(self, val, act):
        self.nodevalue = val
        self.finalval = act
        self.outweight = random.uniform(-0.1, 0.1)
        # self.inweight = random.randint(-1000,1000)


class inputlayernode:
    def __init__(self, lst):
        self.val = lst
        self.weight = [random.uniform(-0.1, 0.1) for x in range(100)]
        # self.weight = np.random.rand(100)


class opnode:
    def __init__(self, val, act):
        self.nodevalue = val
        self.finalval = act
        # self.inweight = []


def feedforwardbackpropogation(inputnodes, label):
    eta = 0.1
    epochs = 1000
    hiddennodes = []
    hiddendelta = []
    while epochs:
        epochs -= 1
        for i in range(100):
            tmp = 0
            for j in inputnodes:
                # print(j.val,j.weight[i])
                tmp += (j.val*j.weight[i])
            tmp *= eta
            # print(tmp, sigmoid(tmp))
            hiddentmp = hiddenlayernode(tmp, sigmoid(tmp))
            hiddennodes.append(hiddentmp)
        opval = 0
        for x in hiddennodes:
            # print(x.nodevalue, x.finalval, x.outweight)
            opval += (x.finalval*x.outweight)
        opval *= eta
        # print(opval, sigmoid(opval))
        outputnode = opnode(opval, sigmoid(opval))
        errval = label - sigmoid(opval)
        # print(errval)
        if errval >= -0.04 and errval <= 0.04:
            break
        delta = dersig(outputnode.nodevalue) * errval
        # print(delta)
        for x in hiddennodes:
            temp = (delta * x.outweight * dersig(x.nodevalue))
            hiddendelta.append(temp)
            x.outweight += (eta * x.finalval * delta)
        # print(len(hiddendelta))
        for x in inputnodes:
            for y in range(100):
                x.weight[y] += (eta * x.val * hiddendelta[y])
    # print(outputnode.nodevalue, outputnode.finalval)
    return inputnodes


def prediction(inputnodes, label):
    global predictioncount
    eta = 0.1
    hiddennodes = []
    for i in range(100):
        tmp = 0
        for j in inputnodes:
            # print(j.val,j.weight[i])
            tmp += (j.val * j.weight[i])
        tmp *= eta
        # print(tmp, sigmoid(tmp))
        hiddentmp = hiddenlayernode(tmp, sigmoid(tmp))
        hiddennodes.append(hiddentmp)
    opval = 0
    for x in hiddennodes:
        # print(x.nodevalue, x.finalval, x.outweight)
        opval += (x.finalval * x.outweight)
    opval *= eta
    # print(opval, sigmoid(opval))
    # outputnode = opnode(opval, sigmoid(opval))
    if sigmoid(opval) >= 0.5 and label == 1:
        predictioncount += 1
    if sigmoid(opval) < 0.5 and label == 0:
        predictioncount += 1
    return inputnodes


def main():
    inputfile = open("downgesture_train.list", "r")
    filecontent = inputfile.read().splitlines()
    global predictioncount
    predictionfilecount = 0
    inputnodes = []
    val = filecontent[0]
    if 'down' in val:
        label = 1
    else:
        label = 0
    print('File name: ', val, 'Label: ', label)
    img = Image.open(val)
    lst = np.array(img)
    for var in lst:
        for x in var:
            tmpvar = inputlayernode(int(x))
            inputnodes.append(tmpvar)
    inputnodes = feedforwardbackpropogation(inputnodes, label)

    tmplst = []
    for x in range(1, len(filecontent)):
        tmplst.append(x)
    random.shuffle(tmplst)
    # print(tmplst)
    # print(len(tmplst))
    # for val in tmplst:

    for val in range(1, len(filecontent)):
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
        inputnodes = feedforwardbackpropogation(inputnodes, label)
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
        inputnodes = prediction(inputnodes, label)
    print(predictioncount, predictionfilecount, (predictioncount / predictionfilecount) * 100)


if __name__ == '__main__':
    main()
