import numpy as np
import matplotlib.pyplot as plt

def sigmoid(x):
    return np.exp(x)/(1+np.exp(x))

def checkconstrainsts(data,label,w):
    count = 0
    pending = 0
    for i in range(len(data)):
        val = np.dot(data[i], w)
        prob = sigmoid(val)
        if prob<0.5 and label[i]==-1:
            count+=1
        elif prob>=0.5 and label[i]==1:
            count+=1
        else:
            pending+=1
    return count, pending

def main():
    data = []
    label = []
    n = 7000
    eta = 0.01
    w = np.random.rand(4)
    xlist = []
    ylist = []
    lr_count = 0
    lr_w = []
    lr_i = 0
    print('Randomly initialized weights: ')
    print(w)
    inputfile = open("classification.txt", "r")
    filecontent = inputfile.read().splitlines()
    for val in filecontent:
        templst = []
        templst.append(float(1.0))
        templst.append(float(val.split(',')[0]))
        templst.append(float(val.split(',')[1]))
        templst.append(float(val.split(',')[2]))
        data.append(templst)
        label.append(int(val.split(',')[4]))
    for i in range(n):
        count, pending = checkconstrainsts(data, label, w)
        xlist.append(i)
        ylist.append(pending)
        if count > lr_count:
            lr_count = count
            lr_w = w
            lr_i = i
        if count==2000:
            break
        else:
            for x in range(len(data)):
                prediction = sigmoid(np.dot(data[x],w))
                diffval = (label[x] - prediction) * prediction * (1-prediction)
                addval = eta * (np.dot(diffval,data[x]))
                if prediction>=0.5 and label[x]==-1:
                    w += addval
                if prediction<0.5 and label[x]==1:
                    w += addval
    print('Weights where we had least constraints violated: ')
    print(lr_w)
    print('Least amount of constraints violated: ')
    print(2000-lr_count)
    print('Iteration where least number of constraints were violated: ')
    print(lr_i)
    plt.plot(xlist, ylist)
    plt.xlabel('Number of Iterations')
    plt.ylabel('Violated constraints')
    plt.show()

if __name__ == '__main__':
    main()