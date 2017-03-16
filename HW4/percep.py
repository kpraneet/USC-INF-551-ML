<<<<<<< HEAD
import pandas as pd
import matplotlib.pyplot as plt
import pylab
from mpl_toolkits.mplot3d import Axes3D
from random import choice
from numpy import array, dot, random

def main():
    input = pd.read_csv("classification.txt", sep=",", header=None)
    del input[4]
    input.insert(0, '-1', 1)
    for j in range (len(input[0])):
        if input [3][j]== -1:
            input.set_value(j, 3, 0)
    print(input)
    fig = pylab.figure()
    ax=Axes3D(fig)
    for x in range(len(input[3])):
        if input[3][x]==1:
            ax.scatter(input[0][x],input[1][x],input[2][x],c='r', marker='o')
        else:
            ax.scatter(input[0][x], input[1][x], input[2][x],c='b',marker='o')

    unit_step = lambda x:0 if x<0 else 1

    weights = [-0.01, 0.22750441, 0.1735722, 0.6444108] #random.rand(4)
    # weights = array([0.01,0.01,0.01,0.01])
    print("initial weights")
    print(weights)

    eta = 0.01

    for i in range(len(input[0])): #len(input[0])
        random_var = input.sample()
        label = random_var[3].values[0]
        # print(label)
        data = array([random_var['-1'].values[0],random_var[0].values[0],random_var[1].values[0],random_var[2].values[0]])
        result = dot(data, weights)
        # print(result)
        error = label - unit_step(result)
        # print(error)
        weights += eta * data * error

    yvalues = []
    xvalue = []
    for j in range(5):
        xvalues = random.rand(3)
        xvalue.append(xvalues)
        yvalues.append(weights[0]+weights[1]*xvalues[0]+weights[2]*xvalues[1]+weights[3]*xvalues[2])
    print(xvalue)



    print("final weights")
    print(weights)
    # ax.plot(weights)

    plt.show()

=======
import numpy as np

def checkconstraints(val,label):
    count = 0
    pending = 0
    for i in range(len(val)):
        if val[i]>0 and label[i]>0:
            count+=1
        elif val[i]<0 and label[i]<0:
            count+=1
        else:
            pending+=1
    return count

def main():
    data = []
    label = []
    tmp = np.random.rand(3)
    thr = 0.01
    w = [thr,tmp[0],tmp[1],tmp[2]]
    n = 10000
    eta = 0.01
    totalviolated = 0
    print('Randomly initialized weights: ')
    print(w[0],',',w[1],',',w[2],',',w[3])
    inputfile = open("classification.txt", "r")
    filecontent = inputfile.read().splitlines()
    for val in filecontent:
        templst = []
        templst.append(float(1.0))
        templst.append(float(val.split(',')[0]))
        templst.append(float(val.split(',')[1]))
        templst.append(float(val.split(',')[2]))
        data.append(templst)
        label.append(int(val.split(',')[3]))
    data=np.array(data)
    w=np.array(w)
    for i in range(n):
        val=np.dot(data,w)
        count = checkconstraints(val,label)
        if count==2000:
            break
        else:
            for x in range(len(data)):
                indvalarray=np.array(data[x])
                if((np.dot(indvalarray,w)<0.0) and label[x]==1):
                    tmpval = (eta * indvalarray)
                    w += tmpval
                elif((np.dot(indvalarray,w)>0.0) and label[x]==-1):
                    tmpval = (eta * indvalarray)
                    w -= tmpval
                else:
                    totalviolated += 1
    print('Final weights: ')
    print(w[0],',',w[1],',',w[2],',',w[3])
    print('Number of constraints satisfied: ')
    print(count)
    print('Number of iterations: ')
    print(i)
>>>>>>> a1bdd60ccd1cef0cce1dc4db921e6c67dad9eaee

if __name__ == '__main__':
    main()