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


if __name__ == '__main__':
    main()