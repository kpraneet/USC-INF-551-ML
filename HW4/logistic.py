import matplotlib.pyplot as plt
import numpy as np
from sklearn import linear_model
import pandas as pd

def main():
    inputfile = open("classification.txt", "r")
    filecontent = inputfile.read().splitlines()
    label = []
    data = []
    for val in filecontent:
        templst = []
        templst.append(float(val.split(',')[0]))
        templst.append(float(val.split(',')[1]))
        templst.append(float(val.split(',')[2]))
        label.append(float(val.split(',')[3]))
        data.append(templst)
    # Create linear regression object
    regr = linear_model.Perceptron()

    # Train the model using the training sets
    regr.fit(data, label)

    # The coefficients
    print('Coefficients: \n', regr.coef_)

if __name__ == '__main__':
    main()