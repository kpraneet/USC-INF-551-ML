# Code source: Jaques Grobler
# License: BSD 3 clause


import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets, linear_model
import pandas as pd

def main():
    inputfile = open("linear-regression.txt", "r")
    filecontent = inputfile.read().splitlines()
    label = []
    data = []
    for val in filecontent:
        templst = []
        templst.append(float(val.split(',')[0]))
        templst.append(float(val.split(',')[1]))
        label.append(float(val.split(',')[2]))
        data.append(templst)
    # Create linear regression object
    regr = linear_model.LinearRegression()

    # Train the model using the training sets
    regr.fit(data, label)

    # The coefficients
    print('Coefficients: \n', regr.coef_)

if __name__ == '__main__':
    main()