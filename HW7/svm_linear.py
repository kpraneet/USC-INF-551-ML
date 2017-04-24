import numpy as np
import matplotlib.pyplot as plt
from cvxopt import matrix, solvers


#---------------------------------function to prepare the data in the required format-----------------------------------

def prepare_data(data):
    coordinates = []
    labels = []
    for index in range(len(data)):
        coordinates.append((data[:,0][index], data[:,1][index]))
        labels.append(data[:,2][index])
    coordinates = np.array(coordinates)
    labels = np.array(labels)
    number_of_coordinates = len(coordinates)
    reshaped_label = np.reshape(labels,(number_of_coordinates, 1))
    product_of_coordinates_and_labels = reshaped_label * coordinates
    dot_product_of_coordinates_mixed_with_labels = np.dot(product_of_coordinates_and_labels, product_of_coordinates_and_labels.T)
    P = matrix(dot_product_of_coordinates_mixed_with_labels)
    q = matrix(np.negative(np.ones(number_of_coordinates)))
    G = matrix(np.negative(np.identity(number_of_coordinates)))
    h = matrix(np.zeros(number_of_coordinates))
    A = matrix(np.reshape(labels, (1, -1)))
    b = matrix(0.0)
    return P, q, G, h, A, b, coordinates, labels

def plot_data_with_labels(coordinates, labels, plot):
    for label_index, label in enumerate(labels):
        if label > 0:
            plot.scatter(coordinates[label_index][0], coordinates[label_index][1], c='r')
        else:
            plot.scatter(coordinates[label_index][0], coordinates[label_index][1], c='b')

def plot_separator(plot, weight, bias):
    slope = -weight[0] / weight[1]
    intercept = -bias / weight[1]
    x = np.arange(0, 2)
    line_equation =  "Y= ( "+str(slope) + " * X ) - "+ str(-intercept)
    plot.plot(x, x * slope + intercept, 'k-')
    return line_equation

#------------------------------------function which performs quadratic programming--------------------------------------

def quadratic_solver(P, q, G, h, A, b):
    sol = solvers.qp(P, q, G, h, A, b)
    alphas = np.array(sol['x'])
    return alphas

#-----------------------------------function which calculates the weight and the bias-----------------------------------

def calculate_weight_and_bias(alphas, labels, coordinates):
    reshaped_label = np.reshape(labels, (len(coordinates), 1))
    weights = np.sum(alphas * reshaped_label * coordinates, axis=0)
    alpha_index_array = []
    optimal_alphas = []
    for alpha_index, alpha in enumerate(alphas):
        if alpha > 1e-4:
            alpha_index_array.append(alpha_index)
            optimal_alphas.append(alpha[0])
    bias = []
    for index in alpha_index_array:
        bias.append(labels[index] - np.dot(coordinates[index], weights))

    return weights, bias[0], optimal_alphas

#---------------------------------------------------main function-------------------------------------------------------

def main():

    # read data from csv format using numpy library
    data = np.genfromtxt('linsep.txt', delimiter=',')
    P, q, G, h, A, b, coordinates, labels = prepare_data(data)
    alphas = quadratic_solver(P, q, G, h, A, b)
    weight, bias, optimal_alphas = calculate_weight_and_bias(alphas, labels, coordinates)

    print("The optimal Alphas are : ", optimal_alphas)
    print("The weights are : ", weight)
    print("Bias is as follows : ", bias)
    # show data and w
    figure, plot = plt.subplots()
    line_equation = plot_separator(plot, weight, bias)
    print("The line equation is :", line_equation)
    plot_data_with_labels(coordinates, labels, plot)
    plt.show()

#--------------------------helper function to indicate to start with main function -------------------------------------

if __name__ == '__main__':
    main()