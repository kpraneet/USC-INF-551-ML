import numpy as np
import matplotlib.pyplot as plt
from cvxopt import matrix, solvers
#--------------------------------------------------kernel function------------------------------------------------------

def kernel(coordinates):
    new_dimensions = []
    for coordinate in range(len(coordinates)):
        temporary_array = [(coordinates[coordinate, 0]) ** 2, (coordinates[coordinate, 1]) ** 2]
        new_dimensions.append(temporary_array)
    new_dimensions = np.array(new_dimensions)
    return new_dimensions
#-----------------------function to prepare the data in the required format for quadratic solver------------------------

def prepare_data(data):
    coordinates = []
    labels = []
    for index in range(len(data)):
        coordinates.append((data[:,0][index], data[:,1][index]))
        labels.append(data[:,2][index])
    coordinates = np.array(coordinates)
    original_data = coordinates
    labels = np.array(labels)
    new_dimension_coordinates = kernel(coordinates)
    coordinates = np.array(new_dimension_coordinates)
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
    return P, q, G, h, A, b, coordinates, labels, original_data

#---------------------------------------helper function to plot the data points-----------------------------------------

def plot_data_points(coordinates, labels, plot):
    plot.scatter(coordinates[:, 0], coordinates[:, 1], c=labels)

#--------------------------------------helper function to plot the seperator line---------------------------------------

def plot_separator(plot, weight, bias):
    slope = -weight[0] / weight[1]
    intercept = -bias / weight[1]
    x = np.arange(0, 110)
    plot.plot(x, x * slope + intercept, 'k-')

#------------------------------------function which performs quadratic programming--------------------------------------

def quadratic_solver(P, q, G, h, A, b):
    sol = solvers.qp(P, q, G, h, A, b)
    alphas = np.array(sol['x'])
    return alphas

#-----------------------------------function which calculates the weight and the bias-----------------------------------

def calculate_weight_and_bias(alphas, labels, coordinates, original_data):
    reshaped_label = np.reshape(labels, (len(coordinates), 1))
    weights = np.sum(alphas * reshaped_label * coordinates, axis=0)
    alpha_index_array = []
    optimal_alphas = []
    for alpha_index, alpha in enumerate(alphas):
        if alpha >1e-4:
            alpha_index_array.append(alpha_index)
            optimal_alphas.append(alpha[0])
    bias = []
    support_vectors = []
    original_support_vectors = []
    for index in alpha_index_array:
        bias.append(labels[index] - np.dot(coordinates[index], weights))
        support_vectors.append(tuple(coordinates[index]))
        original_support_vectors.append(tuple(original_data[index]))
    return weights, bias[0], optimal_alphas, support_vectors, original_support_vectors

#---------------------------------------------------main function-------------------------------------------------------

def main():

    # read data from csv format using numpy library
    data = np.genfromtxt('nonlinsep.txt', delimiter=',')
    P, q, G, h, A, b, coordinates, labels, original_data = prepare_data(data)
    alphas = quadratic_solver(P, q, G, h, A, b)
    weight, bias, optimal_alphas, support_vectors, original_support_vectors = calculate_weight_and_bias(alphas, labels, coordinates, original_data)

    print("The optimal Alphas are : ", optimal_alphas)
    print("The weights are : ", weight)
    print("Bias is as follows : ", bias)
    print("Support Vectors are : ", support_vectors)
    print("Support Vectors in the original data set before application of kernel are : ", original_support_vectors)
    # show data and w
    figure, plot = plt.subplots()
    plot_separator(plot, weight, bias)
    plot_data_points(coordinates, labels, plot)
    plot.set_xlim([-10, 140])
    plot.set_ylim([-10, 140])
    plt.show()

#--------------------------helper function to indicate to start with main function -------------------------------------

if __name__ == '__main__':
    main()