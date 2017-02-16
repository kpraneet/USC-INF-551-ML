# Function: K Means
# -------------
# K-Means is an algorithm that takes in a dataset and a constant
# k and returns k centroids (which define clusters of data in the
# dataset which are similar to one another).
# def kmeans(dataSet, k):
    # Initialize centroids randomly
    # numFeatures = dataSet.getNumFeatures()
    # centroids = getRandomCentroids(numFeatures, k)

    # Initialize book keeping vars.
    # iterations = 0
    # oldCentroids = None

    # Run the main k-means algorithm
    # while not shouldStop(oldCentroids, centroids, iterations):
        # Save old centroids for convergence test. Book keeping.
        # oldCentroids = centroids
        # iterations += 1

        # Assign labels to each datapoint based on centroids
        # labels = getLabels(dataSet, centroids)

        # Assign centroids based on datapoint labels
        # centroids = getCentroids(dataSet, labels, k)

    # We can get the labels too by calling getLabels(dataSet, centroids)
    # return centroids


# Function: Should Stop
# -------------

# def shouldStop(oldCentroids, centroids, iterations):
#     if iterations > MAX_ITERATIONS: return True
#     return oldCentroids == centroids


# Function: Get Labels
# -------------
# Returns a label for each piece of data in the dataset. 
# def getLabels(dataSet, centroids):


# For each element in the dataset, chose the closest centroid.
# Make that centroid the element's label.
# Function: Get Centroids
# -------------
# Returns k random centroids, each of dimension n.
# def getCentroids(dataSet, labels, k):

# Each centroid is the geometric mean of the points that
# have that centroid's label. Important: If a centroid is empty (no points have
# that centroid's label) you should randomly re-initialize it.

#------------------------------------------ CONSTANTS ------------------------------------------------------------------

MAX_ITERATIONS = 1000
#------------------------------------------ get random centroids -------------------------------------------------------

def get_random_centroids(data_points, num_clusters):
    import random
    # data_points = [{'x':1, 'y':2},{'x':3, 'y':4},{'x':5, 'y':6}]
    list_of_centriods = []
    for i in range(num_clusters):
        list_of_centriods.append(random.choice(data_points))
    # print(list_of_centriods)
    return list_of_centriods

#------------------------------------------ condition to terminate k-means ---------------------------------------------

# Returns True or False if k-means is done. K-means terminates either because it has run a maximum number of iterations
# OR the centroids stop changing.

def shouldStop(oldCentroids, centroids, iterations):
    if iterations > MAX_ITERATIONS:
        return True
    return oldCentroids == centroids

#------------------------------------------ condition to terminate k-means ---------------------------------------------

# Returns a label for each piece of data in the dataset.
# For each element in the dataset, chose the closest centroid.
# Make that centroid the element's label.

def get_labels(data_set, centroids):
    import sys
    import math
    updated_data_set = []
    for data_point in data_set:
        x = data_point[x]
        y = data_point[y]
        cx = data_point[lx]
        cy = data_point[ly]
        min_dist = sys.float_info.max
        for centroid in centroids:
            x_c = centroid[x]
            y_c = centroid[y]
            dist = math.hypot(x - x_c, y - y_c)
            if dist < min_dist:
                min_dist = dist
                cx = x_c
                cy = y_c
        updated_data_set.append({'x':x, 'y':y, 'lx':cx, 'ly':cy})
    return updated_data_set

#------------------------------------------ the main method ------------------------------------------------------------


def calculate_centroid(list):
    import numpy as np
    a = np.array(list)
    return np.mean(a, axis=0)

def seperate_coordinates(input_list):
    list = []
    for element in input_list:
        current_element = []
        current_element.append(element[x])
        current_element.append(element[y])
        list.append(current_element)
    return list

def plot_values():
    import pylab as pl
    # Make an array of x values
    x = [1, 2, 3, 4, 5]
    # Make an array of y values for each x value
    y = [1, 4, 9, 16, 25]
    # use pylab to plot x and y as red circles
    pl.plot(x, y, ’s’)
    # show the plot on the screen
    pl.show()
