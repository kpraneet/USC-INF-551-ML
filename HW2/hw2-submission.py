import random

k = 3
maxiterations = 1000

def getrandomcentrioids(listpoints,k):
    centroidlist = []
    for x in range(k):
        val = random.choice(listpoints)
        centroidlist.append(val)
    return centroidlist

def stopcheck(oldcentroids,updatedcentroids,iterations):
    if iterations > maxiterations:
        return True
    return oldcentroids == updatedcentroids

def pointlabel(listpoints,centroids):
    print('Blah')

def main():
    listpoints = []
    iterations = 0
    oldcentroids = None
    inputfile = open("clusters.txt","r")
    filecontent = inputfile.read().splitlines()
    for val in filecontent:
        datadict = {}
        indval = val.split(',')
        datadict['x'] = float(indval[0])
        datadict['y'] = float(indval[1])
        listpoints.append(datadict)
    centroids = getrandomcentrioids(listpoints,k)
    while not stopcheck(oldcentroids,centroids,iterations):
        oldcentroids = centroids
        iterations += 1
        labelpoint = pointlabel(listpoints,centroids)


if __name__ == '__main__':
    main()