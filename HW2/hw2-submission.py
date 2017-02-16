import random
import math
import numpy as np
import pylab as pl
import sys

k = 3
maxiterations = 1000000

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
    for pointvar in listpoints:
        x = pointvar['x']
        y = pointvar['y']
        mindist = sys.float_info.max
        for centroidvar in centroids:
            cx = centroidvar['x']
            cy = centroidvar['y']
            dist = math.hypot(x-cx,y-cy)
            if dist < mindist:
                pointvar['lx'] = cx
                pointvar['ly'] = cy
                mindist = dist
    return listpoints

def makearray(listvariable):
    nplist = []
    for var in listvariable:
        tmplist =[]
        x = var['x']
        y = var['y']
        tmplist.append(x)
        tmplist.append(y)
        nplist.append(tmplist)
    return nplist

def calcentroid(listvar):
    a = np.array(listvar)
    var = np.mean(a, axis=0)
    return (var)

def getnewcentroid(labelpoint,centroids,k):
    centroidlist1 = []
    centroidlist2 = []
    centroidlist3 = []
    tmpvar = 0
    for centvar in centroids:
        cx = centvar['x']
        cy = centvar['y']
        for var in labelpoint:
            lx = var['lx']
            ly = var['ly']
            if ((cx == lx) and (cy ==ly)):
                if tmpvar == 0:
                    centroidlist1.append(var)
                elif tmpvar == 1:
                    centroidlist2.append(var)
                elif tmpvar == 2:
                    centroidlist3.append(var)
                else:
                    print('Point error!')
        tmpvar += 1
    nparr1=makearray(centroidlist1)
    nparr2=makearray(centroidlist2)
    nparr3=makearray(centroidlist3)
    newcentroid = []
    val1 = calcentroid(nparr1)
    tmpdict = {}
    x = val1[0]
    y = val1[1]
    tmpdict['x'] = x
    tmpdict['y'] = y
    newcentroid.append(tmpdict)
    val2 = calcentroid(nparr2)
    tmpdict = {}
    x = val2[0]
    y = val2[1]
    tmpdict['x'] = x
    tmpdict['y'] = y
    newcentroid.append(tmpdict)
    val3 = calcentroid(nparr3)
    tmpdict = {}
    x = val3[0]
    y = val3[1]
    tmpdict['x'] = x
    tmpdict['y'] = y
    newcentroid.append(tmpdict)
    return newcentroid

def printclusters(listpoints,centroids):
    tmpvar = 0
    for centvar in centroids:
        lxlist = []
        lylist = []
        cx = centvar['x']
        cy = centvar['y']
        for var in listpoints:
            lx = var['lx']
            ly = var['ly']
            if ((cx == lx) and (cy == ly)):
                lxlist.append(var['x'])
                lylist.append(var['y'])
        if tmpvar == 0:
            pl.plot(lxlist,lylist,'rx')
        elif tmpvar == 1:
            pl.plot(lxlist, lylist, 'gx')
        else:
            pl.plot(lxlist, lylist, 'bx')
        tmpvar += 1
        pl.plot(cx, cy, 'ko')
    pl.show()

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
        datadict['lx'] = sys.float_info.min
        datadict['ly'] = sys.float_info.min
        listpoints.append(datadict)
    centroids = getrandomcentrioids(listpoints,k)
    while not stopcheck(oldcentroids,centroids,iterations):
        oldcentroids = centroids
        iterations += 1
        labelpoint = pointlabel(listpoints,centroids)
        centroids = getnewcentroid(labelpoint,centroids,k)
    printclusters(listpoints,centroids)
    print('Centroids for K-means: ')
    for x in centroids:
        print(x)

if __name__ == '__main__':
    main()