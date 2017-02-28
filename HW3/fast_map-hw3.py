import random
import math
import matplotlib.pyplot as pl

tmp = 0
k = 2
columnnum = -1
PA = [[0,0],[0,0]]
Xlist = [[0 for x in range(2)] for y in range(10)]

def projections(pointdistances):
    global tmp
    newpointdistances = [[0 for x in range(10)] for y in range(10)]
    for i in range(0,10):
        for j in range(0,10):
            xival = Xlist[i][tmp]
            xjval = Xlist[j][tmp]
            newpointdistances[i][j] = math.sqrt(((pointdistances[i][j])**2)-((xival-xjval)**2))
    tmp += 1
    return newpointdistances

def choose_distance_objects(pointdistances):
    max = -1
    a = -1
    b = random.randint(0,9)
    for i in range(0,10):
        newmax = pointdistances[i][b]
        if newmax > max:
            max = newmax
            a = i
    b = -1
    max = -1
    for i in range(0,10):
        newmax = pointdistances[a][i]
        if newmax > max:
            max = newmax
            b = i
    return a,b,max

def fastmap(pointdistances):
    global columnnum
    global k
    if k<= 0:
        return
    else:
        columnnum += 1
    a,b,dist = choose_distance_objects(pointdistances)
    PA[0][columnnum] = a
    PA[1][columnnum] = b
    if dist == 0:
        for i in range(0,10):
            Xlist[i][columnnum] = 0
            return
    for x in range(0,10):
        newx = (((pointdistances[a][x])**2)+(dist**2)-((pointdistances[b][x])**2))/(2*dist)
        Xlist[x][columnnum] = newx
    updatedpointdistances = projections(pointdistances)
    k -= 1
    fastmap(updatedpointdistances)

def main():
    xlist = []
    ylist = []
    labellist = []
    pointdistances = [[0 for x in range(10)] for y in range(10)]
    tmplist = []
    inputfile = open("fastmap-data.txt", "r")
    filecontent = inputfile.read().splitlines()
    for val in filecontent:
        inputvalue = val.split('\t')
        tmplist.append(inputvalue)
    for x in tmplist:
        x[0] = int(x[0])
        x[1] = int(x[1])
        x[2] = int(x[2])
    for x in tmplist:
        pointdistances[x[0]-1][x[1]-1]=x[2]
        pointdistances[x[1]-1][x[0]-1] = x[2]
    fastmap(pointdistances)
    # Crosscheck choosing of distant points logic
    # Crosscheck fast map correctness with TA
    print('Pivots: ')
    for x in PA:
        print(x)
    print('Values: ')
    for x in Xlist:
        print(x)
        xlist.append(x[0])
        ylist.append(x[1])
    newinputfile = open("fastmap-wordlist.txt","r")
    content = newinputfile.read().splitlines()
    for val in content:
        labellist.append(val)
    pl.plot(xlist,ylist,'ro')
    for x in range(0,10):
        pl.text(xlist[x],ylist[x],labellist[x])
    pl.show()

if __name__ == '__main__':
    main()