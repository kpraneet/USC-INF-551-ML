import random

k = 3
#changing
def getrandomcentrioids(listpoints,k):
    centroidlist = []
    for x in range(k):
        val = random.choice(listpoints)
        centroidlist.append(val)
    return centroidlist

def main():
    listpoints = []
    inputfile = open("clusters.txt","r")
    filecontent = inputfile.read().splitlines()
    for val in filecontent:
        datadict = {}
        indval = val.split(',')
        datadict['x'] = float(indval[0])
        datadict['y'] = float(indval[1])
        listpoints.append(datadict)
    # for x in listpoints:
    #     print(x)
    initialcentroids = getrandomcentrioids(listpoints,k)
    print(initialcentroids)

if __name__ == '__main__':
    main()