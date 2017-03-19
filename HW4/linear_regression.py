import numpy as np

def main():
    data = []
    zlist = []
    inputfile = open("linear-regression.txt", "r")
    filecontent = inputfile.read().splitlines()
    for val in filecontent:
        tmplist = []
        tmplist.append(float(1.0))
        tmplist.append(float(val.split(',')[0]))
        tmplist.append(float(val.split(',')[1]))
        zlist.append(float(val.split(',')[2]))
        data.append(tmplist)
    data = np.array(data)
    zlist = np.array(zlist)
    tmpdata = np.linalg.inv(np.dot((data.T),data))
    updateddata = np.dot(tmpdata,(data.T))
    w = np.dot(updateddata,zlist)
    print('Weights: ')
    print(w[0],',',w[1],',',w[2])

if __name__ == '__main__':
    main()