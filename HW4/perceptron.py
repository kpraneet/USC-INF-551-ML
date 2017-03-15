import matplotlib.pyplot as plt
import pylab
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

def checkconstraints(val,label):
    count = 0
    pending = 0
    for i in range(len(val)):
        if val[i]>0 and label[i]>0:
            count+=1
        elif val[i]<0 and label[i]<0:
            count+=1
        else:
            pending+=1
    return count

def main():
    data = []
    label = []
    w = np.random.rand(3)
    print(w)
    n = 7000
    thr = 0.01
    eta = 0.01
    inputfile = open("classification.txt", "r")
    filecontent = inputfile.read().splitlines()
    for val in filecontent:
        templst = []
        templst.append(float(val.split(',')[0]))
        templst.append(float(val.split(',')[1]))
        templst.append(float(val.split(',')[2]))
        data.append(templst)
        label.append(int(val.split(',')[3]))
    data=np.array(data)
    w=np.array(w)
    for i in range(n):
        val=np.dot(data,w)
        count = checkconstraints(val,label)
        if count==2000:
            break
        else:
            for x in range(len(data)):
                indvalarray=np.array(data[x])
                if((np.dot(indvalarray,w)<0.0) and label[x]==1):
                    tmpval = (eta * indvalarray)
                    w += tmpval
                    # print('If')
                elif((np.dot(indvalarray,w)>0.0) and label[x]==-1):
                    tmpval = (eta * indvalarray)
                    w -= tmpval
                    # print('Else if')
                # else:
                #     print(data[x],val[x])
    print(w)
    print(count)
    print(i)
    # fig = pylab.figure()
    # ax=Axes3D(fig)
    # for x in range(len(data)):
    #     if label[x]==1:
    #         ax.scatter(data[x][0],data[x][1],data[x][2], c='b', marker='o')
    #     elif label[x]==-1:
    #         ax.scatter(data[x][0], data[x][1], data[x][2], c='g', marker='o')
    # ax.plot_surface(w[0],w[1],w[2], color='r')
    # plt.show()

if __name__ == '__main__':
    main()