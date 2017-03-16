import numpy as np
import matplotlib.pyplot as plt

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
    return count,pending

def main():
    data = []
    label = []
    tmp = np.random.rand(3)
    thr = 0.01
    w = [thr,tmp[0],tmp[1],tmp[2]]
    n = 7000
    eta = 0.01
    totalviolated = 0
    xlist = []
    ylist = []
    pa_count = 0
    pa_w = []
    pa_i = 0
    print('Randomly initialized weights: ')
    print(w[0],',',w[1],',',w[2],',',w[3])
    inputfile = open("classification.txt", "r")
    filecontent = inputfile.read().splitlines()
    for val in filecontent:
        templst = []
        templst.append(float(1.0))
        templst.append(float(val.split(',')[0]))
        templst.append(float(val.split(',')[1]))
        templst.append(float(val.split(',')[2]))
        data.append(templst)
        label.append(int(val.split(',')[4]))
    data=np.array(data)
    w=np.array(w)
    for i in range(n):
        val=np.dot(data,w)
        count,pending = checkconstraints(val,label)
        xlist.append(i)
        ylist.append(pending)
        if count > pa_count:
            pa_count = count
            pa_w = w
            pa_i = i
        if count==2000:
            break
        else:
            for x in range(len(data)):
                indvalarray=np.array(data[x])
                if((np.dot(indvalarray,w)<0.0) and label[x]==1):
                    tmpval = (eta * indvalarray)
                    w += tmpval
                elif((np.dot(indvalarray,w)>0.0) and label[x]==-1):
                    tmpval = (eta * indvalarray)
                    w -= tmpval
                else:
                    totalviolated += 1
    print('Weights where we had least constraints violated: ')
    print(pa_w[0],',',pa_w[1],',',pa_w[2],',',pa_w[3])
    print('Least amount of constraints violated: ')
    print(2000-pa_count)
    print('Iteration where least number of constraints were violated: ')
    print(pa_i)
    plt.plot(xlist,ylist)
    plt.xlabel('Number of Iterations')
    plt.ylabel('Violated constraints')
    plt.show()

if __name__ == '__main__':
    main()