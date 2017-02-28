import numpy as np
import operator
from sklearn.decomposition import PCA as sklearnPCA

def main():
    points = []
    tmplist1 = []
    tmplist2 = []
    tmplist3 = []
    eigenvalue = []
    eigenvector = []
    eigendic = {}
    inputfile = open("pca-data.txt", "r")
    filecontent = inputfile.read().splitlines()
    for val in filecontent:
        datadict = []
        indval = val.split('\t')
        datadict.append(float(indval[0]))
        datadict.append(float(indval[1]))
        datadict.append(float(indval[2]))
        points.append(datadict)
    x = np.array(points).T
    covar = np.cov(x)
    val, vec = np.linalg.eig(covar)
    for x in vec:
        tmplist1.append(x[0])
        tmplist2.append(x[1])
        tmplist3.append(x[2])
    eigendic[val[0]] = tmplist1
    eigendic[val[1]] = tmplist2
    eigendic[val[2]] = tmplist3
    sortedeigendic = sorted(eigendic.items(), key=operator.itemgetter(0), reverse = True)
    for x in range(len(sortedeigendic)-2):
        sortedeigendic.pop()
    for x in sortedeigendic:
        eigenvalue.append(x[0])
        eigenvector.append(x[1])
    eigenvector = np.array(eigenvector).T
    print('Value: \n',eigenvalue)
    print('Vector: \n',eigenvector)
    newpoints = np.array(points).T
    newdata = eigenvector.T.dot(newpoints).T
    print('Data: \n',newdata)
    pcaval = sklearnPCA(n_components=2)
    transf = pcaval.fit_transform(points,y=None)
    print('Comparision: sklearn: \n',transf)

if __name__ == '__main__':
    main()