# Authors: Karthik Ravindra Rao - raokarth@usc.edu & Praneet Kalluri - pkalluri@usc.edu

import random
import numpy as np
from scipy.stats import multivariate_normal

k = 3
maxiterations = 100

def stopcheck(clusters,oldclusters):
    # Write updated logic for convergence
    newmean = []
    oldmean = []
    for var in clusters:
        newmean.append(var['mean'])
    if oldclusters:
        for var in oldclusters:
            oldmean.append(var['mean'])
    return newmean == oldmean

def weightedmean(listpoints):
    mean1 = []
    mean2 = []
    mean3 = []
    xmean1 = 0.0
    xmean2 = 0.0
    xmean3 = 0.0
    ymean1 = 0.0
    ymean2 = 0.0
    ymean3 = 0.0
    ric1sum = 0.0
    ric2sum = 0.0
    ric3sum = 0.0
    for var in listpoints:
        xmean1 += var['x'] * var['ric1']
        ymean1 += var['y'] * var['ric1']
        xmean2 += var['x'] * var['ric2']
        ymean2 += var['y'] * var['ric2']
        xmean3 += var['x'] * var['ric3']
        ymean3 += var['y'] * var['ric3']
        ric1sum += var['ric1']
        ric2sum += var['ric2']
        ric3sum += var['ric3']
    xmean1 = xmean1 / ric1sum
    ymean1 = ymean1 / ric1sum
    xmean2 = xmean2 / ric2sum
    ymean2 = ymean2 / ric2sum
    xmean3 = xmean3 / ric3sum
    ymean3 = ymean3 / ric3sum
    mean1.append(xmean1)
    mean1.append(ymean1)
    mean2.append(xmean2)
    mean2.append(ymean2)
    mean3.append(xmean3)
    mean3.append(ymean3)
    return mean1,mean2,mean3

def weightedcovar(listpoints,riclist):
    #Check logic correctness
    finallist = []
    for var in listpoints:
        tmplist = []
        tmplist.append(var['x'])
        tmplist.append(var['y'])
        finallist.append(tmplist)
    x = np.array(finallist).T
    covar = np.cov(x,aweights=riclist)
    return (covar)

def newclusters(listpoints):
    riclist1 = []
    riclist2 = []
    riclist3 = []
    rlist1 = []
    rlist2 = []
    rlist3 = []
    amp1 = 0.0
    amp2 = 0.0
    amp3 = 0.0
    for var in listpoints:
        amp1 += var['ric1']
        amp2 += var['ric2']
        amp3 += var['ric3']
        rlist1.append(var['ric1'])
        rlist2.append(var['ric2'])
        rlist3.append(var['ric3'])
    amp1 = amp1 / 150
    amp2 = amp2 / 150
    amp3 = amp3 / 150
    mean1,mean2,mean3 = weightedmean(listpoints)
    for var in listpoints:
        riclist1.append(var['ric1'])
        riclist2.append(var['ric2'])
        riclist3.append(var['ric3'])
    covar1 = weightedcovar(listpoints,riclist1)
    covar2 = weightedcovar(listpoints, riclist2)
    covar3 = weightedcovar(listpoints, riclist3)
    updatedclusters = [{'covar':covar1, 'mean':mean1, 'amp':amp1},{'covar':covar2, 'mean':mean2, 'amp':amp2},{'covar':covar3, 'mean':mean3, 'amp':amp3}]
    return updatedclusters

def covariance(listpoints):
    finallist = []
    for var in listpoints:
        tmplist = []
        tmplist.append(var['x'])
        tmplist.append(var['y'])
        finallist.append(tmplist)
    x = np.array(finallist).T
    covar = np.cov(x)
    return (covar)

def meancal(listpoints):
    finallist = []
    for var in listpoints:
        tmplist = []
        tmplist.append(var['x'])
        tmplist.append(var['y'])
        finallist.append(tmplist)
    a = np.array(finallist)
    var = np.mean(a, axis=0)
    return (var)

def ric(clusters,listpoints):
    for var in listpoints:
        tmplist = []
        nrval = 0.0
        drsum = 0.0
        tmpval = 1
        tmplist.append(var['x'])
        tmplist.append(var['y'])
        for cluster in clusters:
            value = multivariate_normal(mean = cluster['mean'], cov = cluster['covar'])
            drsum += (cluster['amp'] * value.pdf(tmplist))
        for cluster in clusters:
            value = multivariate_normal(mean = cluster['mean'], cov = cluster['covar'])
            nrval = (cluster['amp'] * (multivariate_normal(mean = cluster['mean'], cov = cluster['covar'])).pdf(tmplist))
            ricval = nrval/drsum
            if tmpval == 1:
                var['ric1'] = ricval
            elif tmpval == 2:
                var['ric2'] = ricval
            else:
                var['ric3'] = ricval
            tmpval += 1
    return listpoints

def ricinitialization(listpoints):
    tmplist = [1,2,3]
    for x in listpoints:
        val = random.choice(tmplist)
        if val == 1:
            x['ric1'] = 1.0
            x['ric2'] = 0.0
            x['ric3'] = 0.0
        elif val == 2:
            x['ric1'] = 0.0
            x['ric2'] = 1.0
            x['ric3'] = 0.0
        else:
            x['ric1'] = 0.0
            x['ric2'] = 0.0
            x['ric3'] = 1.0
    return listpoints

def splitcontent(listpoints):
    riclist1 = []
    riclist2 = []
    riclist3 = []
    for var in listpoints:
        if var['ric1'] == 1:
            riclist1.append(var)
        elif var['ric2'] == 1:
            riclist2.append(var)
        else:
            riclist3.append(var)
    covric1 = covariance(riclist1)
    covric2 = covariance(riclist2)
    covric3 = covariance(riclist3)
    meanric1 = meancal(riclist1)
    meanric2 = meancal(riclist2)
    meanric3 = meancal(riclist3)
    ric1amp = len(riclist1)
    ric2amp = len(riclist2)
    ric3amp = len(riclist3)
    clusters = [{'covar':covric1, 'mean':meanric1, 'amp':ric1amp},{'covar':covric2, 'mean':meanric2, 'amp':ric2amp},{'covar':covric3, 'mean':meanric3, 'amp':ric3amp}]
    return clusters

def main():
    listpoints = []
    iterations = 0
    inputfile = open("clusters.txt", "r")
    filecontent = inputfile.read().splitlines()
    for val in filecontent:
        datadict = {}
        indval = val.split(',')
        datadict['x'] = float(indval[0])
        datadict['y'] = float(indval[1])
        datadict['ric1'] = -999999
        datadict['ric2'] = -999999
        datadict['ric3'] = -999999
        listpoints.append(datadict)
    listpoints = ricinitialization(listpoints)
    clusters = splitcontent(listpoints)
    listpoints = ric(clusters, listpoints)
    while 1:
        if iterations > maxiterations:
            break
        iterations += 1
        clusters = newclusters(listpoints)
        listpoints = ric(clusters,listpoints)
    for var in clusters:
        print('Mean: ',var['mean'],'Covariance: ',var['covar'],'Amplitude: ',var['amp'])

if __name__ == '__main__':
    main()