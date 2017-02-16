import numpy as np
def covariance(cluster):
    x = np.array(cluster).T
    co = np.cov(x)
    print(co)


def mean(cluster):
    return np.mean(cluster)

def amplitude(ric, cluster, data_set):
    num = len(data_set)
    ric_sum = 0
    for data in data_set:
        ric_sum += data[ric1]
    return ric_sum/num

def ric(clusters, data_set):
    from scipy.stats import multivariate_normal
    for point in data_set:
        summation = 0.0
        for cluster in clusters:
            var = multivariate_normal(mean=cluster[mean], cov=cluster[covariance])
            summation += cluster[amplitude]*var.pdf(point)
        for cluster in clusters:
            var = multivariate_normal(mean=cluster[mean], cov=cluster[covariance])
            (cluster[amplitude] * var.pdf(point))/(summation)

