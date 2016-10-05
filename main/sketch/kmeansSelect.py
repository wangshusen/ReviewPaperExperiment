import numpy
import scipy.cluster.vq as vq

def mixedUniformKmeans(matX, s):
    numCluster = 50 # can be tuned
    if s <= numCluster:
        arrIndex = kmeansSelection(matX, s)
    else:
        arrIndex1 = kmeansSelection(matX, numCluster)
        s2 = s - arrIndex1.shape[0]
        n = matX.shape[0]
        arrIndex2 = numpy.random.choice(n, s2, replace=False)
        arrIndex = numpy.concatenate((arrIndex1, arrIndex2))
        arrIndex = numpy.unique(arrIndex)
    return arrIndex

def kmeansSelection(matX, s):
    numIter = 10 # can be tuned
    matX = vq.whiten(matX)
    matXSub, dist = vq.kmeans(matX, s, numIter)
    matDist = l2dist(matX, matXSub)
    arrIndex = numpy.argmin(matDist, axis=0)
    arrIndex = numpy.unique(arrIndex)
    return arrIndex
    
def l2dist(matrixX1, matrixX2):
    n1 = matrixX1.shape[0]
    n2 = matrixX2.shape[0]
    K = numpy.dot(matrixX1, matrixX2.T)
    rowNormX1 = numpy.sum(numpy.square(matrixX1), 1) / 2
    rowNormX2 = numpy.sum(numpy.square(matrixX2), 1) / 2
    K = K - rowNormX1.reshape(n1, 1)
    K = K - rowNormX2.reshape(1, n2)
    return -K


