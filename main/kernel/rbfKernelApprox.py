import numpy
import kernelFun 
import adaptiveSampling
import sys
sys.path.append('../../main/sketch/')
import kmeansSelect


def nystrom(matX, sigma, s, sketching):
    k = int(numpy.ceil(s * 0.8)) # can be tuned
    n = matX.shape[0]
    if sketching == 'Uniform':
        arrIndex = numpy.random.choice(n, s, replace=False)
    elif sketching == 'Kmeans':
        arrIndex = kmeansSelect.mixedUniformKmeans(matX, s)
    elif sketching == 'Adaptive':
        arrIndex = adaptiveSampling.adaptiveSamplingRBF(matX, sigma, s)
    matC = kernelFun.rbf(matX, matX[arrIndex, :], sigma)
    matW = matC[arrIndex, :]
    matUW, vecSW, matVW = numpy.linalg.svd(matW, full_matrices=False)
    matUW = matUW[:, 0:k] / numpy.sqrt(vecSW[0:k])
    matC = numpy.dot(matC, matUW)
    matUL, vecSL, matVL = numpy.linalg.svd(matC, full_matrices=False)
    return matUL, vecSL



def fastSPSD(matX, sigma, s, sketching):
    p = s * 2 # can be tuned
    k = int(numpy.ceil(s * 0.8)) # can be tuned
    n = matX.shape[0]
    if sketching == 'Uniform':
        arrIndex = numpy.random.choice(n, s, replace=False)
    elif sketching == 'Kmeans':
        arrIndex = kmeansSelect.mixedUniformKmeans(matX, s)
    elif sketching == 'Adaptive':
        arrIndex = adaptiveSampling.adaptiveSamplingRBF(matX, sigma, s)
    matC = kernelFun.rbf(matX, matX[arrIndex, :], sigma)
    matC = numpy.linalg.qr(matC, mode='reduced')[0]
    arrIndex2 = numpy.random.choice(n, p, replace=False)
    arrIndex2 = numpy.concatenate((arrIndex, arrIndex2))
    arrIndex2 = numpy.unique(arrIndex2)
    matSC = matC[arrIndex2, :]
    matSCpinv = numpy.linalg.pinv(matSC)
    matSKS = kernelFun.rbf(matX[arrIndex2, :], matX[arrIndex2, :], sigma)
    matW = numpy.dot(numpy.dot(matSCpinv, matSKS), matSCpinv.T)
    matUW, vecSW, matVW = numpy.linalg.svd(matW, full_matrices=False)
    matUL = numpy.dot(matC, matUW[:, 0:k])
    return matUL, numpy.sqrt(vecSW[0:k])


def randFeature(matX, sigma, s):
    k = int(numpy.ceil(s * 0.8)) # can be tuned
    d = matX.shape[1]
    matW = numpy.random.standard_normal((d, s)) / sigma
    vecV = numpy.random.rand(1, s) * 2 * numpy.pi
    matL = numpy.dot(matX, matW) + vecV
    del matW
    matL = numpy.cos(matL) * numpy.sqrt(2/s)
    matUL, vecSL, matVL = numpy.linalg.svd(matL, full_matrices=False)
    vecSL = vecSL[0:k]
    return matUL[:, 0:k], vecSL





