import numpy
import kernelFun 

def adaptiveSamplingRBF(matX, sigma, s):
    n = matX.shape[0]
    s1 = int(numpy.floor(s/3))
    s2 = s1
    s3 = s - s1 - s2
    n0 = min(n, 10 * s) # the oversampling parameter can be tuned
    # ================== Over-Sampling ================== #
    arrIdx0 = numpy.random.choice(n, n0, replace=False)
    matA = kernelFun.rbf(matX, matX[arrIdx0, :], sigma)
    # ================ Uniform Sampling ================ #
    arrIdx1 = list(range(s1))
    matC1 = matA[:, arrIdx1]
    matQ = numpy.linalg.qr(matC1, mode='reduced')[0]
    del matC1
    # ================ Adaptive Sampling 1 ================ #
    # Compute Residual
    matRes = numpy.dot(matQ, numpy.dot(matQ.T, matA))
    matRes = matA - matRes
    # Sampling
    matRes = numpy.square(matRes)
    vecProb = sum(matRes)
    del matRes
    vecProb = vecProb / sum(vecProb)
    arrIdx2 = numpy.random.choice(n0, s2, replace=False, p=vecProb)
    # Combine
    arrIdx = numpy.concatenate((arrIdx1, arrIdx2))
    matC2 = matA[:, arrIdx]
    matQ = numpy.linalg.qr(matC2, mode='reduced')[0]
    del matC2
    # ================ Adaptive Sampling 2 ================ #
    # Compute Residual
    matRes = numpy.dot(matQ, numpy.dot(matQ.T, matA))
    del matQ
    matRes = matA - matRes
    # Sampling
    matRes = numpy.square(matRes)
    vecProb = sum(matRes)
    del matRes
    vecProb = vecProb / sum(vecProb)
    arrIdx3 = numpy.random.choice(n0, s3, replace=False, p=vecProb)
    # Combine
    arrIdx = numpy.concatenate((arrIdx, arrIdx3))
    
    arrIndex = numpy.unique(arrIdx0[arrIdx])
    return arrIndex
