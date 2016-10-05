import numpy
import kernelFun 

def adaptiveSamplingRBF(matX, sigma, s):
    n = matX.shape[0]
    s1 = int(numpy.ceil(s/3))
    n0 = min(n, 5 * s) # the oversampling parameter can be tuned
    # ================== Over-Sampling ================== #
    arrIdx0 = numpy.random.choice(n, n0, replace=False)
    matA = kernelFun.rbf(matX, matX[arrIdx0, :], sigma)
    # ================ Uniform Sampling ================ #
    arrIdx1 = list(range(s1))
    matC1 = matA[:, arrIdx1]# compute the residual
    matQ = numpy.linalg.qr(matC1, mode='reduced')[0]
    del matC1
    # ================ Compute Residual ================ #
    matRes = numpy.dot(matQ, numpy.dot(matQ.T, matA))
    matRes = matA - matRes
    # ==================== Sampling ==================== #
    matRes = numpy.square(matRes)
    vecProb = sum(matRes)
    del matRes
    vecProb = vecProb / sum(vecProb)
    arrIdx2 = numpy.random.choice(n0, s-s1, replace=False, p=vecProb)
    # ==================== Combine ==================== #
    arrIdx = numpy.concatenate((arrIdx1, arrIdx2))
    arrIndex = numpy.unique(arrIdx0[arrIdx])
    return arrIndex
