import numpy

def adaptiveSamplingSparse(matA, s):
    n = matA.shape[1]
    s1 = int(numpy.floor(s/4))
    s2 = s1
    s3 = s - s1 - s2
    n0 = min(n, 10 * s) # the oversampling parameter can be tuned
    # ================== Over-Sampling ================== #
    arrIdx0 = numpy.random.choice(n, n0, replace=False)
    matA0 = matA[:, arrIdx0]
    # ================ Uniform Sampling ================ #
    arrIdx1 = list(range(s1))
    matC1 = matA0[:, arrIdx1].todense()
    matQ = numpy.linalg.qr(matC1, mode='reduced')[0]
    del matC1
    # ================ Adaptive Sampling 1 ================ #
    # Compute Residual
    matRes = numpy.dot(matQ, matQ.T * matA0)
    matRes = matRes - matA0
    matRes = numpy.array(matRes)
    # Sampling
    matRes = numpy.square(matRes)
    vecProb = numpy.sum(matRes, axis=0)
    del matRes
    vecProb = vecProb / sum(vecProb)
    arrIdx2 = numpy.random.choice(n0, s2, replace=False, p=vecProb)
    # Combine
    arrIdx = numpy.concatenate((arrIdx1, arrIdx2))
    matC2 = matA0[:, arrIdx].todense()
    matQ = numpy.linalg.qr(matC2, mode='reduced')[0]
    del matC2
    # ================ Adaptive Sampling 2 ================ #
    # Compute Residual
    matRes = numpy.dot(matQ, matQ.T * matA0)
    del matQ
    matRes = matRes - matA0
    del matA0
    matRes = numpy.array(matRes)
    # Sampling
    matRes = numpy.square(matRes)
    vecProb = numpy.sum(matRes, axis=0)
    del matRes
    vecProb = vecProb / sum(vecProb)
    arrIdx3 = numpy.random.choice(n0, s3, replace=False, p=vecProb)
    # Combine
    arrIdx = numpy.concatenate((arrIdx, arrIdx3))
    
    arrIndex = numpy.unique(arrIdx0[arrIdx])
    return arrIndex