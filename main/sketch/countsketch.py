import numpy
import scipy.sparse


def countsketch(matX, matY, s):
    n, d = matX.shape
    m = matY.shape[1]
    hashedIndices = numpy.random.choice(s, n, replace=True)
    randSigns = numpy.random.choice(2, n, replace=True) * 2 - 1 
    matXsketch = numpy.zeros((s, d))
    matYsketch = numpy.zeros((s, m))
    for j in range(n):
        vecX = matX[j, :]
        vecY = matY[j, :]
        h = hashedIndices[j]
        g = randSigns[j]
        matXsketch[h, :] += g * vecX
        matYsketch[h, :] += g * vecY
    return matXsketch, matYsketch


def countsketch1(matX,  s):
    n, d = matX.shape
    hashedIndices = numpy.random.choice(s, n, replace=True)
    randSigns = numpy.random.choice(2, n, replace=True) * 2 - 1 
    matXsketch = numpy.zeros((s, d))
    for j in range(n):
        vecX = matX[j, :]
        h = hashedIndices[j]
        g = randSigns[j]
        matXsketch[h, :] += g * vecX
    return matXsketch

def countsketchSparse1(matX, s):
    n, d = matX.shape
    hashedIndices = numpy.random.choice(s, n, replace=True)
    randSigns = numpy.random.choice(2, n, replace=True) * 2 - 1
    matRandSigns = scipy.sparse.spdiags(randSigns, 0, len(randSigns), len(randSigns))
    matX = matRandSigns * matX
    matXsketch = numpy.zeros((s, d))
    for j in range(s):
        idx = (hashedIndices == j)
        matB = matX[idx, :]
        matXsketch[j, :] = matB.sum(axis=0)
    return matXsketch
        
        
        
def countsketchSparse2(matX, matY, s):
    '''
    matX: n-by-d sparse matrix
    matY: n-by-m dense matrix
    '''
    n, d = matX.shape
    m = matY.shape[1]
    hashedIndices = numpy.random.choice(s, n, replace=True)
    randSigns = numpy.random.choice(2, n, replace=True) * 2 - 1 
    matRandSigns = scipy.sparse.spdiags(randSigns, 0, len(randSigns), len(randSigns))
    matX = matRandSigns * matX
    matY = matY * randSigns.reshape(n, 1)
    matXsketch = numpy.zeros((s, d))
    matYsketch = numpy.zeros((s, m))
    for j in range(s):
        idx = (hashedIndices == j)
        matB = matX[idx, :]
        matXsketch[j, :] = matB.sum(axis=0)
        matC = matY[idx, :]
        matYsketch[j, :] = matC.sum(axis=0)
    return matXsketch, matYsketch
        
        
    
