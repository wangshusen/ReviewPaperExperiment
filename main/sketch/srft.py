import numpy
import scipy.sparse


def realfft(matA):
    n = matA.shape[0]
    matFFT = numpy.fft.fft(matA, n=None, axis=0) / numpy.sqrt(n)
    if n % 2 == 1:
        tmp = int((n+1) / 2)
        idxReal = list(range(1, tmp))
        idxImag = list(range(tmp, n))
    else:
        tmp = int(n/2)
        idxReal = list(range(1, tmp))
        idxImag = list(range(tmp+1, n))
    matRealFFT = matFFT.real
    matRealFFT[idxReal, :] = matFFT[idxReal, :].real * numpy.sqrt(2)
    matRealFFT[idxImag, :] = matFFT[idxImag, :].imag * numpy.sqrt(2)
    return matRealFFT
    
def srft(matA, s):
    '''
    The Subsampled Randomized Fourier Transform
    Input
        matA: n-by-d
        s: sketch size
    Output
        s-by-d matrix
    '''
    n = matA.shape[0]
    randSigns = numpy.random.choice(2, n) * 2 - 1
    randIndices = numpy.random.choice(n, s, replace=False)
    matA = matA * randSigns.reshape(n, 1)
    matA = realfft(matA)
    #matU, vecS, matV = numpy.linalg.svd(matA, full_matrices=False)
    #lev = numpy.sum(matU ** 2, axis=1)
    #print(max(lev))
    return matA[randIndices, :] * numpy.sqrt(n/s)

def srft2(matA, matB, s):
    '''
    The Subsampled Randomized Fourier Transform
    Input
        matA: n-by-d
        matB: n-by-k
        s: sketch size
    Output
        s-by-d matrix
        s-by-k matrix
    '''
    n = matA.shape[0]
    randSigns = numpy.random.choice(2, n) * 2 - 1
    randIndices = numpy.random.choice(n, s, replace=False)
    matA = matA * randSigns.reshape(n, 1)
    matB = matB * randSigns.reshape(n, 1)
    matA = realfft(matA)
    matB = realfft(matB)
    #matU, vecS, matV = numpy.linalg.svd(matA, full_matrices=False)
    #lev = numpy.sum(matU ** 2, axis=1)
    #print(max(lev))
    return matA[randIndices, :] * numpy.sqrt(n/s), matB[randIndices, :] * numpy.sqrt(n/s)



def srftSparse(matA, s):
    '''
    The Subsampled Randomized Fourier Transform for Sparse Data
    Perform fft on matA block-by-block to avoid densify matA
    Input
        matA: n-by-d
        s: sketch size
    Output
        s-by-d matrix
    '''
    n, d = matA.shape
    randSigns = numpy.random.choice(2, n) * 2 - 1
    randIndices = numpy.random.choice(n, s, replace=False)
    matRandSigns = scipy.sparse.spdiags(randSigns, 0, len(randSigns), len(randSigns))
    matA = matRandSigns * matA
    matAsketch = numpy.zeros((s, d))
    
    b = 53 # number of blocks, can be tuned
    blkSize = int(numpy.ceil(d / b)) # block size
    pos = d - (b-1) * blkSize
    idx = list(range(0, pos))
    matB = matA[:, idx].todense()
    matB = realfft(matB)
    matAsketch[:, idx] = matB[randIndices, :]
    
    for i in range(b-1):
        idx = list(range(pos, pos + blkSize))
        pos = pos + blkSize
        matB = matA[:, idx]
        matB = realfft(matB)
        matAsketch[:, idx] = matB[randIndices, :]
        
    del matB
    
    return matAsketch * numpy.sqrt(n/s)






