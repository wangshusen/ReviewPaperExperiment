import numpy

def generateCovMatrix(d):
    covMatrix = numpy.zeros((d, d));
    for i in range(d):
        for j in range(d):
            covMatrix[i, j] = 0.5 ** (abs(i-j))
    return covMatrix


def mvtrnd(mu, Cov, v, n):
    '''
    Output:
    Produce M samples of d-dimensional multivariate t distribution
    Input:
    mu = mean (d dimensional numpy array or scalar)
    Cov = covariance matrix (d-by-d matrix)
    v = degrees of freedom
    n = # of samples to produce
    '''
    d = len(Cov)
    g = numpy.tile(numpy.random.gamma(v/2., 2./v, n), (d,1)).T
    Z = numpy.random.multivariate_normal(numpy.zeros(d), Cov, n)
    return mu + Z / numpy.sqrt(g)


def generateX(n, d, datatype):
    # ================= Generate the Bases ================= #
    v = 2 # degree of freedom of the t distribution
    mu = numpy.ones(d)
    covMatrix = generateCovMatrix(d);
    if datatype == 'NG' or datatype == 'NB':
        matX = mvtrnd(mu, covMatrix, 2, n)
    elif datatype == 'UG' or datatype == 'UB':
        matX = numpy.random.multivariate_normal(mu, covMatrix, n)
    matU = numpy.linalg.qr(matX, mode='reduced')[0]
    
    # ============ Generate the Singular Values ============ #
    if datatype == 'NG' or datatype == 'UG':
        vecS = numpy.linspace (1 , 1e-1, d)
    elif datatype == 'NB' or datatype == 'UB':
        vecS = numpy.logspace(0, -6, d)
        
    # ================ Generate the X matrix ================ #
    matV = numpy.random.randn(d, d)
    matV = numpy.linalg.qr(matV, mode='reduced')[0]
    matX = matU * vecS.reshape(1, d)
    matX = numpy.dot(matX, matV)
    
    return matX


def generateXW(n, d, datatype):
    d1 = int(numpy.ceil(d / 5))
    vecW1 = numpy.ones((d1, 1))
    vecW2 = numpy.ones((d-2*d1, 1))
    vecW = numpy.concatenate((vecW1, 0.1 * vecW2, vecW1))

    matX = generateX(n, d, datatype)
    return matX, vecW

def generateDataLSR(n, d, datatype, sigma):
    d1 = int(numpy.ceil(d / 5))
    vecW1 = numpy.ones((d1, 1))
    vecW2 = numpy.ones((d-2*d1, 1))
    vecW = numpy.concatenate((vecW1, 0.1 * vecW2, vecW1))

    matX = generateX(n, d, datatype)
    vecF = numpy.dot(matX, vecW)

    vecNoise = numpy.random.randn(n, 1) * sigma
    vecY= vecF + vecNoise
    return matX, vecY, vecW
