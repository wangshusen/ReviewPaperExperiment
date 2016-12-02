import numpy
import sys
sys.path.append('../../main/sketch/')
import countsketch as csketch
import srft


def solveW(vecS, matV, matUY, ngamma):
    '''
    Input
        X = U * diag(S) * V
        UY = U^T * Y
        ngamma = n * gamma
    '''
    vec1 = ngamma / vecS
    vec1 = vec1 + vecS
    mat1 = matUY / vec1.reshape(len(vec1), 1)
    matW = numpy.dot(matV.T, mat1)
    return matW

def solveWhat(vecS, matV, matXY, ngamma):
    vec1 = vecS * vecS + ngamma
    mat1 = numpy.dot(matV, matXY)
    mat1 = mat1 / vec1.reshape(len(vec1), 1)
    matW = numpy.dot(matV.T, mat1)
    return matW


def objFunValOpt(matX, matY, vecS, matV, matUY, vecGamma):
    n = matX.shape[0]
    lenGamma = len(vecGamma)
    vecObjOpt = numpy.zeros(lenGamma)
    for i in range(lenGamma):
        gamma = vecGamma[i]
        matW = solveW(vecS, matV, matUY, n * gamma)
        matRes = numpy.dot(matX, matW) - matY
        err = numpy.linalg.norm(matRes)
        del matRes
        reg = numpy.linalg.norm(matW)
        del matW
        vecObjOpt[i] = err * err / n + gamma * reg * reg
    return vecObjOpt
    

def modelAveraging(matX, matY, matXY, vecGamma, sketch, s, vecLev):
    repeat = 200
    n, d = matX.shape
    lenGamma = len(vecGamma)
    
    modelTilde = numpy.zeros((lenGamma, repeat, d))
    modelHat = numpy.zeros((lenGamma, repeat, d))
    
    if sketch == 'srft':
        randSigns = numpy.random.choice(2, n) * 2 - 1
        matXrft = matX * randSigns.reshape(n, 1)
        matYrft = matY * randSigns.reshape(n, 1)
        matXrft = srft.realfft(matXrft)
        matYrft = srft.realfft(matYrft)
    
    for j in range(repeat):
        # ================== sketching ================== #
        if sketch == 'uni':
            idx = numpy.random.choice(n, s, replace=False)
            matXsketch = matX[idx, :] * numpy.sqrt(n/s)
            matYsketch = matY[idx, :] * numpy.sqrt(n/s)
        elif sketch == 'lev':
            prob = vecLev / numpy.sum(vecLev)
            scaling = 1 / numpy.sqrt(s * prob)
            scaling = scaling.reshape(n, 1)
            idx = numpy.random.choice(n, s, replace=False, p=prob)
            matXsketch = matX[idx, :] * scaling[idx]
            matYsketch = matY[idx, :] * scaling[idx]
        elif sketch == 'shrink':
            prob = vecLev / numpy.sum(vecLev)
            prob = (prob + 1/n) / 2
            scaling = 1 / numpy.sqrt(s * prob)
            scaling = scaling.reshape(n, 1)
            idx = numpy.random.choice(n, s, replace=False, p=prob)
            matXsketch = matX[idx, :] * scaling[idx]
            matYsketch = matY[idx, :] * scaling[idx]
        elif sketch == 'gauss':
            matSketch = numpy.random.randn(s, n) / numpy.sqrt(s) 
            matXsketch = numpy.dot(matSketch, matX)
            matYsketch = numpy.dot(matSketch, matY)
        elif sketch == 'srft':
            idx = numpy.random.choice(n, s, replace=False)
            matXsketch = matXrft[idx, :] * numpy.sqrt(n/s)
            matYsketch = matYrft[idx, :] * numpy.sqrt(n/s)
        elif sketch == 'count':
            matXsketch, matYsketch = csketch.countsketch(matX, matY, s)

        matU, vecS, matV = numpy.linalg.svd(matXsketch, full_matrices=False)
        matUY = numpy.dot(matU.T, matYsketch)

        # ================ compute Wtilde ================ #
        for i in range(lenGamma):
            gamma = vecGamma[i]
            matW = solveW(vecS, matV, matUY, n * gamma)
            modelTilde[i, j, :] = matW.reshape(matW.shape[0])

        # ================ compute What ================ #
        for i in range(lenGamma):
            gamma = vecGamma[i]
            matW = solveWhat(vecS, matV, matXY, n * gamma)
            modelHat[i, j, :] = matW.reshape(matW.shape[0])
        
    return modelTilde, modelHat
    
    
def objFunValSketch(matX, matY, vecGamma, models, vecG):
    lenG = len(vecG)
    n, d = matX.shape
    repeat = models.shape[1]
    lenGamma = len(vecGamma)
    
    matObj = numpy.zeros((lenGamma, lenG))

    for i in range(lenGamma):
        gamma = vecGamma[i]
        for l in range(lenG):
            g = vecG[l]
            r = repeat - g + 1
            objTmp = numpy.zeros(r)
            for p in range(r):
                matW = numpy.mean(models[i, p:p+g, :], axis=0)
                matRes = numpy.dot(matX, matW.reshape(d, 1)) - matY
                err = numpy.linalg.norm(matRes)
                del matRes
                reg = numpy.linalg.norm(matW)
                del matW
                objTmp[p] = err * err / n + gamma * reg * reg
            matObj[i, l] = numpy.mean(objTmp)
    return matObj


def objExperiment(matX, matW, s, vecGamma, xi):
    n, d = matX.shape
    vecG = [1, 2, 3, 4, 6, 8, 10, 12, 15, 18, 22, 26, 30, 35, 40, 45, 50]
    
    matU, vecS, matV = numpy.linalg.svd(matX, full_matrices=False)
    vecLev = numpy.sum(matU ** 2, axis=1)
    
    matNoise = numpy.random.randn(n, 1)
    matY = numpy.dot(matX, matW) + matNoise * xi
    matXY = numpy.dot(matX.T, matY)
    matUY = numpy.dot(matU.T, matY)
    
    # optimal
    print('Doing optimal ridge regression...')
    objOpt = objFunValOpt(matX, matY, vecS, matV, matUY, vecGamma)
    # uniform
    print('Doing uniform sampling...')
    modelTilde, modelHat = modelAveraging(matX, matY, matXY, vecGamma, 'uni', s, vecLev)
    objTildeUni = objFunValSketch(matX, matY, vecGamma, modelTilde, vecG)
    objHatUni = objFunValSketch(matX, matY, vecGamma, modelHat, vecG)
    # leverage
    print('Doing leverage score sampling...')
    modelTilde, modelHat = modelAveraging(matX, matY, matXY, vecGamma, 'lev', s, vecLev)
    objTildeLev = objFunValSketch(matX, matY, vecGamma, modelTilde, vecG)
    objHatLev = objFunValSketch(matX, matY, vecGamma, modelHat, vecG)
    # shrinkage leverage
    print('Doing shrinkage leverage score sampling...')
    modelTilde, modelHat = modelAveraging(matX, matY, matXY, vecGamma, 'shrink', s, vecLev)
    objTildeShrink = objFunValSketch(matX, matY, vecGamma, modelTilde, vecG)
    objHatShrink = objFunValSketch(matX, matY, vecGamma, modelHat, vecG)
    # Gaussian projection
    print('Doing Gaussian projection...')
    modelTilde, modelHat = modelAveraging(matX, matY, matXY, vecGamma, 'gauss', s, vecLev)
    objTildeGauss = objFunValSketch(matX, matY, vecGamma, modelTilde, vecG)
    objHatGauss = objFunValSketch(matX, matY, vecGamma, modelHat, vecG)
    # SRFT
    print('Doing SRFT...')
    modelTilde, modelHat = modelAveraging(matX, matY, matXY, vecGamma, 'srft', s, vecLev)
    objTildeSrft = objFunValSketch(matX, matY, vecGamma, modelTilde, vecG)
    objHatSrft = objFunValSketch(matX, matY, vecGamma, modelHat, vecG)
    # Count Sketch
    print('Doing count sketch...')
    modelTilde, modelHat = modelAveraging(matX, matY, matXY, vecGamma, 'count', s, vecLev)
    objTildeCount = objFunValSketch(matX, matY, vecGamma, modelTilde, vecG)
    objHatCount = objFunValSketch(matX, matY, vecGamma, modelHat, vecG)

    resultDict = {'gamma': vecGamma,
                  'g': vecG,
                  'Opt': objOpt,
                 'TildeUni': objTildeUni,
                 'HatUni': objHatUni,
                 'TildeLev': objTildeLev,
                 'HatLev': objHatLev,
                 'TildeShrink': objTildeShrink,
                 'HatShrink': objHatShrink,
                 'TildeGauss': objTildeGauss,
                 'HatGauss': objHatGauss,
                 'TildeSrft': objTildeSrft,
                 'HatSrft': objHatSrft,
                 'TildeCount': objTildeCount,
                 'HatCount': objHatCount }
    return resultDict
