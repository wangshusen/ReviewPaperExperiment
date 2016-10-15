import numpy
import sys
sys.path.append('../../main/sketch')
import countsketch as csketch
import srft as srft


def sketchedBiasVar(matU, vecS, sketch, s, matSVW, vecGamma, xi, vecLev):
    Repeat = 10
    n, d = matU.shape
    
    lenGamma = len(vecGamma)
    biasTmp = numpy.zeros((lenGamma, Repeat))
    varTmp = numpy.zeros((lenGamma, Repeat))
    
    for rep in range(Repeat):
        if sketch == 'uniform':
            idx = numpy.random.choice(n, s, replace=False)
            matUsketch = matU[idx, :] * numpy.sqrt(n/s)
        elif sketch == 'lev':
            prob = vecLev / numpy.sum(vecLev)
            scaling = 1 / numpy.sqrt(s * prob)
            scaling = scaling.reshape(n, 1)
            idx = numpy.random.choice(n, s, replace=False, p=prob)
            matUsketch = matU[idx, :] * scaling[idx]
        elif sketch == 'levU':
            prob = vecLev / numpy.sum(vecLev)
            idx = numpy.random.choice(n, s, replace=False, p=prob)
            matUsketch = matU[idx, :] * numpy.sqrt(n/s)
        elif sketch == 'mix':
            prob = vecLev / numpy.sum(vecLev)
            prob = (prob + 1/n) / 2
            scaling = 1 / numpy.sqrt(s * prob)
            scaling = scaling.reshape(n, 1)
            idx = numpy.random.choice(n, s, replace=False, p=prob)
            matUsketch = matU[idx, :] * scaling[idx]
        elif sketch == 'gaussian':
            matSketch = numpy.random.randn(s, n) / numpy.sqrt(s)
            matUsketch = numpy.dot(matSketch, matU)
        elif sketch == 'count':
            matUsketch = csketch.countsketch1(matU, s)
        elif sketch == 'srft':
            matUsketch = srft.srft(matU, s)
            
        matUSSU = numpy.dot(matUsketch.T, matUsketch)
        
        for i in range(lenGamma):
            gamma = vecGamma[i]
            # bias
            matInv = numpy.diag((n * gamma) / (vecS * vecS))
            matInv = matUSSU + matInv
            matInv = numpy.linalg.pinv(matInv)
            matBias = numpy.dot(matInv, matSVW) - matSVW
            err = numpy.linalg.norm(matBias)
            biasTmp[i, rep] = err * err / n
            del matBias
            # variance
            err = numpy.linalg.norm(matInv)
            varTmp[i, rep] = err * err * xi * xi / n
            del matInv
            
        del matUSSU
        del matUsketch
    return biasTmp, varTmp


def biasVariance(matX, vecW, xi, vecGamma, sketchSizes, matU, vecS, matV, vecLev):
    
    n, d = matX.shape

    vecF = numpy.dot(matX, vecW)
    vecUf = numpy.dot(matU.T, vecF)
    
    lenGamma = len(vecGamma)
    lenS = len(sketchSizes)
    
    # =============== solve RR optimally =============== #
    print('Testing Optimal Ridge Regression')
    biasRR = numpy.zeros((lenGamma, 1))
    varRR = numpy.zeros((lenGamma, 1))
    for i in range(lenGamma):
        gamma = vecGamma[i]
        # bias
        vecSig = vecS * vecS + n * gamma
        err = numpy.linalg.norm(vecUf / vecSig.reshape(d, 1))
        biasRR[i] = err * err * n * gamma * gamma
        # variance
        vecSig2 = vecS / vecSig * vecS
        err = numpy.linalg.norm(vecSig2)
        varRR[i] = err * err  * xi * xi / n
        del vecSig
        del vecSig2
    print(biasRR + varRR)
    resultDict = {'vecGamma': vecGamma,
                 'sketchSizes': sketchSizes,
                 'biasRR': biasRR,
                 'varRR': varRR}
    
    
    
    # ========== Pre-computation ========== #
    matSVW = numpy.dot(matV, vecW)
    matSVW = matSVW * vecS.reshape(d, 1)
    
    
    # ==========  solve RR with Uniform Sampling ========== #
    print('Testing Uniform Sampling')
    biasMean = numpy.zeros((lenGamma, lenS))
    varMean = numpy.zeros((lenGamma, lenS))
    
    for j in range(lenS):
        s = sketchSizes[j]
        biasTmp, varTmp = sketchedBiasVar(matU, vecS, 'uniform', s, matSVW, vecGamma, xi, vecLev)
        biasMean[:, j] = numpy.mean(biasTmp, axis=1)
        varMean[:, j] = numpy.mean(varTmp, axis=1)
        
    resultDict['biasUniform'] = biasMean
    resultDict['varUniform'] = varMean
    
    
    # ==========  solve RR with Leverage Score Sampling ========== #
    print('Testing Leverage Score Sampling')
    biasMean = numpy.zeros((lenGamma, lenS))
    varMean = numpy.zeros((lenGamma, lenS))
    
    for j in range(lenS):
        s = sketchSizes[j]
        biasTmp, varTmp = sketchedBiasVar(matU, vecS, 'lev', s, matSVW, vecGamma, xi, vecLev)
        biasMean[:, j] = numpy.mean(biasTmp, axis=1)
        varMean[:, j] = numpy.mean(varTmp, axis=1)
        
    resultDict['biasLev'] = biasMean
    resultDict['varLev'] = varMean
    
    
    
    # ==========  solve RR with Leverage Score Sampling (No scaling) ========== #
    print('Testing Leverage Score Sampling')
    biasMean = numpy.zeros((lenGamma, lenS))
    varMean = numpy.zeros((lenGamma, lenS))
    
    for j in range(lenS):
        s = sketchSizes[j]
        biasTmp, varTmp = sketchedBiasVar(matU, vecS, 'levU', s, matSVW, vecGamma, xi, vecLev)
        biasMean[:, j] = numpy.mean(biasTmp, axis=1)
        varMean[:, j] = numpy.mean(varTmp, axis=1)
        
    resultDict['biasLevU'] = biasMean
    resultDict['varLevU'] = varMean
    
    
    # ==========  solve RR with Mixed Sampling ========== #
    print('Testing Mixed Sampling')
    biasMean = numpy.zeros((lenGamma, lenS))
    varMean = numpy.zeros((lenGamma, lenS))
    
    for j in range(lenS):
        s = sketchSizes[j]
        biasTmp, varTmp = sketchedBiasVar(matU, vecS, 'mix', s, matSVW, vecGamma, xi, vecLev)
        biasMean[:, j] = numpy.mean(biasTmp, axis=1)
        varMean[:, j] = numpy.mean(varTmp, axis=1)
        
    resultDict['biasMix'] = biasMean
    resultDict['varMix'] = varMean
    
    
    # ==========  solve RR with Gaussian ========== #
    print('Testing Gaussian Projection')
    biasMean = numpy.zeros((lenGamma, lenS))
    varMean = numpy.zeros((lenGamma, lenS))
    
    for j in range(lenS):
        s = sketchSizes[j]
        biasTmp, varTmp = sketchedBiasVar(matU, vecS, 'gaussian', s, matSVW, vecGamma, xi, vecLev)
        biasMean[:, j] = numpy.mean(biasTmp, axis=1)
        varMean[:, j] = numpy.mean(varTmp, axis=1)
        
    resultDict['biasGauss'] = biasMean
    resultDict['varGauss'] = varMean
    
    
    # ==========  solve RR with SRFT ========== #
    print('Testing SRFT')
    biasMean = numpy.zeros((lenGamma, lenS))
    varMean = numpy.zeros((lenGamma, lenS))
    
    for j in range(lenS):
        s = sketchSizes[j]
        biasTmp, varTmp = sketchedBiasVar(matU, vecS, 'srft', s, matSVW, vecGamma, xi, vecLev)
        biasMean[:, j] = numpy.mean(biasTmp, axis=1)
        varMean[:, j] = numpy.mean(varTmp, axis=1)
        
    resultDict['biasSRFT'] = biasMean
    resultDict['varSRFT'] = varMean
    
    
    # ==========  solve RR with count sketch ========== #
    print('Testing Count Sketch')
    biasMean = numpy.zeros((lenGamma, lenS))
    varMean = numpy.zeros((lenGamma, lenS))
    
    for j in range(lenS):
        s = sketchSizes[j]
        biasTmp, varTmp = sketchedBiasVar(matU, vecS, 'count', s, matSVW, vecGamma, xi, vecLev)
        biasMean[:, j] = numpy.mean(biasTmp, axis=1)
        varMean[:, j] = numpy.mean(varTmp, axis=1)
        
    resultDict['biasCount'] = biasMean
    resultDict['varCount'] = varMean
    
    
    return resultDict
