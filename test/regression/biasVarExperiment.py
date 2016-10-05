import numpy
import sys
sys.path.append('../../main/sketch')
import countsketch as csketch
import srft as srft



def biasVariance(matX, vecW, xi, vecGamma, sketchSizes, matU, vecS, matV, vecLev):
    Repeat = 10
    
    n, d = matX.shape

    vecF = numpy.dot(matX, vecW)
    vecUf = numpy.dot(matU.T, vecF)
    vecVUf = numpy.dot(matV.T, vecUf)
    
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
    
    # ==========  solve RR with SRFT ========== #
    print('Testing SRFT')
    biasSRFT = numpy.zeros((lenGamma, lenS))
    varSRFT = numpy.zeros((lenGamma, lenS))
    for j in range(lenS):
        s = sketchSizes[j]
        biasTmp = numpy.zeros((lenGamma, Repeat))
        varTmp = numpy.zeros((lenGamma, Repeat))
        for rep in range(Repeat):
            matXsketch = srft.srft(matX, s) #### srft
            matUsketch, vecSsketch, matVsketch = numpy.linalg.svd(matXsketch, full_matrices=False)
            vecUFtilde = numpy.dot(matVsketch, vecVUf)
            for i in range(lenGamma):
                gamma = vecGamma[i]
                vecSig = vecSsketch * vecSsketch + n * gamma
                # bias
                err = numpy.linalg.norm(vecUFtilde / vecSig.reshape(d, 1))
                biasTmp[i, rep] = err * err * n * gamma * gamma
                # variance
                vecSig2 = vecSsketch / vecSig
                matTmp = numpy.dot(matV, matVsketch.T) * vecSig2.reshape(1, d)
                matTmp = matTmp * vecS.reshape(d, 1)
                matTmp = numpy.dot(matTmp, matUsketch.T)
                err = numpy.linalg.norm(matTmp)
                varTmp[i, rep] = err * err * (n / s) * xi * xi / n #### srft
                del matTmp
            del matXsketch
            del matUsketch
            del vecSsketch
            del matVsketch
            del vecUFtilde
        biasSRFT[:, j] = numpy.mean(biasTmp, axis=1)
        varSRFT[:, j] = numpy.mean(varTmp, axis=1)
    
    print(biasSRFT + varSRFT)
    resultDict['biasSRFT'] = biasSRFT
    resultDict['varSRFT'] = varSRFT
    
    # ==========  solve RR with uniform sampling ========== #
    print('Testing Uniform Sampling')
    biasUniform = numpy.zeros((lenGamma, lenS))
    varUniform = numpy.zeros((lenGamma, lenS))
    for j in range(lenS):
        s = sketchSizes[j]
        biasTmp = numpy.zeros((lenGamma, Repeat))
        varTmp = numpy.zeros((lenGamma, Repeat))
        for rep in range(Repeat):
            idx = numpy.random.choice(n, s, replace=False) #### Uniform Sampling
            matXsketch = matX[idx, :] * numpy.sqrt(n/s) #### Uniform Sampling
            matUsketch, vecSsketch, matVsketch = numpy.linalg.svd(matXsketch, full_matrices=False)
            vecUFtilde = numpy.dot(matVsketch, vecVUf)
            for i in range(lenGamma):
                gamma = vecGamma[i]
                vecSig = vecSsketch * vecSsketch + n * gamma
                # bias
                err = numpy.linalg.norm(vecUFtilde / vecSig.reshape(d, 1))
                biasTmp[i, rep] = err * err * n * gamma * gamma
                # variance
                vecSig2 = vecSsketch / vecSig
                matTmp = numpy.dot(matV, matVsketch.T) * vecSig2.reshape(1, d)
                matTmp = matTmp * vecS.reshape(d, 1)
                matTmp = numpy.dot(matTmp, matUsketch.T)
                err = numpy.linalg.norm(matTmp) #### Uniform Sampling
                varTmp[i, rep] = err * err * (n / s) * xi * xi / n #### Uniform Sampling
                del matTmp
            del matXsketch
            del matUsketch
            del vecSsketch
            del matVsketch
            del vecUFtilde
        biasUniform[:, j] = numpy.mean(biasTmp, axis=1)
        varUniform[:, j] = numpy.mean(varTmp, axis=1)
    
    print(biasUniform + varUniform)
    resultDict['biasUniform'] = biasUniform
    resultDict['varUniform'] = varUniform
    
    # ==========  solve RR with leverage score sampling ========== #
    print('Testing Leverage Score Sampling')
    biasLev = numpy.zeros((lenGamma, lenS))
    varLev = numpy.zeros((lenGamma, lenS))
    prob = vecLev / numpy.sum(vecLev)
    scaling = 1 / numpy.sqrt(s * prob)
    scaling = scaling.reshape(n, 1)
    for j in range(lenS):
        s = sketchSizes[j]
        biasTmp = numpy.zeros((lenGamma, Repeat))
        varTmp = numpy.zeros((lenGamma, Repeat))
        for rep in range(Repeat):
            idx = numpy.random.choice(n, s, replace=False, p=prob) #### leverage score sampling
            matXsketch = matX[idx, :] * scaling[idx] #### leverage score sampling
            matUsketch, vecSsketch, matVsketch = numpy.linalg.svd(matXsketch, full_matrices=False)
            vecUFtilde = numpy.dot(matVsketch, vecVUf)
            for i in range(lenGamma):
                gamma = vecGamma[i]
                vecSig = vecSsketch * vecSsketch + n * gamma
                # bias
                err = numpy.linalg.norm(vecUFtilde / vecSig.reshape(d, 1))
                biasTmp[i, rep] = err * err * n * gamma * gamma
                # variance
                vecSig2 = vecSsketch / vecSig
                matTmp = numpy.dot(matV, matVsketch.T) * vecSig2.reshape(1, d)
                matTmp = matTmp * vecS.reshape(d, 1)
                matTmp = numpy.dot(matTmp, matUsketch.T)
                err = numpy.linalg.norm(matTmp * scaling[idx].reshape(1, s)) #### leverage score sampling
                varTmp[i, rep] = err * err * xi * xi / n #### leverage score sampling
                del matTmp
            del matXsketch
            del matUsketch
            del vecSsketch
            del matVsketch
            del vecUFtilde
        biasLev[:, j] = numpy.mean(biasTmp, axis=1)
        varLev[:, j] = numpy.mean(varTmp, axis=1)
    
    print(biasLev + varLev)
    resultDict['biasLev'] = biasLev
    resultDict['varLev'] = varLev
    
    # ==========  solve RR with uniform + leverage score sampling ========== #
    print('Testing Mixed Sampling')
    biasMix = numpy.zeros((lenGamma, lenS))
    varMix = numpy.zeros((lenGamma, lenS))
    prob = vecLev / numpy.sum(vecLev)
    prob = (prob + 1/n) / 2
    scaling = 1 / numpy.sqrt(s * prob)
    scaling = scaling.reshape(n, 1)
    for j in range(lenS):
        s = sketchSizes[j]
        biasTmp = numpy.zeros((lenGamma, Repeat))
        varTmp = numpy.zeros((lenGamma, Repeat))
        for rep in range(Repeat):
            idx = numpy.random.choice(n, s, replace=False, p=prob) #### mixed sampling
            matXsketch = matX[idx, :] * scaling[idx] #### mixed sampling
            matUsketch, vecSsketch, matVsketch = numpy.linalg.svd(matXsketch, full_matrices=False)
            vecUFtilde = numpy.dot(matVsketch, vecVUf)
            for i in range(lenGamma):
                gamma = vecGamma[i]
                vecSig = vecSsketch * vecSsketch + n * gamma
                # bias
                err = numpy.linalg.norm(vecUFtilde / vecSig.reshape(d, 1))
                biasTmp[i, rep] = err * err * n * gamma * gamma
                # variance
                vecSig2 = vecSsketch / vecSig
                matTmp = numpy.dot(matV, matVsketch.T) * vecSig2.reshape(1, d)
                matTmp = matTmp * vecS.reshape(d, 1)
                matTmp = numpy.dot(matTmp, matUsketch.T)
                err = numpy.linalg.norm(matTmp * scaling[idx].reshape(1, s)) #### mixed sampling
                varTmp[i, rep] = err * err * xi * xi / n #### mixed sampling
                del matTmp
            del matXsketch
            del matUsketch
            del vecSsketch
            del matVsketch
            del vecUFtilde
        biasMix[:, j] = numpy.mean(biasTmp, axis=1)
        varMix[:, j] = numpy.mean(varTmp, axis=1)
    
    print(biasMix + varMix)
    resultDict['biasMix'] = biasMix
    resultDict['varMix'] = varMix
    
    # ==========  solve RR with Gaussian Projection ========== #
    print('Testing Gaussian Projection')
    biasGauss = numpy.zeros((lenGamma, lenS))
    varGauss = numpy.zeros((lenGamma, lenS))
    for j in range(lenS):
        s = sketchSizes[j]
        biasTmp = numpy.zeros((lenGamma, Repeat))
        varTmp = numpy.zeros((lenGamma, Repeat))
        for rep in range(Repeat):
            matSketch = numpy.random.randn(s, n) / numpy.sqrt(s) #### Gaussian Projection
            matXsketch = numpy.dot(matSketch, matX) #### Gaussian Projection
            matUsketch, vecSsketch, matVsketch = numpy.linalg.svd(matXsketch, full_matrices=False)
            vecUFtilde = numpy.dot(matVsketch, vecVUf)
            for i in range(lenGamma):
                gamma = vecGamma[i]
                vecSig = vecSsketch * vecSsketch + n * gamma
                # bias
                err = numpy.linalg.norm(vecUFtilde / vecSig.reshape(d, 1))
                biasTmp[i, rep] = err * err * n * gamma * gamma
                # variance
                vecSig2 = vecSsketch / vecSig
                matTmp = numpy.dot(matV, matVsketch.T) * vecSig2.reshape(1, d)
                matTmp = matTmp * vecS.reshape(d, 1)
                matTmp = numpy.dot(matTmp, matUsketch.T)
                #err = numpy.linalg.norm(numpy.dot(matTmp, matSketch)) #### Gaussian Projection
                #varTmp[i, rep] = err * err * xi * xi / n #### Gaussian Projection
                err = numpy.linalg.norm(matTmp) 
                varTmp[i, rep] = err * err * (n / s) * xi * xi / n 
                del matTmp
            del matXsketch
            del matUsketch
            del vecSsketch
            del matVsketch
            del vecUFtilde
        biasGauss[:, j] = numpy.mean(biasTmp, axis=1)
        varGauss[:, j] = numpy.mean(varTmp, axis=1)
        
    print(biasGauss + varGauss)
    resultDict['biasGauss'] = biasGauss
    resultDict['varGauss'] = varGauss
    
    # ==========  solve RR with count sketch ========== #
    print('Testing Count Sketch')
    biasCount = numpy.zeros((lenGamma, lenS))
    varCount = numpy.zeros((lenGamma, lenS))
    for j in range(lenS):
        s = sketchSizes[j]
        biasTmp = numpy.zeros((lenGamma, Repeat))
        varTmp = numpy.zeros((lenGamma, Repeat))
        for rep in range(Repeat):
            matXsketch = csketch.countsketch1(matX, s) #### count sketch
            matUsketch, vecSsketch, matVsketch = numpy.linalg.svd(matXsketch, full_matrices=False)
            vecUFtilde = numpy.dot(matVsketch, vecVUf)
            for i in range(lenGamma):
                gamma = vecGamma[i]
                vecSig = vecSsketch * vecSsketch + n * gamma
                # bias
                err = numpy.linalg.norm(vecUFtilde / vecSig.reshape(d, 1))
                biasTmp[i, rep] = err * err * n * gamma * gamma
                # variance
                vecSig2 = vecSsketch / vecSig
                matTmp = numpy.dot(matV, matVsketch.T) * vecSig2.reshape(1, d)
                matTmp = matTmp * vecS.reshape(d, 1)
                matTmp = numpy.dot(matTmp, matUsketch.T)
                err = numpy.linalg.norm(matTmp)
                varTmp[i, rep] = err * err * (n / s) * xi * xi / n #### count sketch
                del matTmp
            del matXsketch
            del matUsketch
            del vecSsketch
            del matVsketch
            del vecUFtilde
        biasCount[:, j] = numpy.mean(biasTmp, axis=1)
        varCount[:, j] = numpy.mean(varTmp, axis=1)
    
    print(biasCount + varCount)
    resultDict['biasCount'] = biasCount
    resultDict['varCount'] = varCount 
    
    return resultDict

