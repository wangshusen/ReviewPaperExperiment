import numpy
import sys
sys.path.append('../../main/sketch')
import countsketch as csketch
import srft


def rrSolver(matU, vecS, matV, vecUy, ngamma):
    d = vecUy.shape[0]
    vecSig = vecS + ngamma / vecS
    vecRR = vecUy / vecSig.reshape(d, 1)
    vecRR = numpy.dot(matV.T, vecRR)
    return vecRR

def rrSolver2(matVsketch, vecSsketch, vecXy, ngamma):
    d = vecXy.shape[0]
    vecSig = vecSsketch ** 2 + ngamma
    vecModel = numpy.dot(matVsketch, vecXy)
    vecModel = vecModel / vecSig.reshape(d, 1)
    vecModel = numpy.dot(matVsketch.T, vecModel)
    return vecModel


def empiricalRiskRROnce(matX, vecW, xi, vecGamma, sketchSizes, matU, vecS, matV, vecLev):
    Repeat = 10 #### can be tuned
    
    n, d = matX.shape
    vecF = numpy.dot(matX, vecW)
    vecNoise = numpy.random.randn(n, 1) * xi
    vecY = vecF + vecNoise
    vecXy = numpy.dot(matX.T, vecY)
    
    lenGamma = len(vecGamma)
    lenS = len(sketchSizes)
    
    # solve RR optimally
    print('Doing optimal RR...')
    riskRR = numpy.zeros((lenGamma, 1))
    objRR = numpy.zeros((lenGamma, 1))
    vecUy = numpy.dot(matU.T, vecY)
    for i in range(lenGamma):
        gamma = vecGamma[i]
        vecRR = rrSolver(matU, vecS, matV, vecUy, n * gamma)
        vecXw = numpy.dot(matX, vecRR)
        # risk function
        err = numpy.linalg.norm(vecXw - vecF)
        riskRR[i] = err ** 2 / n
        # objective function value
        err = numpy.linalg.norm(vecXw - vecY)
        reg = numpy.linalg.norm(vecRR)
        del vecRR
        del vecXw
        objRR[i] = err ** 2 / n + gamma * (reg ** 2)
    resultDict = {'xi': xi,
                 'vecGamma': vecGamma,
                 'sketchSizes': sketchSizes,
                 'riskRR': riskRR,
                 'objRR': objRR}
        
    # solve RR with uniform sampling
    print('Doing uniform sampling...')
    riskUniform = numpy.zeros((lenGamma, lenS))
    objUniform = numpy.zeros((lenGamma, lenS))
    riskUniform2 = numpy.zeros((lenGamma, lenS))
    objUniform2 = numpy.zeros((lenGamma, lenS))
    for j in range(lenS):
        s = sketchSizes[j]
        Errors = numpy.zeros((lenGamma, Repeat))
        Objs = numpy.zeros((lenGamma, Repeat))
        Errors2 = numpy.zeros((lenGamma, Repeat))
        Objs2 = numpy.zeros((lenGamma, Repeat))
        for rep in range(Repeat):
            idx = numpy.random.choice(n, s, replace=False)
            idx = numpy.unique(idx)
            matXsketch = matX[idx, :] * numpy.sqrt(n/s)
            vecYsketch = vecY[idx] * numpy.sqrt(n/s)
            matUsketch, vecSsketch, matVsketch = numpy.linalg.svd(matXsketch, full_matrices=False)
            vecUy = numpy.dot(matUsketch.T, vecYsketch)
            for i in range(lenGamma):
                gamma = vecGamma[i]
                # Method 1
                vecRR = rrSolver(matUsketch, vecSsketch, matVsketch, vecUy, n * gamma)
                vecXw = numpy.dot(matX, vecRR)
                err = numpy.linalg.norm(vecXw - vecF)
                Errors[i, rep] = err ** 2 / n
                err = numpy.linalg.norm(vecXw - vecY)
                reg = numpy.linalg.norm(vecRR)
                Objs[i, rep] = (err ** 2) / n + gamma * (reg ** 2)
                # Method 2
                vecRR = rrSolver2(matVsketch, vecSsketch, vecXy, n * gamma)
                vecXw = numpy.dot(matX, vecRR)
                err = numpy.linalg.norm(vecXw - vecF)
                Errors2[i, rep] = err ** 2 / n
                err = numpy.linalg.norm(vecXw - vecY)
                reg = numpy.linalg.norm(vecRR)
                Objs2[i, rep] = (err ** 2) / n + gamma * (reg ** 2)
                del vecRR
                del vecXw
            del vecUy
            del matXsketch
            del vecYsketch
            del matUsketch
            del vecSsketch
            del matVsketch
        riskUniform[:, j] = numpy.mean(Errors, axis=1)
        objUniform[:, j] = numpy.mean(Objs, axis=1)
        riskUniform2[:, j] = numpy.mean(Errors2, axis=1)
        objUniform2[:, j] = numpy.mean(Objs2, axis=1)
    resultDict['riskUniform'] = riskUniform
    resultDict['objUniform'] = objUniform
    resultDict['riskUniform2'] = riskUniform2
    resultDict['objUniform2'] = objUniform2
    
    
    # solve RR with leverage score sampling
    print('Doing leverage score sampling...')
    prob = vecLev / numpy.sum(vecLev)
    scaling = 1 / numpy.sqrt(s * prob)
    scaling = scaling.reshape(n, 1)
    riskLev = numpy.zeros((lenGamma, lenS))
    objLev = numpy.zeros((lenGamma, lenS))
    riskLev2 = numpy.zeros((lenGamma, lenS))
    objLev2 = numpy.zeros((lenGamma, lenS))
    for j in range(lenS):
        s = sketchSizes[j]
        Errors = numpy.zeros((lenGamma, Repeat))
        Objs = numpy.zeros((lenGamma, Repeat))
        Errors2 = numpy.zeros((lenGamma, Repeat))
        Objs2 = numpy.zeros((lenGamma, Repeat))
        for rep in range(Repeat):
            idx = numpy.random.choice(n, s, replace=False, p=prob)
            idx = numpy.unique(idx)
            matXsketch = matX[idx, :] * scaling[idx]
            vecYsketch = vecY[idx] * scaling[idx]
            matUsketch, vecSsketch, matVsketch = numpy.linalg.svd(matXsketch, full_matrices=False)
            vecUy = numpy.dot(matUsketch.T, vecYsketch)
            for i in range(lenGamma):
                gamma = vecGamma[i]
                # Method 1
                vecRR = rrSolver(matUsketch, vecSsketch, matVsketch, vecUy, n * gamma)
                vecXw = numpy.dot(matX, vecRR)
                err = numpy.linalg.norm(vecXw - vecF)
                Errors[i, rep] = err ** 2 / n
                err = numpy.linalg.norm(vecXw - vecY)
                reg = numpy.linalg.norm(vecRR)
                Objs[i, rep] = (err ** 2) / n + gamma * (reg ** 2)
                # Method 2
                vecRR = rrSolver2(matVsketch, vecSsketch, vecXy, n * gamma)
                vecXw = numpy.dot(matX, vecRR)
                err = numpy.linalg.norm(vecXw - vecF)
                Errors2[i, rep] = err ** 2 / n
                err = numpy.linalg.norm(vecXw - vecY)
                reg = numpy.linalg.norm(vecRR)
                Objs2[i, rep] = (err ** 2) / n + gamma * (reg ** 2)
                del vecRR
                del vecXw
            del vecUy
            del matXsketch
            del vecYsketch
            del matUsketch
            del vecSsketch
            del matVsketch
        riskLev[:, j] = numpy.mean(Errors, axis=1)
        objLev[:, j] = numpy.mean(Objs, axis=1)
        riskLev2[:, j] = numpy.mean(Errors2, axis=1)
        objLev2[:, j] = numpy.mean(Objs2, axis=1)
    resultDict['riskLev'] = riskLev
    resultDict['objLev'] = objLev
    resultDict['riskLev2'] = riskLev2
    resultDict['objLev2'] = objLev2
    
    
    
    # solve RR with leverage score sampling (without scaling)
    print('Doing leverage score sampling (without scaling)...')
    prob = vecLev / numpy.sum(vecLev) 
    riskLevU = numpy.zeros((lenGamma, lenS))
    objLevU = numpy.zeros((lenGamma, lenS))
    riskLevU2 = numpy.zeros((lenGamma, lenS))
    objLevU2 = numpy.zeros((lenGamma, lenS))
    for j in range(lenS):
        s = sketchSizes[j]
        Errors = numpy.zeros((lenGamma, Repeat))
        Objs = numpy.zeros((lenGamma, Repeat))
        Errors2 = numpy.zeros((lenGamma, Repeat))
        Objs2 = numpy.zeros((lenGamma, Repeat))
        for rep in range(Repeat):
            idx = numpy.random.choice(n, s, replace=False, p=prob)
            idx = numpy.unique(idx)
            matXsketch = matX[idx, :] * numpy.sqrt(n/s)
            vecYsketch = vecY[idx] * numpy.sqrt(n/s)
            matUsketch, vecSsketch, matVsketch = numpy.linalg.svd(matXsketch, full_matrices=False)
            vecUy = numpy.dot(matUsketch.T, vecYsketch)
            for i in range(lenGamma):
                gamma = vecGamma[i]
                # Method 1
                vecRR = rrSolver(matUsketch, vecSsketch, matVsketch, vecUy, n * gamma)
                vecXw = numpy.dot(matX, vecRR)
                err = numpy.linalg.norm(vecXw - vecF)
                Errors[i, rep] = err ** 2 / n
                err = numpy.linalg.norm(vecXw - vecY)
                reg = numpy.linalg.norm(vecRR)
                Objs[i, rep] = (err ** 2) / n + gamma * (reg ** 2)
                # Method 2
                vecRR = rrSolver2(matVsketch, vecSsketch, vecXy, n * gamma)
                vecXw = numpy.dot(matX, vecRR)
                err = numpy.linalg.norm(vecXw - vecF)
                Errors2[i, rep] = err ** 2 / n
                err = numpy.linalg.norm(vecXw - vecY)
                reg = numpy.linalg.norm(vecRR)
                Objs2[i, rep] = (err ** 2) / n + gamma * (reg ** 2)
                del vecRR
                del vecXw
            del vecUy
            del matXsketch
            del vecYsketch
            del matUsketch
            del vecSsketch
            del matVsketch
        riskLevU[:, j] = numpy.mean(Errors, axis=1)
        objLevU[:, j] = numpy.mean(Objs, axis=1)
        riskLevU2[:, j] = numpy.mean(Errors2, axis=1)
        objLevU2[:, j] = numpy.mean(Objs2, axis=1)
    resultDict['riskLevU'] = riskLevU
    resultDict['objLevU'] = objLevU
    resultDict['riskLevU2'] = riskLevU2
    resultDict['objLevU2'] = objLevU2

    # solve RR with leverage+uniform sampling
    print('Doing mixed sampling...')
    prob = vecLev / numpy.sum(vecLev)
    prob = (prob + 1/n) / 2
    scaling = 1 / numpy.sqrt(s * prob)
    scaling = scaling.reshape(n, 1)
    riskMix = numpy.zeros((lenGamma, lenS))
    objMix = numpy.zeros((lenGamma, lenS))
    riskMix2 = numpy.zeros((lenGamma, lenS))
    objMix2 = numpy.zeros((lenGamma, lenS))
    for j in range(lenS):
        s = sketchSizes[j]
        Errors = numpy.zeros((lenGamma, Repeat))
        Objs = numpy.zeros((lenGamma, Repeat))
        Errors2 = numpy.zeros((lenGamma, Repeat))
        Objs2 = numpy.zeros((lenGamma, Repeat))
        for rep in range(Repeat):
            idx = numpy.random.choice(n, s, replace=False, p=prob)
            idx = numpy.unique(idx)
            matXsketch = matX[idx, :] * scaling[idx]
            vecYsketch = vecY[idx] * scaling[idx]
            matUsketch, vecSsketch, matVsketch = numpy.linalg.svd(matXsketch, full_matrices=False)
            vecUy = numpy.dot(matUsketch.T, vecYsketch)
            for i in range(lenGamma):
                gamma = vecGamma[i]
                # Method 1
                vecRR = rrSolver(matUsketch, vecSsketch, matVsketch, vecUy, n * gamma)
                vecXw = numpy.dot(matX, vecRR)
                err = numpy.linalg.norm(vecXw - vecF)
                Errors[i, rep] = err ** 2 / n
                err = numpy.linalg.norm(vecXw - vecY)
                reg = numpy.linalg.norm(vecRR)
                Objs[i, rep] = (err ** 2) / n + gamma * (reg ** 2)
                # Method 2
                vecRR = rrSolver2(matVsketch, vecSsketch, vecXy, n * gamma)
                vecXw = numpy.dot(matX, vecRR)
                err = numpy.linalg.norm(vecXw - vecF)
                Errors2[i, rep] = err ** 2 / n
                err = numpy.linalg.norm(vecXw - vecY)
                reg = numpy.linalg.norm(vecRR)
                Objs2[i, rep] = (err ** 2) / n + gamma * (reg ** 2)
                del vecRR
                del vecXw
            del vecUy
            del matXsketch
            del vecYsketch
            del matUsketch
            del vecSsketch
            del matVsketch
        riskMix[:, j] = numpy.mean(Errors, axis=1)
        objMix[:, j] = numpy.mean(Objs, axis=1)
        riskMix2[:, j] = numpy.mean(Errors2, axis=1)
        objMix2[:, j] = numpy.mean(Objs2, axis=1)
    resultDict['riskMix'] = riskMix
    resultDict['objMix'] = objMix
    resultDict['riskMix2'] = riskMix2
    resultDict['objMix2'] = objMix2
    
    
    # solve RR with leverage+uniform sampling (without scaling)
    print('Doing mixed sampling (without scaling)...')
    prob = vecLev / numpy.sum(vecLev)
    prob = (prob + 1/n) / 2 
    riskMixU = numpy.zeros((lenGamma, lenS))
    objMixU = numpy.zeros((lenGamma, lenS))
    riskMixU2 = numpy.zeros((lenGamma, lenS))
    objMixU2 = numpy.zeros((lenGamma, lenS))
    for j in range(lenS):
        s = sketchSizes[j]
        Errors = numpy.zeros((lenGamma, Repeat))
        Objs = numpy.zeros((lenGamma, Repeat))
        Errors2 = numpy.zeros((lenGamma, Repeat))
        Objs2 = numpy.zeros((lenGamma, Repeat))
        for rep in range(Repeat):
            idx = numpy.random.choice(n, s, replace=False, p=prob)
            idx = numpy.unique(idx)
            matXsketch = matX[idx, :] * numpy.sqrt(n/s)
            vecYsketch = vecY[idx] * numpy.sqrt(n/s)
            matUsketch, vecSsketch, matVsketch = numpy.linalg.svd(matXsketch, full_matrices=False)
            vecUy = numpy.dot(matUsketch.T, vecYsketch)
            for i in range(lenGamma):
                gamma = vecGamma[i]
                # Method 1
                vecRR = rrSolver(matUsketch, vecSsketch, matVsketch, vecUy, n * gamma)
                vecXw = numpy.dot(matX, vecRR)
                err = numpy.linalg.norm(vecXw - vecF)
                Errors[i, rep] = err ** 2 / n
                err = numpy.linalg.norm(vecXw - vecY)
                reg = numpy.linalg.norm(vecRR)
                Objs[i, rep] = (err ** 2) / n + gamma * (reg ** 2)
                # Method 2
                vecRR = rrSolver2(matVsketch, vecSsketch, vecXy, n * gamma)
                vecXw = numpy.dot(matX, vecRR)
                err = numpy.linalg.norm(vecXw - vecF)
                Errors2[i, rep] = err ** 2 / n
                err = numpy.linalg.norm(vecXw - vecY)
                reg = numpy.linalg.norm(vecRR)
                Objs2[i, rep] = (err ** 2) / n + gamma * (reg ** 2)
                del vecRR
                del vecXw
            del vecUy
            del matXsketch
            del vecYsketch
            del matUsketch
            del vecSsketch
            del matVsketch
        riskMixU[:, j] = numpy.mean(Errors, axis=1)
        objMixU[:, j] = numpy.mean(Objs, axis=1)
        riskMixU2[:, j] = numpy.mean(Errors2, axis=1)
        objMixU2[:, j] = numpy.mean(Objs2, axis=1)
    resultDict['riskMixU'] = riskMixU
    resultDict['objMixU'] = objMixU
    resultDict['riskMixU2'] = riskMixU2
    resultDict['objMixU2'] = objMixU2
    
    
    # solve RR with Gaussian Projection
    print('Doing Gaussian projection...')
    riskGauss = numpy.zeros((lenGamma, lenS))
    objGauss = numpy.zeros((lenGamma, lenS))
    riskGauss2 = numpy.zeros((lenGamma, lenS))
    objGauss2 = numpy.zeros((lenGamma, lenS))
    for j in range(lenS):
        s = sketchSizes[j]
        Errors = numpy.zeros((lenGamma, Repeat))
        Objs = numpy.zeros((lenGamma, Repeat))
        Errors2 = numpy.zeros((lenGamma, Repeat))
        Objs2 = numpy.zeros((lenGamma, Repeat))
        for rep in range(Repeat):
            matSketch = numpy.random.randn(s, n) / numpy.sqrt(s)
            matXsketch = numpy.dot(matSketch, matX)
            vecYsketch = numpy.dot(matSketch, vecY)
            matUsketch, vecSsketch, matVsketch = numpy.linalg.svd(matXsketch, full_matrices=False)
            vecUy = numpy.dot(matUsketch.T, vecYsketch)
            for i in range(lenGamma):
                gamma = vecGamma[i]
                # Method 1
                vecRR = rrSolver(matUsketch, vecSsketch, matVsketch, vecUy, n * gamma)
                vecXw = numpy.dot(matX, vecRR)
                err = numpy.linalg.norm(vecXw - vecF)
                Errors[i, rep] = err ** 2 / n
                err = numpy.linalg.norm(vecXw - vecY)
                reg = numpy.linalg.norm(vecRR)
                Objs[i, rep] = (err ** 2) / n + gamma * (reg ** 2)
                # Method 2
                vecRR = rrSolver2(matVsketch, vecSsketch, vecXy, n * gamma)
                vecXw = numpy.dot(matX, vecRR)
                err = numpy.linalg.norm(vecXw - vecF)
                Errors2[i, rep] = err ** 2 / n
                err = numpy.linalg.norm(vecXw - vecY)
                reg = numpy.linalg.norm(vecRR)
                Objs2[i, rep] = (err ** 2) / n + gamma * (reg ** 2)
                del vecRR
                del vecXw
            del vecUy
            del matXsketch
            del vecYsketch
            del matUsketch
            del vecSsketch
            del matVsketch
        riskGauss[:, j] = numpy.mean(Errors, axis=1)
        objGauss[:, j] = numpy.mean(Objs, axis=1)
        riskGauss2[:, j] = numpy.mean(Errors2, axis=1)
        objGauss2[:, j] = numpy.mean(Objs2, axis=1)
    resultDict['riskGauss'] = riskGauss
    resultDict['objGauss'] = objGauss
    resultDict['riskGauss2'] = riskGauss2
    resultDict['objGauss2'] = objGauss2
    
    # solve RR with SRFT
    print('Doing SRFT...')
    riskSRFT = numpy.zeros((lenGamma, lenS))
    objSRFT = numpy.zeros((lenGamma, lenS))
    riskSRFT2 = numpy.zeros((lenGamma, lenS))
    objSRFT2 = numpy.zeros((lenGamma, lenS))
    for j in range(lenS):
        s = sketchSizes[j]
        Errors = numpy.zeros((lenGamma, Repeat))
        Objs = numpy.zeros((lenGamma, Repeat))
        Errors2 = numpy.zeros((lenGamma, Repeat))
        Objs2 = numpy.zeros((lenGamma, Repeat))
        for rep in range(Repeat):
            matXsketch, vecYsketch = srft.srft2(matX, vecY, s)
            matUsketch, vecSsketch, matVsketch = numpy.linalg.svd(matXsketch, full_matrices=False)
            vecUy = numpy.dot(matUsketch.T, vecYsketch)
            for i in range(lenGamma):
                gamma = vecGamma[i]
                # Method 1
                vecRR = rrSolver(matUsketch, vecSsketch, matVsketch, vecUy, n * gamma)
                vecXw = numpy.dot(matX, vecRR)
                err = numpy.linalg.norm(vecXw - vecF)
                Errors[i, rep] = err ** 2 / n
                err = numpy.linalg.norm(vecXw - vecY)
                reg = numpy.linalg.norm(vecRR)
                Objs[i, rep] = (err ** 2) / n + gamma * (reg ** 2)
                # Method 2
                vecRR = rrSolver2(matVsketch, vecSsketch, vecXy, n * gamma)
                vecXw = numpy.dot(matX, vecRR)
                err = numpy.linalg.norm(vecXw - vecF)
                Errors2[i, rep] = err ** 2 / n
                err = numpy.linalg.norm(vecXw - vecY)
                reg = numpy.linalg.norm(vecRR)
                Objs2[i, rep] = (err ** 2) / n + gamma * (reg ** 2)
                del vecRR
                del vecXw
            del vecUy
            del matXsketch
            del vecYsketch
            del matUsketch
            del vecSsketch
            del matVsketch
        riskSRFT[:, j] = numpy.mean(Errors, axis=1)
        objSRFT[:, j] = numpy.mean(Objs, axis=1)
        riskSRFT2[:, j] = numpy.mean(Errors2, axis=1)
        objSRFT2[:, j] = numpy.mean(Objs2, axis=1)
    resultDict['riskSRFT'] = riskSRFT
    resultDict['objSRFT'] = objSRFT
    resultDict['riskSRFT2'] = riskSRFT2
    resultDict['objSRFT2'] = objSRFT2
    
    # solve RR with count sketch
    print('Doing count sketch...')
    riskCount = numpy.zeros((lenGamma, lenS))
    objCount = numpy.zeros((lenGamma, lenS))
    riskCount2 = numpy.zeros((lenGamma, lenS))
    objCount2 = numpy.zeros((lenGamma, lenS))
    for j in range(lenS):
        s = sketchSizes[j]
        Errors = numpy.zeros((lenGamma, Repeat))
        Objs = numpy.zeros((lenGamma, Repeat))
        Errors2 = numpy.zeros((lenGamma, Repeat))
        Objs2 = numpy.zeros((lenGamma, Repeat))
        for rep in range(Repeat):
            matXsketch, vecYsketch = csketch.countsketch(matX, vecY, s)
            matUsketch, vecSsketch, matVsketch = numpy.linalg.svd(matXsketch, full_matrices=False)
            vecUy = numpy.dot(matUsketch.T, vecYsketch)
            for i in range(lenGamma):
                gamma = vecGamma[i]
                # Method 1
                vecRR = rrSolver(matUsketch, vecSsketch, matVsketch, vecUy, n * gamma)
                vecXw = numpy.dot(matX, vecRR)
                err = numpy.linalg.norm(vecXw - vecF)
                Errors[i, rep] = err ** 2 / n
                err = numpy.linalg.norm(vecXw - vecY)
                reg = numpy.linalg.norm(vecRR)
                Objs[i, rep] = (err ** 2) / n + gamma * (reg ** 2)
                # Method 2
                vecRR = rrSolver2(matVsketch, vecSsketch, vecXy, n * gamma)
                vecXw = numpy.dot(matX, vecRR)
                err = numpy.linalg.norm(vecXw - vecF)
                Errors2[i, rep] = err ** 2 / n
                err = numpy.linalg.norm(vecXw - vecY)
                reg = numpy.linalg.norm(vecRR)
                Objs2[i, rep] = (err ** 2) / n + gamma * (reg ** 2)
                del vecRR
                del vecXw
            del vecUy
            del matXsketch
            del vecYsketch
            del matUsketch
            del vecSsketch
            del matVsketch
        riskCount[:, j] = numpy.mean(Errors, axis=1)
        objCount[:, j] = numpy.mean(Objs, axis=1)
        riskCount2[:, j] = numpy.mean(Errors2, axis=1)
        objCount2[:, j] = numpy.mean(Objs2, axis=1)
    resultDict['riskCount'] = riskCount
    resultDict['objCount'] = objCount
    resultDict['riskCount2'] = riskCount2
    resultDict['objCount2'] = objCount2
    
    return resultDict

def empiricalRiskRR(matX, vecW, xi, vecGamma, sketchSizes, matU, vecS, matV, vecLev):
    Repeat = 10
    
    n, d = matX.shape
    lenGamma = len(vecGamma)
    lenS = len(sketchSizes)
    
    riskRR = numpy.zeros((lenGamma, lenS))
    objRR = numpy.zeros((lenGamma, lenS))
    riskUniform = numpy.zeros((lenGamma, lenS))
    objUniform = numpy.zeros((lenGamma, lenS))
    riskLev = numpy.zeros((lenGamma, lenS))
    objLev = numpy.zeros((lenGamma, lenS))
    riskLevU = numpy.zeros((lenGamma, lenS))
    objLevU = numpy.zeros((lenGamma, lenS))
    riskMix = numpy.zeros((lenGamma, lenS))
    objMix = numpy.zeros((lenGamma, lenS))
    riskMixU = numpy.zeros((lenGamma, lenS))
    objMixU = numpy.zeros((lenGamma, lenS))
    riskGauss = numpy.zeros((lenGamma, lenS))
    objGauss = numpy.zeros((lenGamma, lenS))
    riskSRFT = numpy.zeros((lenGamma, lenS))
    objSRFT = numpy.zeros((lenGamma, lenS))
    riskCount = numpy.zeros((lenGamma, lenS))
    objCount = numpy.zeros((lenGamma, lenS))
    riskUniform2 = numpy.zeros((lenGamma, lenS))
    objUniform2 = numpy.zeros((lenGamma, lenS))
    riskLev2 = numpy.zeros((lenGamma, lenS))
    objLev2 = numpy.zeros((lenGamma, lenS))
    riskLevU2 = numpy.zeros((lenGamma, lenS))
    objLevU2 = numpy.zeros((lenGamma, lenS))
    riskMix2 = numpy.zeros((lenGamma, lenS))
    objMix2 = numpy.zeros((lenGamma, lenS))
    riskMixU2 = numpy.zeros((lenGamma, lenS))
    objMixU2 = numpy.zeros((lenGamma, lenS))
    riskGauss2 = numpy.zeros((lenGamma, lenS))
    objGauss2 = numpy.zeros((lenGamma, lenS))
    riskSRFT2 = numpy.zeros((lenGamma, lenS))
    objSRFT2 = numpy.zeros((lenGamma, lenS))
    riskCount2 = numpy.zeros((lenGamma, lenS))
    objCount2 = numpy.zeros((lenGamma, lenS))
    
    
    
    for rep in range(Repeat):
        print('Pass #' + str(rep))
        resultDict1 = empiricalRiskRROnce(matX, vecW, xi, vecGamma, sketchSizes, matU, vecS, matV, vecLev)
        
        obj = resultDict1['objRR']
        
        riskRR += resultDict1['riskRR'] / Repeat
        
        riskUniform += resultDict1['riskUniform'] / Repeat
        objUniform += resultDict1['objUniform'] / obj / Repeat
        riskLev += resultDict1['riskLev'] / Repeat
        objLev += resultDict1['objLev'] / obj / Repeat
        riskLevU += resultDict1['riskLevU'] / Repeat
        objLevU += resultDict1['objLevU'] / obj / Repeat
        riskMix += resultDict1['riskMix'] / Repeat
        objMix += resultDict1['objMix'] / obj / Repeat
        riskMixU += resultDict1['riskMixU'] / Repeat
        objMixU += resultDict1['objMixU'] / obj / Repeat
        riskGauss += resultDict1['riskGauss'] / Repeat
        objGauss += resultDict1['objGauss'] / obj / Repeat
        riskSRFT += resultDict1['riskSRFT'] / Repeat
        objSRFT += resultDict1['objSRFT'] / obj / Repeat
        riskCount += resultDict1['riskCount'] / Repeat
        objCount += resultDict1['objCount'] / obj / Repeat
        
        
        riskUniform2 += resultDict1['riskUniform2'] / Repeat
        objUniform2 += resultDict1['objUniform2'] / obj / Repeat
        riskLev2 += resultDict1['riskLev2'] / Repeat
        objLev2 += resultDict1['objLev2'] / obj / Repeat
        riskLevU2 += resultDict1['riskLevU2'] / Repeat
        objLevU2 += resultDict1['objLevU2'] / obj / Repeat
        riskMix2 += resultDict1['riskMix2'] / Repeat
        objMix2 += resultDict1['objMix2'] / obj / Repeat
        riskMixU2 += resultDict1['riskMixU2'] / Repeat
        objMixU2 += resultDict1['objMixU2'] / obj / Repeat
        riskGauss2 += resultDict1['riskGauss2'] / Repeat
        objGauss2 += resultDict1['objGauss2'] / obj / Repeat
        riskSRFT2 += resultDict1['riskSRFT2'] / Repeat
        objSRFT2 += resultDict1['objSRFT2'] / obj / Repeat
        riskCount2 += resultDict1['riskCount2'] / Repeat
        objCount2 += resultDict1['objCount2'] / obj / Repeat
        
    resultDict = {'xi': xi,
                 'vecGamma': vecGamma,
                 'sketchSizes': sketchSizes,
                 'riskRR': riskRR,
                 'riskUniform': riskUniform,
                 'objUniform': objUniform,
                 'riskLev': riskLev,
                 'objLev': objLev,
                 'riskLevU': riskLevU,
                 'objLevU': objLevU,
                 'riskMix': riskMix,
                 'objMix': objMix,
                 'riskMixU': riskMixU,
                 'objMixU': objMixU,
                 'riskGauss': riskGauss,
                 'objGauss': objGauss,
                 'riskSRFT': riskSRFT,
                 'objSRFT': objSRFT,
                 'riskCount': riskCount,
                 'objCount': objCount,
                 'riskUniform2': riskUniform2,
                 'objUniform2': objUniform2,
                 'riskLev2': riskLev2,
                 'objLev2': objLev2,
                 'riskLevU2': riskLevU2,
                 'objLevU2': objLevU2,
                 'riskMix2': riskMix2,
                 'objMix2': objMix2,
                 'riskMixU2': riskMixU2,
                 'objMixU2': objMixU2,
                 'riskGauss2': riskGauss2,
                 'objGauss2': objGauss2,
                 'riskSRFT2': riskSRFT2,
                 'objSRFT2': objSRFT2,
                 'riskCount2': riskCount2,
                 'objCount2': objCount2}
    return resultDict
    
