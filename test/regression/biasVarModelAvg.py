import numpy
import sys
sys.path.append('../../main/sketch/')
import countsketch as csketch
import srft
sys.path.append('../../main/regression/')
import scipy.io


def bvOptimal(vecS, matVW0, n, gamma, xi):
    # bias
    vec1 = vecS + n * gamma / vecS
    mat1 = matVW0 / vec1.reshape(len(vec1), 1)
    bias = numpy.linalg.norm(mat1) * numpy.sqrt(n) * gamma
    # var
    vec2 = n * gamma / vecS / vecS
    vec2 = 1 / (vec2 + 1)
    var = numpy.linalg.norm(vec2)
    var = var * var * xi * xi / n
    # return
    return bias, var

def bvAvgTilde(vecS, matU, matVW0, gamma, xi, sketch, s, vecG):
    repeat = 20
    lenG = len(vecG)
    maxG = numpy.max(vecG)
    setG = set(vecG)
    vecBias = numpy.zeros((repeat, lenG))
    vecVar = numpy.zeros((repeat, lenG))
    
    n, d = matU.shape
    
    if sketch == 'lev':
        vecLev = numpy.sum(matU ** 2, axis=1)
        prob = vecLev / numpy.sum(vecLev)
        scaling = 1 / numpy.sqrt(s * prob)
        scaling = scaling.reshape(n, 1)
    elif sketch == 'shrink':
        vecLev = numpy.sum(matU ** 2, axis=1)
        prob = vecLev / numpy.sum(vecLev)
        prob = (prob + 1/n) / 2
        scaling = 1 / numpy.sqrt(s * prob)
        scaling = scaling.reshape(n, 1)
    
    for iter in range(repeat):
        matSumB = numpy.zeros((d, 1))
        
        if sketch == 'uni' or sketch == 'lev' or sketch == 'shrink':
            matSumV = numpy.zeros((d, s))
        elif sketch == 'srft':
            matSumV = numpy.zeros((d, s))
            randSigns = numpy.random.choice(2, n) * 2 - 1
            matUsrft = matU * randSigns.reshape(n, 1)
            matUsrft = srft.realfft(matUsrft)
        elif sketch == 'gauss' or sketch == 'count':
            matSumV = numpy.zeros((d, n))

        # model averaging
        pos = 0
        for i in range(maxG):
            if sketch == 'uni':
                idx = numpy.random.choice(n, s, replace=False)
                matUsketch = matU[idx, :] * numpy.sqrt(n/s)
            elif sketch == 'srft':
                idx = numpy.random.choice(n, s, replace=False)
                matUsketch = matUsrft[idx, :] * numpy.sqrt(n/s)
            elif sketch == 'lev' or sketch == 'shrink':
                idx = numpy.random.choice(n, s, replace=False, p=prob)
                matUsketch = matU[idx, :] * scaling[idx]
            elif sketch == 'gauss':
                matSketch = numpy.random.randn(s, n) / numpy.sqrt(s) 
                matUsketch = numpy.dot(matSketch, matU)
            elif sketch == 'count':
                matUsketch, matSketch = csketch.countsketchS(matU, s)
                matSketch = matSketch.T


            mat0 = matUsketch * vecS.reshape(1, d)
            matU1, vecS1, matV1 = numpy.linalg.svd(mat0, full_matrices=False)
            del mat0

            # bias
            vec1 = vecS1 * vecS1 + n * gamma
            mat1 = matVW0 * vecS.reshape(d, 1)
            mat1 = numpy.dot(matV1, mat1)
            mat1 = mat1 / vec1.reshape(d, 1)
            mat1 = numpy.dot(matV1.T, mat1)
            matSumB = matSumB + mat1
            del vec1
            del mat1

            # variance
            vec2 = vecS1 + n * gamma / vecS1
            mat2 = matU1.T / vec2.reshape(d, 1)
            mat2 = numpy.dot(matV1.T, mat2)
            mat2 = mat2 * vecS.reshape(d, 1)
            if sketch == 'uni' or sketch == 'srft':
                mat2 = mat2 * numpy.sqrt(n/s)
            elif sketch == 'lev' or sketch == 'shrink':
                mat2 = mat2 * scaling[idx].reshape(1, s)
            elif sketch == 'gauss' or sketch == 'count':
                mat2 = numpy.dot(mat2, matSketch)
                del matSketch
            matSumV = matSumV + mat2
            del vec2
            del mat2
            del matU1
            del vecS1
            del matV1
            
            g = i + 1
            if g in setG:
                matAvgB = matSumB / g
                matAvgV = matSumV / g
                vecBias[iter, pos] = numpy.linalg.norm(matAvgB) * numpy.sqrt(n) * gamma
                var = numpy.linalg.norm(matAvgV)
                vecVar[iter, pos] = var * var * xi * xi / n
                del matAvgB
                del matAvgV
                pos = pos + 1
                
        del matSumB
        del matSumV
    
    return numpy.mean(vecBias, axis=0), numpy.mean(vecVar, axis=0)
    

    
    
def bvAvgHat(vecS, matU, matVW0, gamma, xi, sketch, s, vecG):
    repeat = 50
    lenG = len(vecG)
    maxG = numpy.max(vecG)
    setG = set(vecG)
    vecBias = numpy.zeros((repeat, lenG))
    vecVar = numpy.zeros((repeat, lenG))
    
    n, d = matU.shape
    matI = numpy.eye(d)
    
    if sketch == 'lev':
        vecLev = numpy.sum(matU ** 2, axis=1)
        prob = vecLev / numpy.sum(vecLev)
        scaling = 1 / numpy.sqrt(s * prob)
        scaling = scaling.reshape(n, 1)
    elif sketch == 'shrink':
        vecLev = numpy.sum(matU ** 2, axis=1)
        prob = vecLev / numpy.sum(vecLev)
        prob = (prob + 1/n) / 2
        scaling = 1 / numpy.sqrt(s * prob)
        scaling = scaling.reshape(n, 1)
    
    for iter in range(repeat):
        if iter % 10 == 0:
            print('iteration ' + str(iter))
        matSumB = numpy.zeros((d, 1))
        matSumV = numpy.zeros((d, d))
        
        if sketch == 'srft':
            randSigns = numpy.random.choice(2, n) * 2 - 1
            matUrft = matU * randSigns.reshape(n, 1)
            matUrft = srft.realfft(matUrft)

        pos = 0
        
        for i in range(maxG):
            if sketch == 'uni':
                idx = numpy.random.choice(n, s, replace=False)
                matUsketch = matU[idx, :] * numpy.sqrt(n/s)
            elif sketch == 'srft':
                idx = numpy.random.choice(n, s, replace=False)
                matUsketch = matUrft[idx, :] * numpy.sqrt(n/s)
            elif sketch == 'lev' or sketch == 'shrink':
                idx = numpy.random.choice(n, s, replace=False, p=prob)
                matUsketch = matU[idx, :] * scaling[idx]
            elif sketch == 'gauss':
                matSketch = numpy.random.randn(s, n) / numpy.sqrt(s) 
                matUsketch = numpy.dot(matSketch, matU)
                del matSketch
            elif sketch == 'count':
                matUsketch = csketch.countsketch1(matU, s)


            mat0 = matUsketch * vecS.reshape(1, d)
            _, vecS1, matV1 = numpy.linalg.svd(mat0, full_matrices=False)
            mat1 = matV1 * vecS.reshape(1, d)
            vec1 = vecS1 * vecS1 + n * gamma
            del matV1
            del vecS1
            mat2 = mat1 / vec1.reshape(d, 1)
            mat2 = numpy.dot(mat1.T, mat2)
            del mat1
            del vec1

            #variance
            #matSumV = matSumV + mat2
            
            # bias
            #mat1 = matVW0 * vecS.reshape(d, 1)
            #mat1 = numpy.dot(mat2, mat1)
            #del mat2
            #mat3 = numpy.dot(mat0.T, mat0) / n / gamma + matI
            #del mat0
            #mat3 = mat3 / vecS.reshape(d, 1) / vecS.reshape(1, d)
            #mat3 = mat3 - matI / n / gamma
            #mat1 = numpy.dot(mat3, mat1)
            #del mat3
            #matSumB = matSumB + mat1
            #del mat1
            
            mat0 = numpy.dot(matUsketch.T, matUsketch)
            mat1 = n * gamma / vecS / vecS
            mat1 = numpy.diag(mat1) + mat0
            mat0 = numpy.linalg.pinv(mat1)
            matSumV = matSumV + mat0
            
            mat2 = mat1 - matI
            del mat1
            mat2 = mat2 / n / gamma
            mat3 = matVW0 * vecS.reshape(d, 1)
            mat4 = numpy.dot(mat0, mat3)
            del mat0
            del mat3
            mat4 = numpy.dot(mat2, mat4)
            del mat2
            matSumB = matSumB + mat4
            del mat4
            
            g = i + 1
            if g in setG:
                matAvgB = matSumB / g
                matAvgV = matSumV / g
                vecBias[iter, pos] = numpy.linalg.norm(matAvgB) * numpy.sqrt(n) * gamma
                var = numpy.linalg.norm(matAvgV)
                vecVar[iter, pos] = var * var * xi * xi / n
                del matAvgB
                del matAvgV
                pos = pos + 1

        del matSumB
        del matSumV
    
    return numpy.mean(vecBias, axis=0), numpy.mean(vecVar, axis=0)
    


def bvExperiment(matX, matW0, s, gamma, xi, vecG, method):
    n, d = matX.shape
    lenG = len(vecG)
    
    matU, vecS, matV = numpy.linalg.svd(matX, full_matrices=False)
    matVW0 = numpy.dot(matV, matW0)
    
    # optimal
    biasOpt, varOpt = bvOptimal(vecS, matVW0, n, gamma, xi)
    
    # uniform
    print('Doing uniform sampling...')
    if method == 'tilde':
        biasUni, varUni = bvAvgTilde(vecS, matU, matVW0, gamma, xi, 'uni', s, vecG)
    elif method == 'hat':
        biasUni, varUni = bvAvgHat(vecS, matU, matVW0, gamma, xi, 'uni', s, vecG)
    
    # leverage
    print('Doing leverage score sampling...')
    if method == 'tilde':
        biasLev, varLev = bvAvgTilde(vecS, matU, matVW0, gamma, xi, 'lev', s, vecG)
    elif method == 'hat':
        biasLev, varLev = bvAvgHat(vecS, matU, matVW0, gamma, xi, 'lev', s, vecG)
    
    # shrinkage leverage
    print('Doing shrinkage leverage score sampling...')
    if method == 'tilde':
        biasShrink, varShrink = bvAvgTilde(vecS, matU, matVW0, gamma, xi, 'shrink', s, vecG)
    elif method == 'hat':
        biasShrink, varShrink = bvAvgHat(vecS, matU, matVW0, gamma, xi, 'shrink', s, vecG)
    
    # Gaussian projection
    print('Doing Gaussian projection...')
    if method == 'tilde':
        biasGauss, varGauss = bvAvgTilde(vecS, matU, matVW0, gamma, xi, 'gauss', s, vecG)
    elif method == 'hat':
        biasGauss, varGauss = bvAvgHat(vecS, matU, matVW0, gamma, xi, 'gauss', s, vecG)

    # SRFT
    print('Doing SRFT...')
    if method == 'tilde':
        biasSrft, varSrft = bvAvgTilde(vecS, matU, matVW0, gamma, xi, 'srft', s, vecG)
    elif method == 'hat':
        biasSrft, varSrft = bvAvgHat(vecS, matU, matVW0, gamma, xi, 'srft', s, vecG)
    
    # Count Sketch
    print('Doing count sketch...')
    if method == 'tilde':
        biasCount, varCount = bvAvgTilde(vecS, matU, matVW0, gamma, xi, 'count', s, vecG)
    elif method == 'hat':
        biasCount, varCount = bvAvgHat(vecS, matU, matVW0, gamma, xi, 'count', s, vecG)

    resultDict = {'gamma': gamma,
                  'g': vecG,
                  'xi': xi,
                  's': s,
                  'biasOpt': biasOpt,
                  'varOpt': varOpt,
                  'biasUni': biasUni,
                  'varUni': varUni,
                  'biasLev': biasLev,
                  'varLev': varLev,
                  'biasShrink': biasShrink,
                  'varShrink': varShrink,
                  'biasGauss': biasGauss,
                  'varGauss': varGauss,
                  'biasSrft': biasSrft,
                  'varSrft': varSrft,
                  'biasCount': biasCount,
                  'varCount': varCount
                 }
    return resultDict


def run(filename):
    # load data
    dataname = filename[0:2]
    dataDict = scipy.io.loadmat(filename)
    matX = dataDict['matX']
    vecW = dataDict['vecW']
    print(matX.shape)
    matU, vecS, matV = numpy.linalg.svd(matX, full_matrices=False)
    
    xi = 0.1
    vecG = [1, 2, 3, 4, 6, 8, 10, 12, 15, 18, 22, 26, 30, 35, 40, 45, 50]
    method = 'hat'

    
    s = 1000
    gamma = 1e-6
    resultDict = bvExperiment(matX, vecW, s, gamma, xi, vecG, method)
    print(resultDict)
    outputFileName = 'avg_' + method +'_' + dataname + '_s' + str(s) + '_gam' + str(gamma) + '.mat'
    scipy.io.savemat(outputFileName, resultDict)
    
    s = 1000
    gamma = 1e-12
    resultDict = bvExperiment(matX, vecW, s, gamma, xi, vecG, method)
    print(resultDict)
    outputFileName = 'avg_' + method +'_' + dataname + '_s' + str(s) + '_gam' + str(gamma) + '.mat'
    scipy.io.savemat(outputFileName, resultDict)
    
    s = 2000
    gamma = 1e-6
    resultDict = bvExperiment(matX, vecW, s, gamma, xi, vecG, method)
    print(resultDict)
    outputFileName = 'avg_' + method +'_' + dataname + '_s' + str(s) + '_gam' + str(gamma) + '.mat'
    scipy.io.savemat(outputFileName, resultDict)
    
    s = 2000
    gamma = 1e-12
    resultDict = bvExperiment(matX, vecW, s, gamma, xi, vecG, method)
    print(resultDict)
    outputFileName = 'avg_' + method +'_' + dataname + '_s' + str(s) + '_gam' + str(gamma) + '.mat'
    scipy.io.savemat(outputFileName, resultDict)
    
    s = 5000
    gamma = 1e-6
    resultDict = bvExperiment(matX, vecW, s, gamma, xi, vecG, method)
    print(resultDict)
    outputFileName = 'avg_' + method +'_' + dataname + '_s' + str(s) + '_gam' + str(gamma) + '.mat'
    scipy.io.savemat(outputFileName, resultDict)
    

    s = 5000
    gamma = 1e-12
    resultDict = bvExperiment(matX, vecW, s, gamma, xi, vecG, method)
    print(resultDict)
    outputFileName = 'avg_' + method +'_' + dataname + '_s' + str(s) + '_gam' + str(gamma) + '.mat'
    scipy.io.savemat(outputFileName, resultDict)
        