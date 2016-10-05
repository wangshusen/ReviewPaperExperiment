import numpy
import sys
sys.path.append('../../main/kernel/')
import rbfKernelApprox

def crossValid(matXtrain, vecYtrain, numFeature, parameters):
    # ====================== Constants ====================== #
    n = matXtrain.shape[0]
        
    # ==================== Data-Partition ==================== #
    foldOfCrossValid = 5 ######### can be tuned
    randperm = numpy.random.permutation(n)
    nBegin = 0
    nStep = int(numpy.ceil(n / foldOfCrossValid)) + 1
    idxTrainValidData = list()
    for i in range(foldOfCrossValid):
        nEnd = min(nBegin + nStep, n)
        idx = randperm[nBegin: nEnd]
        idxTrainValidData.append(idx)
        nBegin = nEnd
        
    listVecY = list()
    for fold in range(foldOfCrossValid):
        idx = idxTrainValidData[fold]
        listVecY.append(vecYtrain[idx, :])
        
    # ====================== Parameters ====================== #
    sigmaRange = uniformSampling(parameters['sigmaLower'], parameters['sigmaUpper'], (parameters['sigmaNum'], 1))
    gammaRange = uniformSampling(parameters['gammaLower'], parameters['gammaUpper'], (parameters['sigmaNum'], parameters['gammaNum']))
    gammaRange = numpy.exp(gammaRange)
    
    # =================== Cross-Validation =================== #
    matMSE = numpy.zeros((parameters['sigmaNum'], parameters['gammaNum']))
    for i in range(parameters['sigmaNum']):
        sigma = sigmaRange[i]
        # ====== extract features from the training data ====== #
        if parameters['method'] == 'Nystrom':
            matUL, vecSL = rbfKernelApprox.nystrom(matXtrain, sigma, numFeature, parameters['sketch'])
        if parameters['method'] == 'FastSPSD':
            matUL, vecSL = rbfKernelApprox.fastSPSD(matXtrain, sigma, numFeature, parameters['sketch'])
        if parameters['method'] == 'RandFeature':
            matUL, vecSL = rbfKernelApprox.randFeature(matXtrain, sigma, numFeature)
       
        # ==================== Data-Partition ==================== #
        listMatUL = list()
        for fold in range(foldOfCrossValid):
            idx = idxTrainValidData[fold]
            listMatUL.append(matUL[idx, :])
        del matUL
        
        # ======= Cross-Validation for Fixed sigma and Various gamma====== #
        arrGamma = gammaRange[i, :]
        arrMSE = crossValidGamma(listMatUL, vecSL, listVecY, arrGamma)
        matMSE[i, :] = arrMSE
    
    # ====== Find the Best sigma and gamma ====== #
    idx = matMSE.argmin()
    idx0 = int(numpy.floor(idx / parameters['gammaNum']))
    idx1 = numpy.mod(idx, parameters['gammaNum'])
    sigmaOpt = sigmaRange[idx0, 0]
    gammaOpt = gammaRange[idx0, idx1]
    mse = matMSE[idx0, idx1]
    print('mse = ' + str(mse) + ', sigma = ' + str(sigmaOpt) + ', gamma = ' + str(gammaOpt))
    return sigmaOpt, gammaOpt, mse
        
            



def crossValidGamma(listMatUL, vecSL, listVecY, arrGamma):
    # ======================= Preparations ======================= #
    foldOfCrossValid = len(listMatUL)
    numGamma = arrGamma.shape[0]
    arrMSE = numpy.zeros(numGamma)
    d = listMatUL[0].shape[1]
    n = 0
    for vecY in listVecY:
        n = n + vecY.shape[0]

    # ======================= Pre-compute ======================= #
    listVecUy = list()
    listMatU = list()
    listVecSsq = list()
    for fold in range(foldOfCrossValid):
        matU, vecS = numpy.linalg.svd(listMatUL[fold] * vecSL.reshape(1, d), full_matrices=False)[0:2]
        vecSsq = vecS * vecS
        vecUy = numpy.dot(matU.T, listVecY[fold])
        listVecUy.append(vecUy)
        listMatU.append(matU)
        listVecSsq.append(vecSsq.reshape(d, 1))
    del matU
    del vecS
    del vecSsq
    del vecUy
    vecSLsq = (vecSL * vecSL).reshape(d, 1)
    
    # ===================== Cross-Validation ===================== #
    for j in range(numGamma):
        gamma = arrGamma[j]
        # ======================== Train ======================== #
        listModel = list()
        for fold in range(foldOfCrossValid):
            # scale gamma
            gammaScaled = gamma * listMatUL[fold].shape[0]
            # train
            model = listVecUy[fold] / (gammaScaled / listVecSsq[fold] + 1)
            model = numpy.dot(listMatU[fold], model)
            model = listVecY[fold] - model
            model = model / gammaScaled
            # pre-compute the test stage
            model = numpy.dot(listMatUL[fold].T, model)
            model = model * vecSLsq
            listModel.append(model)
        del model
        # ======================= Validate ======================= #
        squaredError = 0
        for fold in range(foldOfCrossValid):
            for foldValid in range(foldOfCrossValid):
                if foldValid != fold:
                    # predict
                    vecYpredict = numpy.dot(listMatUL[foldValid], listModel[fold])
                    # sum the squared error
                    err = numpy.linalg.norm(vecYpredict - listVecY[foldValid])
                    squaredError = squaredError + err * err
        arrMSE[j] = squaredError / (foldOfCrossValid - 1) / n
    return arrMSE
    
    

def uniformSampling(lower, upper, size):
    arr = numpy.random.rand(size[0], size[1])
    diff = upper - lower
    arr = arr * diff + lower
    return numpy.sort(arr)
