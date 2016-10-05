import numpy
import scipy.cluster.vq as vq
import sys
sys.path.append('../../main/kernel/')
import rbfKernelApprox


def trainPredict(matXtrain, vecYtrain, matXtest, vecYtest, numFeature, sigmaOpt, gammaOpt, parameters):
    # ====================== Prepare ====================== #
    n = matXtrain.shape[0]
    m = matXtest.shape[0]
    matX = numpy.concatenate((matXtrain, matXtest))
    del matXtrain
    del matXtest
        
    # =========== Extract Features from the Data =========== #
    if parameters['method'] == 'Nystrom':
        matUL, vecSL = rbfKernelApprox.nystrom(matX, sigmaOpt, numFeature, parameters['sketch'])
    if parameters['method'] == 'FastSPSD':
        matUL, vecSL = rbfKernelApprox.fastSPSD(matX, sigmaOpt, numFeature, parameters['sketch'])
    if parameters['method'] == 'RandFeature':
        matUL, vecSL = rbfKernelApprox.randFeature(matX, sigmaOpt, numFeature)
    del matX
    d = matUL.shape[1]
    matULtrain = matUL[0:n, :]
    matULtest = matUL[n:n+m, :]
    del matUL
    
    
    # ======================= Train ======================= #
    # scale gamma
    gammaScaled = gammaOpt * n
    # train
    matU, vecS = numpy.linalg.svd(matULtrain * vecSL.reshape(1, d), full_matrices=False)[0:2]
    model = numpy.dot(matU.T, vecYtrain)
    model = model / (gammaScaled / (vecS * vecS) + 1).reshape(d, 1)
    model = numpy.dot(matU, model)
    model = vecYtrain - model
    model = model / gammaScaled
    del matU
    del vecS
    
    # ==================== Generalization ==================== #
    vecYpredict = numpy.dot(matULtrain.T, model)
    del matULtrain
    vecYpredict = vecYpredict * (vecSL * vecSL).reshape(d, 1)
    vecYpredict = numpy.dot(matULtest, vecYpredict)
    # sum the squared error
    err = numpy.linalg.norm(vecYpredict - vecYtest)
    return err * err / m

