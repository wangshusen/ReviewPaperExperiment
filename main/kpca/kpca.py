import numpy
import scipy.cluster.vq as vq
import sys
sys.path.append('../../main/kernel/')
import rbfKernelApprox

def clusteringKPCA(matX, numClass, targetDim, parameters):
    if parameters['method'] == 'Nystrom':
        matU, vecS = rbfKernelApprox.nystrom(matX, parameters['sigma'], parameters['sizeSketch'], parameters['sketch'])
    elif parameters['method'] == 'FastSPSD':
        matU, vecS = rbfKernelApprox.fastSPSD(matX, parameters['sigma'], parameters['sizeSketch'], parameters['sketch'])
    elif parameters['method'] == 'RandFeature':
        matU, vecS = rbfKernelApprox.randFeature(matX, parameters['sigma'], parameters['sizeSketch'])

    matF = matU[:, 0:targetDim] * vecS[0:targetDim].reshape(1, targetDim)
    #matF = vq.whiten(matF)
    codeBook, _ = vq.kmeans(matF, numClass)
    ypred, _ = vq.vq(matF, codeBook)
    return ypred
