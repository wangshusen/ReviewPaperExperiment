import numpy
import time
import sys
sys.path.append('../../main/sketch/')
import countsketch
import srft
import adaptive

def sparseOptCX(matA, parameters):
    '''
    matA: m-by-n csc_matrix or csr_matrix
    '''
    c = parameters['c']
    m, n = matA.shape
    
    t0 = time.time()
    # ================== compute C ================== #
    if parameters['sketch'] == 'gauss':
        matPC = numpy.random.randn(n, c) / numpy.sqrt(c)
        matC = matA * matPC
        del matPC
    elif parameters['sketch'] == 'count':
        matC = countsketch.countsketchSparse1(matA.T, c)
        matC = matC.T
    elif parameters['sketch'] == 'uniform':
        idxC = numpy.random.choice(n, c, replace=False)
        matC = numpy.array(matA[:, idxC].todense())
    elif parameters['sketch'] == 'adaptive':
        idxC = adaptive.adaptiveSamplingSparse(matA, c)
        matC = numpy.array(matA[:, idxC].todense())
    t1 = time.time()
    
    # ================== compute X ================== #
    matX = numpy.linalg.pinv(matC) * matA
    matX = numpy.array(matX)
    t2 = time.time()
        
    TimeCost = {'t1': t1 - t0,
               't2': t2 - t1}
    return matC, matX, TimeCost


def sparseFastCX(matA, parameters):
    '''
    matA: m-by-n csc_matrix or csr_matrix
    '''
    c = parameters['c']
    r = parameters['r']
    m, n = matA.shape
    
    t0 = time.time()
    # ================== compute C and R ================== #
    if parameters['sketch'] == 'gauss':
        matPC = numpy.random.randn(n, c) / numpy.sqrt(c)
        matC = matA * matPC
        del matPC
        matPR = numpy.random.randn(r, m) / numpy.sqrt(r)
        matR = matPR * matA
        matW = numpy.dot(matPR, matC)
        del matPR
    elif parameters['sketch'] == 'count':
        matC = countsketch.countsketchSparse1(matA.T, c)
        matC = matC.T
        matR, matW = countsketch.countsketchSparse2(matA, matC, r)
    elif parameters['sketch'] == 'uniform':
        idxC = numpy.random.choice(n, c, replace=False)
        idxR = numpy.random.choice(m, r, replace=False)
        matC = numpy.array(matA[:, idxC].todense())
        matR = numpy.array(matA[idxR, :].todense())
        matW = matC[idxR, :]
    elif parameters['sketch'] == 'adaptive':
        idxC = adaptive.adaptiveSamplingSparse(matA, c)
        idxR = adaptive.adaptiveSamplingSparse(matA.T, r)
        matC = numpy.array(matA[:, idxC].todense())
        matR = numpy.array(matA[idxR, :].todense())
        matW = matC[idxR, :]
    t1 = time.time()
    
    # ================== compute U ================== #
    matUW, vecSW, matVW = numpy.linalg.svd(matW, full_matrices=False)
    k = int(numpy.ceil(0.8 * len(vecSW)))
    matVW = matVW[0:k, :] / vecSW[0:k].reshape(k, 1)
    matU = numpy.dot(matUW[:, 0:k], matVW)
    t2 = time.time()
        
    TimeCost = {'t1': t1 - t0,
               't2': t2 - t1}
    return matC, matR, matU.T, TimeCost
        
    

def sparseOptCUR(matA, parameters):
    '''
    matA: m-by-n csc_matrix or csr_matrix
    '''
    c = parameters['c']
    r = parameters['r']
    m, n = matA.shape
    
    t0 = time.time()
    # ================== compute C and R ================== #
    if parameters['sketch'] == 'gauss':
        matPC = numpy.random.randn(n, c) / numpy.sqrt(c)
        matC = matA * matPC
        del matPC
        matPR = numpy.random.randn(r, m) / numpy.sqrt(r)
        matR = matPR * matA
        del matPR
    elif parameters['sketch'] == 'count':
        matC = countsketch.countsketchSparse1(matA.T, c)
        matC = matC.T
        matR = countsketch.countsketchSparse1(matA, r)
    elif parameters['sketch'] == 'uniform':
        idxC = numpy.random.choice(n, c, replace=False)
        idxR = numpy.random.choice(m, r, replace=False)
        matC = numpy.array(matA[:, idxC].todense())
        matR = numpy.array(matA[idxR, :].todense())
    elif parameters['sketch'] == 'adaptive':
        idxC = adaptive.adaptiveSamplingSparse(matA, c)
        idxR = adaptive.adaptiveSamplingSparse(matA.T, r)
        matC = numpy.array(matA[:, idxC].todense())
        matR = numpy.array(matA[idxR, :].todense())
    t1 = time.time()
    
    # ================== compute U ================== #
    matU = numpy.linalg.pinv(matC) * matA
    matU = numpy.dot(matU, numpy.linalg.pinv(matR))
    matU = numpy.array(matU)
    t2 = time.time()
        
    TimeCost = {'t1': t1 - t0,
               't2': t2 - t1}
    return matC, matR, matU, TimeCost
        


def sparseFastCUR(matA, parameters):
    '''
    matA: m-by-n csc_matrix or csr_matrix
    '''
    c = parameters['c']
    r = parameters['r']
    ratio = 20 # can be tuned
    sc = c * ratio
    sr = r * ratio
    m, n = matA.shape
    
    t0 = time.time()
    # ================== sketching: step 1 ================== #
    if parameters['sketch'] == 'gauss':
        matPC = numpy.random.randn(n, c) / numpy.sqrt(c)
        matC = matA * matPC
        del matPC
        matPR = numpy.random.randn(r, m) / numpy.sqrt(r)
        matR = matPR * matA
        matB11 = numpy.dot(matPR, matC)
        del matPR
    elif parameters['sketch'] == 'count':
        matC = countsketch.countsketchSparse1(matA.T, c)
        matC = matC.T
        matR, matB11 = countsketch.countsketchSparse2(matA, matC, r)
    elif parameters['sketch'] == 'uniform':
        idxC = numpy.random.choice(n, c, replace=False)
        idxR = numpy.random.choice(m, r, replace=False)
        matC = numpy.array(matA[:, idxC].todense())
        matR = numpy.array(matA[idxR, :].todense())
        matB11 = matC[idxR, :]
    elif parameters['sketch'] == 'adaptive':
        idxC = adaptive.adaptiveSamplingSparse(matA, c)
        idxR = adaptive.adaptiveSamplingSparse(matA.T, r)
        matC = numpy.array(matA[:, idxC].todense())
        matR = numpy.array(matA[idxR, :].todense())
        matB11 = matC[idxR, :]
        
    # ================== sketching: step 2 ================== #
    if parameters['sketch2'] == 'gauss':
        matSC = numpy.random.randn(sc, m) / numpy.sqrt(sc)
        matA2 = matSC * matA
        matB21 = numpy.dot(matSC, matC)
        del matSC
        matSR = numpy.random.randn(n, sr) / numpy.sqrt(sr)
        matB12 = numpy.dot(matR, matSR)
        matB22 = numpy.dot(matA2, matSR)
        del matSR
    elif parameters['sketch2'] == 'count':
        matA2, matB21 = countsketch.countsketchSparse2(matA, matC, sc)
        matB22, matB12 = countsketch.countsketch(matA2.T, matR.T, sr)
        matB12 = matB12.T
        matB22 = matB22.T
    elif parameters['sketch2'] == 'uniform':
        idxSC = numpy.random.choice(m, sc, replace=False)
        idxSR = numpy.random.choice(n, sr, replace=False)
        matB21 = matC[idxSC, :]
        matB12 = matR[:, idxSR]
        matB22 = matA[idxSC, :]
        matB22 = matB22[:, idxSR]
    elif parameters['sketch2'] == 'leverage':
        matQC = numpy.linalg.qr(matC, mode='reduced')[0]
        vecLevC = numpy.sum(matQC ** 2, axis=1)
        vecLevC = vecLevC / numpy.sum(vecLevC)
        matQR = numpy.linalg.qr(matR.T, mode='reduced')[0]
        vecLevR = numpy.sum(matQR ** 2, axis=1)
        vecLevR = vecLevR / numpy.sum(vecLevR)
        idxSC = numpy.random.choice(m, sc, replace=False, p=vecLevC)
        idxSR = numpy.random.choice(n, sr, replace=False, p=vecLevR)
        matB21 = matC[idxSC, :]
        matB12 = matR[:, idxSR]
        matB22 = matA[idxSC, :]
        matB22 = matB22[:, idxSR]
        
    t1 = time.time()
        
    # ================== compute U ================== #
    matB1c = numpy.zeros((r+sc, c))
    matB1c[0:r, :] = matB11
    matB1c[r:r+sc, :] = matB21
    matB1c = numpy.linalg.pinv(matB1c)
    matB1c1 = matB1c[:, 0:r]
    matB1c2 = matB1c[:, r:r+sc]
    matU1 = numpy.dot(matB1c1, matB12)
    if parameters['sketch2'] == 'gauss' or parameters['sketch2'] == 'count':
        matU2 = numpy.dot(matB1c2, matB22)
    elif parameters['sketch2'] == 'uniform' or parameters['sketch2'] == 'leverage':
        matU2 = matB1c2 * matB22
    
    matU = matU1 + matU2
    matB1r = numpy.zeros((r, c+sr))
    matB1r[:, 0:c] = matB11
    matB1r[:, c:c+sr] = matB12
    matB1r = numpy.linalg.pinv(matB1r)
    matU = numpy.dot(matU, matB1r[c:c+sr, :])
    matU = matU + matB1r[0:c, :]
        
    t2 = time.time()
    
    TimeCost = {'t1': t1 - t0,
               't2': t2 - t1}
    return matC, matR, matU, TimeCost