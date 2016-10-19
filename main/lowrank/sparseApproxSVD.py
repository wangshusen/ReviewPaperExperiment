import numpy
import time
import sys
sys.path.append('../../main/sketch/')
import countsketch
import adaptive

def sparseOptCXSVD(matA, parameters):
    '''
    matA: m-by-n csc_matrix or csr_matrix
    '''
    c = parameters['c']
    k = parameters['k']
    m, n = matA.shape
    
    t0 = time.time()
    # ================== compute C ================== #
    if parameters['sketch'] == 'Gauss':
        matPC = numpy.random.randn(n, c) / numpy.sqrt(c)
        matC = matA * matPC
        del matPC
    elif parameters['sketch'] == 'Count':
        matC = countsketch.countsketchSparse1(matA.T, c)
        matC = matC.T
    elif parameters['sketch'] == 'Uniform':
        idxC = numpy.random.choice(n, c, replace=False)
        matC = numpy.array(matA[:, idxC].todense())
    elif parameters['sketch'] == 'Adaptive':
        idxC = adaptive.adaptiveSamplingSparse(matA, c)
        matC = numpy.array(matA[:, idxC].todense())
    t1 = time.time()
    
    # ================== compute Q ================== #
    matQC = numpy.linalg.qr(matC, mode='reduced')[0]
    
    # ================== compute X ================== #
    matX = matQC.T * matA
    matX = numpy.array(matX)
    
    # ================== compute SVD ================== #
    matUX, vecSX, matVX = numpy.linalg.svd(matX, full_matrices=False)
    matQC = numpy.dot(matQC, matUX[:, 0:k])
    t2 = time.time()
        
    TimeCost = {'t1': t1 - t0,
               't2': t2 - t1}
    return matQC, vecSX[0:k], matVX[0:k, :], TimeCost




def sparseFastCXSVD(matA, parameters):
    '''
    matA: m-by-n csc_matrix or csr_matrix
    '''
    c = parameters['c']
    r = parameters['r']
    k = parameters['k']
    m, n = matA.shape
    
    t0 = time.time()
    # ================== compute C and R ================== #
    if parameters['sketch'] == 'Gauss':
        matPC = numpy.random.randn(n, c) / numpy.sqrt(c)
        matC = matA * matPC
        del matPC
        matPR = numpy.random.randn(r, m) / numpy.sqrt(r)
        matR = matPR * matA
        matW = numpy.dot(matPR, matC)
        del matPR
    elif parameters['sketch'] == 'Count':
        matC = countsketch.countsketchSparse1(matA.T, c)
        matC = matC.T
        matR, matW = countsketch.countsketchSparse2(matA, matC, r)
    elif parameters['sketch'] == 'Uniform':
        idxC = numpy.random.choice(n, c, replace=False)
        idxR = numpy.random.choice(m, r, replace=False)
        matC = numpy.array(matA[:, idxC].todense())
        matR = numpy.array(matA[idxR, :].todense())
        matW = matC[idxR, :]
    elif parameters['sketch'] == 'Adaptive':
        idxC = adaptive.adaptiveSamplingSparse(matA, c)
        idxR = adaptive.adaptiveSamplingSparse(matA.T, r)
        matC = numpy.array(matA[:, idxC].todense())
        matR = numpy.array(matA[idxR, :].todense())
        matW = matC[idxR, :]
    t1 = time.time()
    
    # ================== compute U ================== #
    matUW, vecSW, matVW = numpy.linalg.svd(matW, full_matrices=False)
    rho = int(numpy.ceil(0.8 * len(vecSW)))
    matVW = matVW[0:rho, :] / vecSW[0:rho].reshape(rho, 1)
    matUW = matUW[:, 0:rho]
    del vecSW
    
    # ================== compute SVD ================== #
    matQC, matRC = numpy.linalg.qr(matC, mode='reduced')[0:2]
    matRC = numpy.dot(matRC, matVW.T)
    del matVW
    matQR, matRR = numpy.linalg.qr(matR.T, mode='reduced')[0:2]
    matRR = numpy.dot(matRR, matUW)
    del matUW
    matU = numpy.dot(matRC, matRR.T)
    del matRC
    del matRR
    matUU, vecSU, matVU = numpy.linalg.svd(matU, full_matrices=False)
    matQC = numpy.dot(matQC, matUU[:, 0:k])
    matQR = numpy.dot(matVU[0:k, :], matQR.T)
    
    t2 = time.time()
        
    TimeCost = {'t1': t1 - t0,
               't2': t2 - t1}
    return matQC, vecSU[0:k], matQR, TimeCost
        
    
def sparseOptCURSVD(matA, parameters):
    '''
    matA: m-by-n csc_matrix or csr_matrix
    '''
    c = parameters['c']
    r = parameters['r']
    k = parameters['k']
    m, n = matA.shape
    
    t0 = time.time()
    # ================== compute C and R ================== #
    if parameters['sketch'] == 'Gauss':
        matPC = numpy.random.randn(n, c) / numpy.sqrt(c)
        matC = matA * matPC
        del matPC
        matPR = numpy.random.randn(r, m) / numpy.sqrt(r)
        matR = matPR * matA
        del matPR
    elif parameters['sketch'] == 'Count':
        matC = countsketch.countsketchSparse1(matA.T, c)
        matC = matC.T
        matR = countsketch.countsketchSparse1(matA, r)
    elif parameters['sketch'] == 'Uniform':
        idxC = numpy.random.choice(n, c, replace=False)
        idxR = numpy.random.choice(m, r, replace=False)
        matC = numpy.array(matA[:, idxC].todense())
        matR = numpy.array(matA[idxR, :].todense())
    elif parameters['sketch'] == 'Adaptive':
        idxC = adaptive.adaptiveSamplingSparse(matA, c)
        idxR = adaptive.adaptiveSamplingSparse(matA.T, r)
        matC = numpy.array(matA[:, idxC].todense())
        matR = numpy.array(matA[idxR, :].todense())
    t1 = time.time()
    
    # ================== compute SVD ================== #
    matQC = numpy.linalg.qr(matC, mode='reduced')[0]
    matQR = numpy.linalg.qr(matR.T, mode='reduced')[0]
    matU = matQC.T * matA
    matU = numpy.dot(matU, matQR)
    matU = numpy.array(matU)
    matUU, vecSU, matVU = numpy.linalg.svd(matU, full_matrices=False)
    matQC = numpy.dot(matQC, matUU[:, 0:k])
    vecSU = vecSU[0:k]
    matQR = numpy.dot(matVU[0:k, :], matQR.T)
    
    
    t2 = time.time()
        
    TimeCost = {'t1': t1 - t0,
               't2': t2 - t1}
    return matQC, vecSU, matQR, TimeCost




def sparseFastCURSVD2(matA, parameters):
    '''
    matA: m-by-n csc_matrix or csr_matrix
    '''
    c = parameters['c']
    r = parameters['r']
    k = parameters['k']
    ratio = 10 # can be tuned
    sc = c * ratio
    sr = r * ratio
    m, n = matA.shape
    
    t0 = time.time()
    # ================== sketching: step 1 ================== #
    if parameters['sketch'] == 'Gauss':
        matPC = numpy.random.randn(n, c) / numpy.sqrt(c)
        matC = matA * matPC
        del matPC
        matPR = numpy.random.randn(r, m) / numpy.sqrt(r)
        matR = matPR * matA
        matB11 = numpy.dot(matPR, matC)
        del matPR
    elif parameters['sketch'] == 'Count':
        matC = countsketch.countsketchSparse1(matA.T, c)
        matC = matC.T
        matR, matB11 = countsketch.countsketchSparse2(matA, matC, r)
    elif parameters['sketch'] == 'Uniform':
        idxC = numpy.random.choice(n, c, replace=False)
        idxR = numpy.random.choice(m, r, replace=False)
        matC = numpy.array(matA[:, idxC].todense())
        matR = numpy.array(matA[idxR, :].todense())
        matB11 = matC[idxR, :]
    elif parameters['sketch'] == 'Adaptive':
        idxC = adaptive.adaptiveSamplingSparse(matA, c)
        idxR = adaptive.adaptiveSamplingSparse(matA.T, r)
        matC = numpy.array(matA[:, idxC].todense())
        matR = numpy.array(matA[idxR, :].todense())
        matB11 = matC[idxR, :]
    t1 = time.time()
    time0 = t1 - t0
        
    
    matQC = numpy.linalg.qr(matC, mode='reduced')[0]
    matQR = numpy.linalg.qr(matR.T, mode='reduced')[0]
    
    t2 = time.time()
    time1 = t2 - t1
    
    
    # ================== sketching: step 2 ================== #
    if parameters['sketch2'] == 'Gauss':
        matSC = numpy.random.randn(sc, m) / numpy.sqrt(sc)
        matA2 = matSC * matA
        matB21 = numpy.dot(matSC, matC)
        del matSC
        matSR = numpy.random.randn(n, sr) / numpy.sqrt(sr)
        matB12 = numpy.dot(matR, matSR)
        matB22 = numpy.dot(matA2, matSR)
        del matSR
    elif parameters['sketch2'] == 'Count':
        matA2, matB21 = countsketch.countsketchSparse2(matA, matC, sc)
        matB22, matB12 = countsketch.countsketch(matA2.T, matR.T, sr)
        matB12 = matB12.T
        matB22 = matB22.T
    elif parameters['sketch2'] == 'Uniform':
        idxSC = numpy.random.choice(m, sc, replace=False)
        idxSR = numpy.random.choice(n, sr, replace=False)
        matB21 = matC[idxSC, :]
        matB12 = matR[:, idxSR]
        matB22 = matA[idxSC, :]
        matB22 = matB22[:, idxSR]
    elif parameters['sketch2'] == 'Leverage':
        vecLevC = numpy.sum(matQC ** 2, axis=1)
        vecLevC = vecLevC / numpy.sum(vecLevC)
        vecLevR = numpy.sum(matQR ** 2, axis=1)
        vecLevR = vecLevR / numpy.sum(vecLevR)
        idxSC = numpy.random.choice(m, sc, replace=False, p=vecLevC)
        idxSR = numpy.random.choice(n, sr, replace=False, p=vecLevR)
        matB21 = matC[idxSC, :]
        matB12 = matR[:, idxSR]
        matB22 = matA[idxSC, :]
        matB22 = matB22[:, idxSR]
        
    t3 = time.time()
    time0 = time0 + t3 - t2
        
    # ================== compute U ================== #
    matB1c = numpy.zeros((r+sc, c))
    matB1c[0:r, :] = matB11
    matB1c[r:r+sc, :] = matB21
    matB1c = numpy.linalg.pinv(matB1c)
    matB1c1 = matB1c[:, 0:r]
    matB1c2 = matB1c[:, r:r+sc]
    matU1 = numpy.dot(matB1c1, matB12)
    if parameters['sketch2'] == 'Gauss' or parameters['sketch2'] == 'Count':
        matU2 = numpy.dot(matB1c2, matB22)
    elif parameters['sketch2'] == 'Uniform' or parameters['sketch2'] == 'Leverage':
        matU2 = matB1c2 * matB22
    
    matU = matU1 + matU2
    matB1r = numpy.zeros((r, c+sr))
    matB1r[:, 0:c] = matB11
    matB1r[:, c:c+sr] = matB12
    matB1r = numpy.linalg.pinv(matB1r)
    matU = numpy.dot(matU, matB1r[c:c+sr, :])
    matU = matU + matB1r[0:c, :]
        
        
    # ================== compute SVD ================== #
    matUU, vecSU, matVU = numpy.linalg.svd(matU, full_matrices=False)
    matQC = numpy.dot(matQC, matUU[:, 0:k])
    vecSU = vecSU[0:k]
    matQR = numpy.dot(matVU[0:k, :], matQR.T)
        
    t4 = time.time()
    time1 = time1 + t4 - t3
    
    TimeCost = {'t1': time0,
               't2': time1}
    return matQC, vecSU, matQR, TimeCost




def sparseFastCURSVD(matA, parameters):
    '''
    matA: m-by-n csc_matrix or csr_matrix
    '''
    c = parameters['c']
    r = parameters['r']
    k = parameters['k']
    ratio = parameters['s']
    sc = c * ratio
    sr = r * ratio
    m, n = matA.shape
    
    t0 = time.time()
    # ================== compute C and R ================== #
    if parameters['sketch'] == 'Gauss':
        matPC = numpy.random.randn(n, c) / numpy.sqrt(c)
        matC = matA * matPC
        del matPC
        matPR = numpy.random.randn(r, m) / numpy.sqrt(r)
        matR = matPR * matA
        del matPR
    elif parameters['sketch'] == 'Count':
        matC = countsketch.countsketchSparse1(matA.T, c)
        matC = matC.T
        matR = countsketch.countsketchSparse1(matA, r)
    elif parameters['sketch'] == 'Uniform':
        idxC = numpy.random.choice(n, c, replace=False)
        idxR = numpy.random.choice(m, r, replace=False)
        matC = numpy.array(matA[:, idxC].todense())
        matR = numpy.array(matA[idxR, :].todense())
    elif parameters['sketch'] == 'Udaptive':
        idxC = adaptive.adaptiveSamplingSparse(matA, c)
        idxR = adaptive.adaptiveSamplingSparse(matA.T, r)
        matC = numpy.array(matA[:, idxC].todense())
        matR = numpy.array(matA[idxR, :].todense())
    t1 = time.time()
    time0 = t1 - t0
        
    
    matQC = numpy.linalg.qr(matC, mode='reduced')[0]
    matQR = numpy.linalg.qr(matR.T, mode='reduced')[0]
    
    t2 = time.time()
    time1 = t2 - t1
    
    
    # ================== compute U ================== #
    vecLevC = numpy.sum(matQC ** 2, axis=1)
    vecLevC = vecLevC / numpy.sum(vecLevC)
    vecLevR = numpy.sum(matQR ** 2, axis=1)
    vecLevR = vecLevR / numpy.sum(vecLevR)
    idxSC = numpy.random.choice(m, sc, replace=False, p=vecLevC)
    idxSR = numpy.random.choice(n, sr, replace=False, p=vecLevR)
    matB21 = matQC[idxSC, :]
    matB12 = matQR[idxSR, :].T
    matB22 = matA[idxSC, :]
    matB22 = matB22[:, idxSR]
        
    t3 = time.time()
    time0 = time0 + t3 - t2
        
    # ================== compute U ================== #
    matU = numpy.linalg.pinv(matB21) * matB22
    matU = numpy.dot(matU, numpy.linalg.pinv(matB12))
    matU = numpy.array(matU)
        
        
    # ================== compute SVD ================== #
    matUU, vecSU, matVU = numpy.linalg.svd(matU, full_matrices=False)
    matQC = numpy.dot(matQC, matUU[:, 0:k])
    vecSU = vecSU[0:k]
    matQR = numpy.dot(matVU[0:k, :], matQR.T)
        
    t4 = time.time()
    time1 = time1 + t4 - t3
    
    TimeCost = {'t1': time0,
               't2': time1}
    return matQC, vecSU, matQR, TimeCost
