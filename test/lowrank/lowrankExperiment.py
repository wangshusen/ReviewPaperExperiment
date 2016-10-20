import numpy
import time
import sys
import scipy.io
import scipy.sparse.linalg
sys.path.append('../../main/lowrank/')
import sparseCUR
import sparseApproxSVD

def lowrankExperiment(matA, param):
    ratio = param['ratio']
    lenRatio = len(ratio)
    rk = param['k']
    Repeat = param['repeat']
    
    normFroA = numpy.sum(sum(matA.multiply(matA)).todense())
    vecS = scipy.sparse.linalg.svds(matA, k=rk+1, return_singular_vectors=False)
    normSpeAres = min(vecS)
    normFroAres = numpy.sqrt(normFroA - numpy.linalg.norm(vecS[1:rk+1]) ** 2)
    
    matErrorFro = numpy.zeros((Repeat, lenRatio))
    matErrorSpe = numpy.zeros((Repeat, lenRatio))
    vecTime0 = numpy.zeros(lenRatio)
    vecTime1 = numpy.zeros(lenRatio)
    
    for l in range(lenRatio):
        c = int(numpy.ceil(rk * ratio[l]))
        print('c = ' + str(c))
        r = c
        parameters = {'c': c,
                      'r': r,
                      'k': rk,
                      's': param['s'],
                      'sketch': param['sketch']}
        
        t0 = numpy.zeros(Repeat)
        t1 = numpy.zeros(Repeat)

        for re in range(Repeat):
            if param['model'] == 'OptCX':
                matU, vecS, matV, TimeCost = sparseApproxSVD.sparseOptCXSVD(matA, parameters)
            elif param['model'] == 'FastCX':
                matU, vecS, matV, TimeCost = sparseApproxSVD.sparseFastCXSVD(matA, parameters)
            elif param['model'] == 'OptCUR':
                matU, vecS, matV, TimeCost = sparseApproxSVD.sparseOptCURSVD(matA, parameters)
            elif param['model'] == 'FastCUR':
                matU, vecS, matV, TimeCost = sparseApproxSVD.sparseFastCURSVD(matA, parameters)
                
            t0[re] = TimeCost['t1']
            t1[re] = TimeCost['t2']
            
            matB = matU * vecS.reshape(1, len(vecS))
            matB = numpy.dot(matB, matV)
            matB = matB - matA
            normFroRes = numpy.linalg.norm(matB, 'fro')
            normSpeRes = numpy.linalg.norm(matB)
            errFro = normFroRes / normFroAres
            errSpe = normSpeRes / normSpeAres
            matErrorFro[re, l] = errFro
            matErrorSpe[re, l] = errSpe
            print('Frobenius Error = ' + str(errFro) + '    Spectral Error = ' + str(errSpe))
            
        vecTime0[l] = numpy.mean(t0)
        vecTime1[l] = numpy.mean(t1)
        print('Time cost is ' + str(vecTime0[l] + vecTime1[l]))
    
    return matErrorFro, matErrorSpe, vecTime0, vecTime1


def runExperiment(matA):
    param = {'ratio': [1.5, 2, 2.5, 3, 3.5, 4, 5, 6, 7, 8, 9, 10],
            'k': 100,
             's': 10,
            'repeat': 20}
    outputFileName = 'result_enron_approxSVD.mat'


    # ============ #
    param['sketch'] = 'Gauss'
    param['model'] = 'OptCX'
    resultName = param['sketch'] + param['model']
    print('')
    print('Sketch is ' + param['sketch'] + ',   Model is ' + param['model'])
    ErrorFro, ErrorSpe, Time0, Time1 = lowrankExperiment(matA, param)
    resultDict = {'ErrorFro' + resultName: ErrorFro, 
             'ErrorSpe' + resultName: ErrorSpe,
             'Time0' + resultName: Time0,
             'Time1' + resultName: Time1}
    scipy.io.savemat(outputFileName, resultDict)



    # ============ #
    param['sketch'] = 'Count'
    param['model'] = 'OptCX'
    resultName = param['sketch'] + param['model']
    print('')
    print('Sketch is ' + param['sketch'] + ',   Model is ' + param['model'])
    ErrorFro, ErrorSpe, Time0, Time1 = lowrankExperiment(matA, param)
    resultDict['ErrorFro' + resultName] = ErrorFro
    resultDict['ErrorSpe' + resultName] = ErrorSpe
    resultDict['Time0' + resultName] = Time0
    resultDict['Time1' + resultName] = Time1
    scipy.io.savemat(outputFileName, resultDict)


    # ============ #
    param['sketch'] = 'Uniform'
    param['model'] = 'OptCX'
    resultName = param['sketch'] + param['model']
    print('')
    print('Sketch is ' + param['sketch'] + ',   Model is ' + param['model'])
    ErrorFro, ErrorSpe, Time0, Time1 = lowrankExperiment(matA, param)
    resultDict['ErrorFro' + resultName] = ErrorFro
    resultDict['ErrorSpe' + resultName] = ErrorSpe
    resultDict['Time0' + resultName] = Time0
    resultDict['Time1' + resultName] = Time1
    scipy.io.savemat(outputFileName, resultDict)
    
    

    # ============ #
    param['sketch'] = 'Gauss'
    param['model'] = 'FastCX'
    resultName = param['sketch'] + param['model']
    print('')
    print('Sketch is ' + param['sketch'] + ',   Model is ' + param['model'])
    ErrorFro, ErrorSpe, Time0, Time1 = lowrankExperiment(matA, param)
    resultDict = {'ErrorFro' + resultName: ErrorFro, 
             'ErrorSpe' + resultName: ErrorSpe,
             'Time0' + resultName: Time0,
             'Time1' + resultName: Time1}
    scipy.io.savemat(outputFileName, resultDict)



    # ============ #
    param['sketch'] = 'Count'
    param['model'] = 'FastCX'
    resultName = param['sketch'] + param['model']
    print('')
    print('Sketch is ' + param['sketch'] + ',   Model is ' + param['model'])
    ErrorFro, ErrorSpe, Time0, Time1 = lowrankExperiment(matA, param)
    resultDict['ErrorFro' + resultName] = ErrorFro
    resultDict['ErrorSpe' + resultName] = ErrorSpe
    resultDict['Time0' + resultName] = Time0
    resultDict['Time1' + resultName] = Time1
    scipy.io.savemat(outputFileName, resultDict)


    # ============ #
    param['sketch'] = 'Uniform'
    param['model'] = 'FastCX'
    resultName = param['sketch'] + param['model']
    print('')
    print('Sketch is ' + param['sketch'] + ',   Model is ' + param['model'])
    ErrorFro, ErrorSpe, Time0, Time1 = lowrankExperiment(matA, param)
    resultDict['ErrorFro' + resultName] = ErrorFro
    resultDict['ErrorSpe' + resultName] = ErrorSpe
    resultDict['Time0' + resultName] = Time0
    resultDict['Time1' + resultName] = Time1
    scipy.io.savemat(outputFileName, resultDict)
    
    

    # ============ #
    param['sketch'] = 'Gauss'
    param['model'] = 'OptCUR'
    resultName = param['sketch'] + param['model']
    print('')
    print('Sketch is ' + param['sketch'] + ',   Model is ' + param['model'])
    ErrorFro, ErrorSpe, Time0, Time1 = lowrankExperiment(matA, param)
    resultDict['ErrorFro' + resultName] = ErrorFro
    resultDict['ErrorSpe' + resultName] = ErrorSpe
    resultDict['Time0' + resultName] = Time0
    resultDict['Time1' + resultName] = Time1
    scipy.io.savemat(outputFileName, resultDict)
    

    # ============ #
    param['sketch'] = 'Count'
    param['model'] = 'OptCUR'
    resultName = param['sketch'] + param['model']
    print('')
    print('Sketch is ' + param['sketch'] + ',   Model is ' + param['model'])
    ErrorFro, ErrorSpe, Time0, Time1 = lowrankExperiment(matA, param)
    resultDict['ErrorFro' + resultName] = ErrorFro
    resultDict['ErrorSpe' + resultName] = ErrorSpe
    resultDict['Time0' + resultName] = Time0
    resultDict['Time1' + resultName] = Time1
    scipy.io.savemat(outputFileName, resultDict)


    # ============ #
    param['sketch'] = 'Uniform'
    param['model'] = 'OptCUR'
    resultName = param['sketch'] + param['model']
    print('')
    print('Sketch is ' + param['sketch'] + ',   Model is ' + param['model'])
    ErrorFro, ErrorSpe, Time0, Time1 = lowrankExperiment(matA, param)
    resultDict['ErrorFro' + resultName] = ErrorFro
    resultDict['ErrorSpe' + resultName] = ErrorSpe
    resultDict['Time0' + resultName] = Time0
    resultDict['Time1' + resultName] = Time1
    scipy.io.savemat(outputFileName, resultDict)
    
    
    # ============ #
    param['sketch'] = 'Gauss'
    param['model'] = 'FastCUR'
    s = 10
    param['s'] = s
    resultName = param['sketch'] + param['model'] + str(s)
    print('')
    print('Sketch is ' + param['sketch'] + ',   Model is ' + param['model'])
    ErrorFro, ErrorSpe, Time0, Time1 = lowrankExperiment(matA, param)
    resultDict['ErrorFro' + resultName] = ErrorFro
    resultDict['ErrorSpe' + resultName] = ErrorSpe
    resultDict['Time0' + resultName] = Time0
    resultDict['Time1' + resultName] = Time1
    scipy.io.savemat(outputFileName, resultDict)
    

    # ============ #
    param['sketch'] = 'Count'
    param['model'] = 'FastCUR'
    s = 10
    param['s'] = s
    resultName = param['sketch'] + param['model'] + str(s)
    print('')
    print('Sketch is ' + param['sketch'] + ',   Model is ' + param['model'])
    ErrorFro, ErrorSpe, Time0, Time1 = lowrankExperiment(matA, param)
    resultDict['ErrorFro' + resultName] = ErrorFro
    resultDict['ErrorSpe' + resultName] = ErrorSpe
    resultDict['Time0' + resultName] = Time0
    resultDict['Time1' + resultName] = Time1
    scipy.io.savemat(outputFileName, resultDict)


    # ============ #
    param['sketch'] = 'Uniform'
    param['model'] = 'FastCUR'
    s = 10
    param['s'] = s
    resultName = param['sketch'] + param['model'] + str(s)
    print('')
    print('Sketch is ' + param['sketch'] + ',   Model is ' + param['model'])
    ErrorFro, ErrorSpe, Time0, Time1 = lowrankExperiment(matA, param)
    resultDict['ErrorFro' + resultName] = ErrorFro
    resultDict['ErrorSpe' + resultName] = ErrorSpe
    resultDict['Time0' + resultName] = Time0
    resultDict['Time1' + resultName] = Time1
    scipy.io.savemat(outputFileName, resultDict)
    
    
    # ============ #
    param['sketch'] = 'Gauss'
    param['model'] = 'FastCUR'
    s = 5
    param['s'] = s
    resultName = param['sketch'] + param['model'] + str(s)
    print('')
    print('Sketch is ' + param['sketch'] + ',   Model is ' + param['model'])
    ErrorFro, ErrorSpe, Time0, Time1 = lowrankExperiment(matA, param)
    resultDict['ErrorFro' + resultName] = ErrorFro
    resultDict['ErrorSpe' + resultName] = ErrorSpe
    resultDict['Time0' + resultName] = Time0
    resultDict['Time1' + resultName] = Time1
    scipy.io.savemat(outputFileName, resultDict)
    

    # ============ #
    param['sketch'] = 'Count'
    param['model'] = 'FastCUR'
    s = 5
    param['s'] = s
    resultName = param['sketch'] + param['model'] + str(s)
    print('')
    print('Sketch is ' + param['sketch'] + ',   Model is ' + param['model'])
    ErrorFro, ErrorSpe, Time0, Time1 = lowrankExperiment(matA, param)
    resultDict['ErrorFro' + resultName] = ErrorFro
    resultDict['ErrorSpe' + resultName] = ErrorSpe
    resultDict['Time0' + resultName] = Time0
    resultDict['Time1' + resultName] = Time1
    scipy.io.savemat(outputFileName, resultDict)


    # ============ #
    param['sketch'] = 'Uniform'
    param['model'] = 'FastCUR'
    s = 5
    param['s'] = s
    resultName = param['sketch'] + param['model'] + str(s)
    print('')
    print('Sketch is ' + param['sketch'] + ',   Model is ' + param['model'])
    ErrorFro, ErrorSpe, Time0, Time1 = lowrankExperiment(matA, param)
    resultDict['ErrorFro' + resultName] = ErrorFro
    resultDict['ErrorSpe' + resultName] = ErrorSpe
    resultDict['Time0' + resultName] = Time0
    resultDict['Time1' + resultName] = Time1
    scipy.io.savemat(outputFileName, resultDict)
    
    
    return resultDict
    
    
        
