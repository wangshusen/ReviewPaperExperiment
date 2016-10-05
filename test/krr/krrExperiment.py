import numpy
import time
import scipy.io
import sys
sys.path.append('../../main/krr/')
import crossValidation as cv
import trainPredict as tp

def krrExperiment(dataName, inputFileName, parameters, sketchSizes):
    # repeat the whole cross-validation + train-predict procedure for "numRepeat1" times
    numRepeat1 = 10
    # with fixed optimal sigma and gamma, repeat trainPredict for "numRepeat2" times
    numRepeat2 = 10
    
    # output file name
    if parameters['method'] != 'RandFeature':
        outputFileName = dataName + '_' + parameters['method'] + '_' + parameters['sketch'] + '.mat'
    else:
        outputFileName = dataName + '_' + parameters['method'] + '.mat'

    # load data
    dataDict = scipy.io.loadmat('../../resource/data/' + inputFileName +'.mat')
    matXtrain =  dataDict['Xtrain']
    matXtest =  dataDict['Xtest']
    vecYtrain =  dataDict['ytrain']
    vecYtest =  dataDict['ytest']
    del dataDict
    
    # number of different sketch sizes
    numS = len(sketchSizes)
    
    # the Mean Squared Error of test
    MSEs = numpy.zeros((numS, numRepeat1 * numRepeat2))
    # the average elapsed time of Cross-Validation and Train-Predict
    TimeCR = numpy.zeros(numS)
    TimeTP = numpy.zeros(numS)

    for l in range(numS):
        s = sketchSizes[l]
        for i in range(numRepeat1):
            t0 = time.time()
            sigmaOpt, gammaOpt, mseTmp = cv.crossValid(matXtrain, vecYtrain, s, parameters)
            t1 = time.time()
            mse = 0
            for j in range(numRepeat2):
                mse =  tp.trainPredict(matXtrain, vecYtrain, matXtest, vecYtest, s, sigmaOpt, gammaOpt, parameters)
                MSEs[l, i * numRepeat2 + j] = mse
            t2 = time.time()

        mseMean = numpy.mean(MSEs[l, :])
        TimeCR[l] = (t1 - t0) / numRepeat1
        TimeTP[l] = (t2 - t1) / numRepeat1 / numRepeat2
        print('s = ' + str(s) + ', MSE = ' + str(mseMean))
        print('Time_CR = ' + str(TimeCR[l]) + ',  Time_TP = ' + str(TimeTP[l]))

        mathDict = {'MSEs': MSEs, 'sketchSizes': sketchSizes, 'TimeCR': TimeCR, 'TimeTP': TimeTP}
        scipy.io.savemat(outputFileName, mathDict)
