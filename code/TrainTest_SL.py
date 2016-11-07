'''
Created on Aug 28, 2016

@author: km_dh
'''
import numpy as np
import time

class TrainTest_SL(object):
    '''
    This class runs the algorithms in their training and testing modes.
    It keeps track of time to complete and accuracy.
    '''


    def __init__(self, learn, trainX=np.array([[]]), trainY=np.array([]), testX=np.array([[]]), testY=np.array([])):
        self.learn = learn
        self.trainX = trainX
        self.trainY = trainY
        self.testX = testX
        self.testY = testY
        
    def run_tt(self):
        output = {}
        print 'Training...'
        print self.trainX.shape
        t0 = time.time()
        #Be sure you understand why this is the training line
        model = self.learn.fit(self.trainX, self.trainY.ravel())
        output['model'] = model
        t1 = time.time()
        output['trainTime'] = t1-t0
        print 'Testing...'
        t0 = time.time()
        print self.testX.shape
        #Be sure you understand why this is the testing line
        Y = self.learn.predict(self.testX)
        t1 = time.time()
        output['testTime'] = t1 - t0
        Y = np.array(Y)
        sizeY = Y.shape
        if len(sizeY) < 2:
            Y = np.reshape(Y, (sizeY[0],1))
        if(len(Y) == len(self.testY)):
            overlp = [Y == self.testY]
            overlp_sum = sum(sum(overlp))
            output['acc'] = overlp_sum/(len(Y)*1.0)
            return output
        print 'cannot determine accuracy'
        return 0
            
    
       
        