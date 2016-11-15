'''
Created on Nov 7, 2016

@author: km_dh
'''
import matplotlib.pyplot as plt
import numpy as np
from GradientDescentOld import GradientDescent as gd


class LinearModel(object):
    '''
    classdocs
    '''
    
    def res(self, mode, X, Y=np.array([0]), eta=0, numIter=0, stochastic=False, lossFn=None, regConst=0,
            traceW=False, model=None):
        # usage is one of the two following:
        #   model = linearModel('train', X, Y, eta, numIter, stochastic, lossFn, regularizer, regConst)
        #   Y     = linearModel('predict', model, X)
        #
        # here, the arguments to 'train' are as follows:
        #   X            NxD matrix of training points
        #   Y            N-vector of labels (must be -1 or +1)
        #   eta          step size for gradient descent
        #   numIter      number of iterations of gradient descent to run
        #   stochastic   set to 1 if you want stochastic GD rather than vanilla GD
        #   lossFn       one of: 'squared', 'perceptron', 'log', 'hinge', or 'exp'
        #   regularizer  one of: 'l2', 'l1'
        #   regConst     the regularization constant
        #
        # we do gradient descent on the function:
        #   regConst * regularizer(w) + sum_n lossFn(w, X(n,:), Y(n))
        
        # Check to make sure no parameters are None
        if eta is None or numIter is None or stochastic is None or regConst is None or X is None or Y is None:
            print 'training requires eight arguments'
            return 0, 0
        
        mode = mode.lower()
        
        if mode == 'name':
            return 'Linear Model'
        
        if mode == 'train':
            # Get the sizes of X and Y into NX,DX,NY,DY
            NX, DX = X.shape
            NY = Y.shape[0]
            print Y
            print NX, NY
            
            # Check that there are the same number of points in X and Y
            # if NX ~= NY,
            if NX != NY:
                print 'there must be the same number of data points in X and Y'
                return 0, 0
            
            # Check that Y only has one column
            if len(Y.shape) != 1:
                print 'Y must have only one column'
                return 0, 0
            
            # Check that eta is greater than zero
            if eta <= 0:
                print 'eta must be > 0'
                return 0, 0
            
            # Check that there is at least one iteration
            if numIter < 1:
                print 'must run for at least one iteration'
                return 0, 0
            
            # Check the loss function is recognized
            if lossFn not in ["squared", "perceptron", "log", "hinge", "exp"]:
                print 'loss function must be one of squared, perceptron, log, hinge or exp'
                return 0, 0
            
            output = {}
            if traceW is False:
                output['weights'], output['bias'] = self.trainLinearModel(X, Y, eta, numIter, stochastic, lossFn,
                                                                          regConst, traceW=False)
                return output
            else:
                output['weights'], output['bias'], wtrace = self.trainLinearModel(X, Y, eta, numIter, stochastic,
                                                                                  lossFn, regConst, traceW=True)
            return output, wtrace
        
        if mode == 'predict':
            # check arguments
            # if there is no model or no X
            if model is None or X is None:
                print 'prediction requires two arguments: the model, and X'
                return [0]
            
            # Check the model has weights
            # if ~isfield(model,'weights'),
            if model.get('weights') is None:
                print 'model does not appear to be a linear model'
                return [0]
            
            # Check the model has bias
            # if ~isfield(model,'bias'),
            if model.get('bias') is None:
                print 'model does not appear to be a linear model'
            
            # set up output
            # Get the number of examples
            # Make a zeros array Nx1
            N, D = X.shape
            output = np.zeros((N, 1))
            # compute predictions
            # For each example
            for n in xrange(N):
                output[n] = self.predictLinearModel(model['weights'], model['bias'], X[n, :])
            
            return output
            
            # otherwise
            # error('unknown linear model mode: need "train" or "predict"');
    
    def trainLinearModel(self, X, Y, eta, numIter, stochastic, lossFn, regConst, traceW=False):
        # we encode the bias by adding a constant 1s feature to X.  remember: don't
        # regularize the bias!
        
        # Get N and D from X
        N, D = X.shape
        # Add a column of 1's to the front of X to hold the bias
        X = np.concatenate((np.ones((N, 1)), X), axis=1)
        
        # initialize weights to zero's (this is not an NN)
        w0 = np.zeros((1, D + 1));
        
        # run gradient descent with and without trace
        gradDesc = gd()
        print traceW
        if traceW is False:
            ww = gradDesc.GD(self.computeGradient(stochastic, lossFn, X, Y, w0, X.shape[0]), eta, numIter,
                             w0, X, Y, stochastic, traceW)
        else:
            ww, wtrace = gradDesc.GD(self.computeGradient(stochastic, lossFn, X, Y, w0, X.shape[0]), eta,
                                     numIter, w0, X, Y, stochastic, traceW)
        
        # now, peel weights and bias off results
        return ww[:, 1:], ww[:, 0]
    
    def computeGradient(self, stochastic, lossFn, regConst, X, Y, w, numPoints=0):
        # we do gradient descent on the function:
        #   regConst * l2(w) + sum_n lossFn(w, X(n,:), Y(n))
        #
        # settings contains all the information we need about the regularization
        # constant, the l2 and the loss function.  if
        # settings.stochastic, then X will be a D-vector and Y will be a scalar.
        # if ~settings.stochastic, then X will be NxD matrix and Y will be an
        # N-vector.  we may assume Y is always +1 or -1 for classification.
        #
        # if we're in stochastic optimization, remember to appropriately divide
        # the regularizer!  settings.numPoints stores the total number of training
        # points.
        
        # Get N and D from X
        print X
        print X.shape
        N, D = X.shape
        
        # Initialize the gradients to zero
        g = np.zeros((1, D))
        
        # sum up the losses of individual points
        # For each example 1 to N
        
        for n in range(N):
            # Get the truth y = Y(n);
            y = Y[n]  # the truth
            # Get the x value & prediction x = X(n,:);
            x = X[n, :]
            t = sum(np.dot(x, np.transpose(w)))
            t = 1 if t > 0 else -1
            # threshold the prediction at 0 for -1 and +1
            if lossFn == 'squared':
                if y * t < 0:  # calc grad
                    g += (2 * (y - t) * -x)
            
            if lossFn == 'perceptron' or lossFn == 'hinge':
                if y * t < 0:
                    g *= (-y * x)
            
            if lossFn == 'log':
                g += ((1 / (1 + np.e(y * t))) * x * -y)
            
            if lossFn == 'exponential':
                g += np.e(-y * t) * x * -y
        
        # if settings.stochastic == 1,
        if stochastic:
            for i in range(D - 1):
                # For the X's without bias for i=2:D,
                g[:, i + 1] = g[:, i + 1] + (regConst * 2 * w[i + 1] / numPoints)  # this is NOT extra credit!
        
        else:
            # For the X's without bias for i=2:D,
            print g
            print w
            print regConst
            g[:, :] += regConst * 2 * w  # this is NOT extra credit!
        return g
    
    def predictLinearModel(self, w, b, X):
        if sum((w * X) + b) < 0:
            return -1
        return 1
