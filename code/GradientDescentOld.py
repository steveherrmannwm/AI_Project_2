'''
Created on Nov 7, 2016

@author: km_dh
'''
import matplotlib.pyplot as plt
import numpy as np
import inspect

TRACE = True


class GradientDescent(object):
    def GD(self, grad=None, eta=None, numIter=None, w0=None, X=None, Y=None, stochastic=False):
        # usage: [w,wTrace] = GD(grad, eta, numIter, w0, X, Y, stochastic) perform
        # gradient descent on function 'grad' on data (X,Y) with step size 'eta'
        # for exactly 'numIter' steps.  we initialize weights with w0 or.  returns
        # the final weight vector 'w', as well as a trace of all values that w
        # took during iterations.  If 'stochastic' is on, then we process one
        # example at a time.
        #
        # grad should have the form 'g = grad(X,Y,w)'.  here, g should be a vector
        # of the same size as w (the gradient).
        
        # Check you have all the parameters you need.
        if None in locals().values():
            return 0, 0
        # Get the size of the X matrix
        n, d = X.shape
        
        # only create trace if it's asked for
        wtrace = {}
        # initialize w by copying old value
        w = w0
        
        # begin gradient steps
        
        # for iterations 1 -> numIter
        for i in range(numIter):
            # if stochastic:
            # be sure to process examples in random order!
            if stochastic:
                perm = np.random.permutation(X.shape[0])
                X = X[perm, :]
                Y = Y[perm]
                # for each example in the permutated training seta
                # calculate the gradient using the gradient grad (from above)
                # new_weight = old_weight - eta * gradient
                # end for
                for row, label in zip(X, Y):
                    g = grad(row, label, w)
                    w = w - eta * g
            else:
                # else
                # compute on all the training points
                # calculate the gradient using the gradient gradient grad (from above) and all examples
                # make sure gradient is the right shape
                # new_weights = old_weights - eta * gradient
                g = grad(X, Y, w)
                # if g.shape[0] != w.shape[0]:
                #     return 0, 0
                w = w - eta * g
            
            # end else
            
            # for safety change any weights that ended up NAN into 0
            w[np.where(w is np.NAN)] = 0
            # generate the trace if we need it
            wtrace[i] = w
        
        # end for iterations
        return w, wtrace
        # end def


def plot_trace(points, title, funct, wtrace):
    f_plot = np.zeros((points.shape[1], 1))
    y_plot = np.zeros((len(wtrace), 2))
    i = 0
    for key in wtrace:
        y_plot[i][0] = wtrace[key]
        y_plot[i][1] = funct(val = wtrace[key])
        i += 1
    
    plt.title(title)
    plt.plot(points[0, :], f_plot, color='red', label='function')
    plt.plot(y_plot[0, :], y_plot[0, :], color='black', marker='o', label='descent')
    plt.legend()
    # plt.ylabel('accuracy')
    # plt.xlabel(xlbl)
    # plt.savefig(figno)
    plt.show()
