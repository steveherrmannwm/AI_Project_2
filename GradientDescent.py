'''
Created on Nov 7, 2016

@author: km_dh
'''
import matplotlib.pyplot as plt
import numpy as np

class GradientDescent(object):
    '''
    classdocs
    '''


    def __init__(self, params):
        '''
        Constructor
        '''
        
    def GD(self, grad, eta, numIter, w0, X, Y, stochastic):
      # usage: [w,wTrace] = GD(grad, eta, numIter, w0, X, Y, stochastic) perform
      # gradient descent on function 'grad' on data (X,Y) with step size 'eta'
      # for exactly 'numIter' steps.  we initialize weights with w0 or.  returns
      # the final weight vector 'w', as well as a trace of all values that w
      # took during iterations.  If 'stochastic' is on, then we process one
      # example at a time.
      #
      # grad should have the form 'g = grad(X,Y,w)'.  here, g should be a vector
      # of the same size as w (the gradient).
      
      #Check you have all the parameters you need.
      
      #Get the size of the X matrix
    
      # only create trace if it's asked for
    
      
      # initialize w by copying old value
      w = w0;
      
      # begin gradient steps
    
      #for iterations 1 -> numIter
    
        #if stochastic:
          #be sure to process examples in random order!
          
          #for each example in the permutated training set
              #calculate the gradient using the gradient grad (from above)
              #new_weight = old_weight - eta * gradient
          #end for
        #else
          # compute on all the training points
    
          #calculate the gradient using the gradient gradient grad (from above) and all examples
          #make sure gradient is the right shape
          #new_weights = old_weights - eta * gradient
        #end else
    
        # for safety change any weights that ended up NAN into 0
        
        
        # generate the trace if we need it
        #wTrace{iter} = w;
        
        #end for iterations
    
    #end def

def plot_trace(xlbl, title, funct, points):
    
    f_plot = np.zeros((points.shape[1],1))
    for i,elt in enumerate(points[0,:]):
        #print "point", i, " is ", elt
        f_plot[i] = funct(elt)
    plt.title(title)
    plt.plot(points[0,:], f_plot, color='red', label='function')
    plt.plot(points[0,:], points[1,:], color='black', marker='o', label='descent')
    plt.legend()
    #plt.ylabel('accuracy')
    plt.xlabel(xlbl)
    #plt.savefig(figno)
    plt.show()