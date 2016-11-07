'''
Created on Aug 28, 2016
@author: km_dh
'''
#from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import SGDClassifier
#import Load20ng
import LoadMNIST
import numpy as np
import matplotlib.pyplot as mlp
import TrainTest_SL as tt

class Controller(object):
    '''
    This class creates an infrastructure to run comparisons using the 
    MNIST data set and the 20 news groups data set with the Scikit-learn
    algorithms.
    '''
    pass

def run_comps(paramType,param,X,Y,X_test,Y_test,title,xlbl,figno):
    '''
    This class runs the comparisons between models. Your job is to
    figure out how to call the appropriate classifier. 
    Note: the parameter that varies will be set to param[i]
    I have given you an example of how I use the paramType variable
    to choose between classifier calls
    '''
    accTr = [] 
    accTe = []
    for i in range(len(param)):
        if 'iterations' in paramType:
            learn = #TODO figure out which classifier call to use
        if 'learnRate' in paramType:
            learn = #TODO figure out which classifier call to use 
        print '\n', xlbl, param[i]
        testRun = tt.TrainTest_SL(learn, X, Y, X_test, Y_test)
        test_info = testRun.run_tt()
        accTe.append(test_info['acc'])
        model_pred = learn.predict(X)
        sizeY = model_pred.shape
        if len(sizeY) < 2:
            model_pred = np.reshape(model_pred, (sizeY[0],1))
        val = [Y == model_pred]
        val_sum = sum(sum(val))
        print 'accTr = ', float(val_sum)/(len(Y)*1.0)
        accTr.append(float(val_sum)/(len(Y)*1.0))

    mlp.title(title)
    mlp.plot(param, accTr, color='blue', marker='x', label='training accuracy')
    mlp.plot(param, accTe, color='black', marker='o', label='test accuracy')
    mlp.legend()
    mlp.ylabel('accuracy')
    mlp.xlabel(xlbl)
    mlp.savefig(figno)
    mlp.show()
    return accTe
 
   
if __name__ == '__main__':
          
    #This loads the data set to change uncomment the one you want.
    #Remember to also uncomment the appropriate import.
    #trX_images, trX, trY, deX, deY = Load20ng.dispBanner()
    trX_images, trX, trY, deX, deY = LoadMNIST.dispBanner()
        
    #If you want to see the MNIST data, uncomment these
    #You cannot see the 20 news groups data, because of MatLibPlot. So I included examples
    #as files.
#    LoadMNIST.dispImages('MNIST',trX_images,trY)
    #raw_input('Press enter to continue...')
 
    print '\nNow we vary the number of iterations for the Gradient Descent'
    print 'with log loss and see how it affects accuracy...'
    print '\nIf the picture does not pop up, go find it...'
    print '\nClose the image to continue...'
    epochs = [1,2,3,10,50,100]
    res = run_comps('iterations', epochs, trX, trY, deX, deY,
                    "Figure 2: Iterations vs. accuracy (MNIST)",
                    "iterations","../figure4.png")
    raw_input('Press enter to continue...')
     
    print '\nNow we vary the learning rate for the Gradient Descent'
    print 'with log loss and see how it affects accuracy...'
    print '\nIf the picture does not pop up, go find it...'
    print '\nClose the image to continue...'
    etas = [0.1,0.3,0.6,0.9]
    res = run_comps('learnRate', etas, trX, trY, deX, deY,
                    "Figure 2: Learning rate vs. accuracy (MNIST)",
                    "eta","../figure5.png")
    raw_input('Press enter to continue...')
     
    print 'Done' 

    
