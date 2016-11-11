'''
Created on Aug 28, 2016
@author: km_dh
'''
from sklearn.neural_network import MLPClassifier
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
        if 'nodes' in paramType:
            learn = MLPClassifier(hidden_layer_sizes=(param[i], ), activation='tanh', solver='sgd', alpha=0)
        if 'learnRate' in paramType:
            rates = ['constant', 'adaptive']
            learn = MLPClassifier(hidden_layer_sizes=(5, ), activation='tanh', solver='sgd',
                                  learning_rate=rates[i], alpha=0)
        if 'regularizer' in paramType:
            learn = MLPClassifier(hidden_layer_sizes=(5, ), activation='tanh', solver='sgd', alpha=param[i])
        print '\n', xlbl, param[i]
        testRun = tt.TrainTest_SL(learn, X, Y, X_test, Y_test)
        test_info = testRun.run_tt()
        accTe.append(test_info['acc'])
        model_pred = learn.predict(X)
        sizeY = model_pred.shape
        if len(sizeY) < 2:
            model_pred = np.reshape(model_pred, (sizeY[0], 1))
        val = [Y == model_pred]
        val_sum = sum(sum(val))
        print 'accTr = ', float(val_sum)/(len(Y)*1.0)
        accTr.append(float(val_sum)/(len(Y)*1.0))

    mlp.title(title)
    print param
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
 
    print 'Vary the number of nodes'
    print 'Close the image to continue...'
    nodes = [2, 5, 10, 25, 50]
    res = run_comps('nodes', nodes, trX, trY, deX, deY,
                    "Figure 1: Nodes vs. accuracy (MNIST)",
                    "Nodes","../MLP_nodes.png")
    raw_input('Press enter to continue...')

    print 'Vary the learning rate'
    print 'Close the image to continue...'
    etas = [1, 2]
    res = run_comps('learnRate', etas, trX, trY, deX, deY,
                    "Figure 2: Learning rate vs. accuracy (MNIST)",
                    "Learning Rate","../MLP_etas.png")
    raw_input('Press enter to continue...')

    print 'Vary the regularizer alpha'
    regularizer = [0.1, 0.25, 0.5, 0.75, 1]
    res = run_comps('regularizer', regularizer, trX, trY, deX, deY,
                    "Figure 3: Regularizer Alpha vs. accuracy (MNIST)",
                    "Regularizer Alpha","../MLP_regularizer.png")
     
    print 'Done'
