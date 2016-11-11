'''
Created on Aug 28, 2016
@author: km_dh
'''

from sklearn.linear_model import SGDClassifier
# import Load20ng
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


def run_comps(paramType, param, X, Y, X_test, Y_test, title, xlbl, figno):
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
            learn = SGDClassifier(loss='log', penalty='none', n_iter=param[i])
        if 'learnRate' in paramType:
            learn = SGDClassifier(loss='log', penalty='none', eta0=param[i])
        if 'regularizer' in paramType:
            learn = SGDClassifier(loss='log', penalty='l2', alpha=param[i])
        if 'loss' in paramType:
            # Let the param function be replaced, because MatPlotLib expects a number
            loss_functions = ["log", "hinge", "squared_loss", "perceptron"]
            learn = SGDClassifier(loss=loss_functions[i], penalty='l2', alpha=0.01, n_iter=10)
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
        print 'accTr = ', float(val_sum) / (len(Y) * 1.0)
        accTr.append(float(val_sum) / (len(Y) * 1.0))

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
    # This loads the data set to change uncomment the one you want.
    # Remember to also uncomment the appropriate import.
    # trX_images, trX, trY, deX, deY = Load20ng.dispBanner()
    trX_images, trX, trY, deX, deY = LoadMNIST.dispBanner()

    # If you want to see the MNIST data, uncomment these
    # You cannot see the 20 news groups data, because of MatLibPlot. So I included examples
    # as files.
    #    LoadMNIST.dispImages('MNIST',trX_images,trY)
    # # raw_input('Press enter to continue...')

    print '\nNow we vary the number of iterations for the Gradient Descent'
    print 'with log loss and see how it affects accuracy...'
    print '\nIf the picture does not pop up, go find it...'
    print '\nClose the image to continue...'
    epochs = [1, 5, 10, 20, 35, 50, 100]
    res = run_comps('iterations', epochs, trX, trY, deX, deY,
                    "Figure 1: Iterations vs. accuracy (MNIST)",
                    "iterations", "../SGD_iterations.png")
    # raw_input('Press enter to continue...')

    print '\nNow we vary the learning rate for the Gradient Descent'
    print 'with log loss and see how it affects accuracy...'
    print '\nIf the picture does not pop up, go find it...'
    print '\nClose the image to continue...'
    etas = [0.1, 0.3, 0.6, 0.9]
    res = run_comps('learnRate', etas, trX, trY, deX, deY,
                    "Figure 2: Learning rate vs. accuracy (MNIST)",
                    "eta", "../SGD_etas.png")
    # raw_input('Press enter to continue...')

    print "\nRunning test on regularizer alpha \n"
    regularizer = [0.1, 0.25, 0.5, 0.75, 1]
    res = run_comps('regularizer', regularizer, trX, trY, deX, deY,
                    "Figure 3: Regularizer Alpha vs. accuracy (MNIST)",
                    "Regularizer Alpha", "../SGD_regularizer.png")
    # raw_input('Press enter to continue...')

    print "\nRunning test on Loss Functions\n"
    print "Loss functions are in this order: ", ["log", "hinge", "squared_loss", "perceptron"], "\n"
    loss = [1, 2, 3, 4]
    res = run_comps('loss', loss, trX, trY, deX, deY,
                    "Figure 5: Loss Function vs. accuracy (MNIST)",
                    "Loss Function", "../SGD_loss_rate.png")
    # raw_input('Press enter to continue...')

    print 'Done'
