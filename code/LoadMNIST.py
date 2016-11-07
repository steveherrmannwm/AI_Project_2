'''
Created on Oct 30, 2016

@author: km_dh
'''
import MNISTcontrol as ms
import os.path
import numpy as np
import matplotlib.pyplot as mlp

class LoadMNIST(object):
    '''
    Keeping the MNIST data set load and display separate from 20 news groups
    it needs to be separate from the MNIST because MatLabPlot goes weird.
    The methods are at the level of the class so they are called on the class.
    '''
        
def dispBanner():
    print '************************************************************************'
    print '**  MNIST DIGITS EXPERIMENTS'
    print '************************************************************************'
        
    print 'First we load the data '
    dirname = os.path.dirname(__file__)
    dirname = dirname.replace("code","MNIST")
    dataset = ms.MNISTcontrol(dirname)
    trX_images, trY = dataset.load_mnist('training')
    #make it a binary problem
    trX_images = trX_images[np.logical_or(np.equal(trY[:,0],8),np.equal(trY[:,0],3))]
    trY = trY[np.logical_or(np.equal(trY[:,0],8),np.equal(trY[:,0],3))]
    print 'classes', np.unique(trY)
    # we need x to be 1d
    sizeX = trX_images.shape
    if len(sizeX) > 2:
        newXdim = 0
        for i in range(sizeX[1]):
            newXdim += len(trX_images[0][i])
        trX = np.reshape(trX_images, (sizeX[0], newXdim))
        
    #read in test data
    deX, deY = dataset.load_mnist('test')
    #make it a binary problem
    deX = deX[np.logical_or(np.equal(deY[:,0],8), np.equal(deY[:,0],3))]
    deY = deY[np.logical_or(np.equal(deY[:,0],8), np.equal(deY[:,0],3))]
    # we need x to be 1d
    sizeX = deX.shape
    if len(sizeX) > 2:
        newXdim = 0
        for i in range(sizeX[1]):
            newXdim += len(deX[0][i])
        deX = np.reshape(deX, (sizeX[0], newXdim))

    print trX.shape, trY.shape, deX.shape, deY.shape   
    raw_input('\nPress enter to continue...')
    return trX_images, trX, trY, deX, deY

def dispImages(title,trX_images,trY):         
    print 'and display some examples from it....'
    print '\nIf the image does not pop up, go and find it...'
    print '\nClose the image to continue...'

    fig = mlp.figure()
    for i in range(8):
        a = fig.add_subplot(2,4,i+1)
        mlp.imshow(trX_images[i*200], cmap=mlp.cm.get_cmap(mlp.gray()))
        label = 'Image Label: ' + str(trY[i*200])
        a.set_title(label)
    mlp.savefig('../figure1.png')
    mlp.show()
    