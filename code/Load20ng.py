'''
Created on Oct 30, 2016

@author: km_dh
'''
from sklearn.datasets.twenty_newsgroups import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
import Tkinter as tk
import matplotlib as mlp


class Load20ng(object):
    '''
    This class is to load and display the 20 news groups data
    it needs to be separate from the MNIST because MatLabPlot goes weird.
    The methods are at the level of the class so they are called on the class.
    '''

def dispBanner():
    print'%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%'
    print'********************************************************************'
    print'** 20 NEWSGROUPS EXPERIMENTS'
    print'********************************************************************\n'
    print 'First we load the data \n'
    cats = ['alt.atheism', 'talk.religion.misc','comp.graphics', 'sci.space']
    news_train = fetch_20newsgroups(subset='train', categories=cats)
    news_test = fetch_20newsgroups(subset='test', categories=cats)
    for i in range(len(news_train.target_names)):
        print "Target number " + str(i) + " is " + news_train.target_names[i]
    
    #Get the training data
    vectorizer = TfidfVectorizer()
    trX = vectorizer.fit_transform(news_train.data) 
    trX_images = news_train.data
    trY = news_train.target
    trY = np.reshape(trY, (trY.shape[0],1))
    
    #Get the test data
    deX = vectorizer.transform(news_test.data)
    deY = news_test.target
    deY = np.reshape(deY, (deY.shape[0],1))
    print trX.shape, trY.shape, deX.shape, deY.shape   
    raw_input('\nPress enter to continue...')
    return trX_images, trX, trY, deX,deY  
    
def dispImages(title,trX_images,trY):
    #This can't be used with the controller. If you want to see the files, let me know.
    print 'and display some examples from it....'
    print '\nIf the image does not pop up, go and find it...'
    print '\nYou might need to find a microphone type icon on your dock...'
    print '\nClose the image to continue...'

    print len(trX_images), trY.shape
    mlp.use('TkAgg')
    for i in range(4):
        ans = "Document Label: " + str(trY[200*i]) + "\n\n"
        label = tk.Label(None, text=ans + trX_images[200*i], 
                         justify='left', font=('Times', '12'),fg='black', 
                         takefocus=True)
        label.pack()
        label.mainloop() 