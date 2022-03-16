import numpy as np
import matplotlib.pyplot as plt
import psutil
import gc
import time

useColab = False	
if useColab:
    #!pip3 install hdf5storage
    
    from google.colab import drive
    drive.mount('/content/gdrive')
import hdf5storage as hdf


def loadData(filename):
     #Get data
    return hdf.loadmat(filename) 


def saveData(filename, trainXs, trainYs, testXs, testYs):
    hdf.savemat(filename, 
            {'features_training': np.array(trainXs, dtype='float32'),
             'labels_training':   np.array(trainYs, dtype='int8'),
             'features_test':     np.array(testXs,  dtype='float32'),
             'labels_test':       np.array(testYs,  dtype='int8') },                 
             do_compression=True, format='5')


#Print number of samples by classes
def samplesByClass(data, printClasses=False):
    #Unique classes
    if printClasses:
        print(np.unique(data, return_counts=True, axis=0)[0])
    
    #Number of samples of each unique class
    print(np.unique(data, return_counts=True, axis=0)[1])
    

def minmax(X):
    #normalize min-max
    #https://swaathi.com/2017/04/29/normalizing-data/
    #[min >= 0, max <=1]
    #minimum = X.min()
    #maximum = X.max()
    #X = (X - minimum) / (maximum - minimum)
    
    #[0, 1]
    X = np.process.minmax_scale(X)
    return X

  
def quantileTransform(X):
    #Normalize Quantile transform
    from sklearn.preprocessing.data import QuantileTransformer
    qTransform = QuantileTransformer(output_distribution='uniform')
    qTransform.fit(X)
    X = qTransform.transform(X)
    return X


def shuffleData(data, numClasses, trainSize=0.8, underSample=1, 
                featuresName='X', labelsName='Y', xType='float32', yType='uint8'):  
    #Get data
    #X = np.array(data[featuresName][:, ::underSample], dtype=xType)
    #Y = np.array(data[labelsName], dtype=yType)
    
    X = data[featuresName][:, ::underSample]
    Y = data[labelsName]

    #Not needed anymore free memory
    data = 0   #del data may free only reference but leave data

    #if NOT onehot encoded
    #make sure Y has cols dim 1 if NOT onehot encoded 
    if len(Y.shape) == 1:
        Y = Y.reshape(Y.shape[0], 1)  

    #Normalize
    print('Normalising ...')
    #X = minmax(X)
    X = quantileTransform(X)
    
    #Separate datasets equally by fault classes
    sinalLength= int(X.shape[1])
    samples    = int(X.shape[0] / numClasses)
    splitPoint = int(samples*trainSize)
    #print(samples, splitPoint)
    
    trainXs = []
    trainYs = []
    testXs  = []
    testYs  = []

    for i in range(numClasses):
        print (i)

        #slice fault
        st = i*samples
        sp = st + splitPoint
        end = (i+1)*samples
        
        #shuffle in place each fault data before slice train/set
        p = np.random.permutation(samples)
        X[st:end, :] = X[st+p, :]
        Y[st:end, :] = Y[st+p, :]
        
        #print(trainXs.shape, trainYs.shape, testXs.shape, testYs.shape)
        #print(X[st:sp, :].shape, Y[st:sp, :].shape, X[sp:end, :].shape, Y[sp:end, :].shape)
            
        trainXs.append(X[st:sp, :])
        trainYs.append(Y[st:sp, :])
        testXs.append(X[sp:end, :])
        testYs.append(Y[sp:end, :])

    #Not needed anymore free memory
    X = []
    Y = []
    
    #Join list of arrays in just one
    trainXs = np.concatenate(trainXs)
    trainYs = np.concatenate(trainYs)
    testXs  = np.concatenate(testXs)
    testYs  = np.concatenate(testYs)
    
    #Faults are ordered by classes
    #this shuffle faults order
    p = np.random.permutation(trainXs.shape[0])
    trainXs = trainXs[p]
    trainYs = trainYs[p]
    
    p = np.random.permutation(testXs.shape[0])
    testXs = testXs[p]
    testYs = testYs[p]

    return trainXs, trainYs, testXs, testYs


#Main
if __name__ == "__main__":
    if useColab:
        storagepath='gdrive/My Drive/Colab Notebooks/classifier/data/'
    else:
        storagepath='/home/hdaniel/Downloads/'

    #loadFN = storagepath + 'datactbu.mat'
    #saveFN = storagepath + 'datactbuset2.mat'
    loadFN = storagepath + 'raw_50000_003.mat'
    saveFN = storagepath + 'raw_50000_003_shuffled-even.mat'

    numClasses   = 36
    split        = 0.9
    featuresName = 'features'
    labelsName   = 'labels'
    #featuresName = 'X'
    #labelsName   = 'Y'
    save         = True
    
    print("Loading data: ", psutil.virtual_memory())
    data = loadData(loadFN)

    samplesByClass(data[labelsName], printClasses=False)
    trainXs, trainYs, testXs, testYs = shuffleData(data, numClasses, split, 1, featuresName, labelsName)
    samplesByClass(trainYs, printClasses=False)
    samplesByClass(testYs, printClasses=False)

    plt.plot(trainXs[23,:])
    plt.show()

    print("Saving shuffled data", psutil.virtual_memory())
    if save:
        saveData(saveFN, trainXs, trainYs, testXs, testYs)