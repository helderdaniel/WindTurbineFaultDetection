import numpy as np
import matplotlib.pyplot as plt
import time, psutil, sys, gc

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
                 'labels_training': np.array(trainYs, dtype='int8'), 
                 'features_test': np.array(testXs, dtype='float32'), 
                 'labels_test': np.array(testYs, dtype='int8') },
                do_compression=True, format='5')

    
#Print number of samples by classes
def samplesByClass(data, printClasses=False):
    #Unique classes
    if printClasses:
        print(np.unique(data, return_counts=True, axis=0)[0])
    #Number of samples of each unique class
    print(np.unique(data, return_counts=True, axis=0)[1])


def createDataset(data, faults, trainSize=0.8, featuresName='X', labelsName='Y'):
  
    print ("start: ", psutil.virtual_memory())

    #Get data
    X = data[featuresName]
    Y = data[labelsName]
    numClasses = len(faults)
    print ("get X Y: ", psutil.virtual_memory())

    #if NOT onehot encoded
    #make sure Y has cols dim 1 if NOT onehot encoded 
    if len(Y.shape) == 1:
        Y = Y.reshape(Y.shape[0], 1)  

    print ("Y reshape: ", psutil.virtual_memory())

    '''
    #Normalize 
    print('Normalize')
    #X = minmax(X)
    X = quantileTransform(X) 
    print ("Normalize: ", psutil.virtual_memory())
    '''
    
'''
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

print ("train: ", psutil.virtual_memory())

#Not needed anymore free memory
X = 0
Y = 0 
del X
del Y

print ("free X Y: ", psutil.virtual_memory())

#Join list of arrays in just one
trainXs = np.concatenate(trainXs)
trainYs = np.concatenate(trainYs)
testXs  = np.concatenate(testXs)
testYs  = np.concatenate(testYs)

print ("concatenate: ", psutil.virtual_memory())

trainXs = np.copy(trainXs)
trainYs = np.copy(trainYs)
testXs = np.copy(testXs)
testYs = np.copy(testYs)

print ("copy: ", psutil.virtual_memory())

#Not needed anymore free memory
gc.collect()
time.sleep(10)

print ("gc: ", psutil.virtual_memory())

#Faults are ordered by classes
#this shuffle faults order
print('shuffle classes')
p = np.random.permutation(trainXs.shape[0])
print('shuffle classes')
trainXs = trainXs[p]
print('shuffle classes')
trainYs = trainYs[p]

p = np.random.permutation(testXs.shape[0])
print('shuffle classes')
testXs = testXs[p]
print('shuffle classes')
testYs = testYs[p]

print ("shuffle: ", psutil.virtual_memory())

return trainXs, trainYs, testXs, testYs
'''

#Main
#if __name__ == "main":

#C00_C01_C03_C08_C10_C15_C30_S50_L1_Ch_01_to_06

if useColab:
    storagepath='gdrive/My Drive/Colab Notebooks/classifier/data/'
else:
    storagepath='/home/hdaniel/Downloads/'

#define individual channels data
loadFNs = []
loadFNs.append(storagepath + 'raw_50000_003.mat')
loadFNs.append(storagepath + 'raw_50000_004.mat')
loadFNs.append(storagepath + 'raw_50000_005.mat')

featuresName = 'features'
labelsName   = 'labels'
numFiles = len(loadFNs)

#output dataset
saveFN = storagepath + 'raw_3channels.mat'
save = True

#Config dataset generation
faults = [0, 1, 3, 8, 10, 15, 30]
numClasses   = len(faults)
split        = 0.8


#%whos

outputData = []
for i in range(numFiles):
    filename = loadFNs[i]
    print("Loading: ", filename, psutil.virtual_memory())
    inData = loadData(filename)
    if i == 0:
        fshape = inData[featuresName].shape
        lshape = inData[labelsName].shape
        print("Data  shape", fshape, lshape)
        
        #p = np.random.permutation(---HOW to do it--- samples)
    else:
        if fshape != inData[featuresName].shape or lshape != inData[labelsName].shape:
            raise Exception('channel data shape expected {} and {}, but got {} and {}'.format(
                  fshape, lshape, inData[featuresName].shape, inData[labelsName].shape))
    #trainXs, trainYs, testXs, testYs = createDataset(inData, faults, split, featuresName, labelsName)


  
print("dataset created", psutil.virtual_memory())
samplesByClass(inData[labelsName], printClasses=False)
'''
trainXs, trainYs, testXs, testYs = shuffleData(data, numClasses, split, 1, featuresName, labelsName)
samplesByClass(trainYs, printClasses=False)
samplesByClass(testYs, printClasses=False)
'''
plt.plot(inData[featuresName][23,:])
#plt.plot(trainXs[23,:])
plt.show()
