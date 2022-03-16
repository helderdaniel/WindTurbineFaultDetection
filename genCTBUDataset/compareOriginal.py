import scipy.io, numpy as np
from os import sep as folderSep
from CTBUDataset import DatasetIO

folder2store = '/storage/OneDrive - Universidade do Algarve/Works/I&D/04-Projectos/00-Turbinas/Dados-turbinas/datactbu-original/generatedDatasets'
originalCTBUFN = folder2store + folderSep + '..' + folderSep + 'original-3chan' + folderSep + 'datactbu-ch3-5.mat'
datasetFN      = folder2store + folderSep + 'raw_3channels-w2000-r01.mat'


#Check if original CTBU and generated are equal
print('Comparing to: ', originalCTBUFN)
print(scipy.io.whosmat(originalCTBUFN))
print(scipy.io.whosmat(datasetFN))
d0 = DatasetIO.load(originalCTBUFN)
d1 = DatasetIO.load(datasetFN)
X0=d0['X']; X1=d1['X']
Y0=d0['Y']; Y1=d1['Y']
#Note: DOES NOT work if arrays have different shape
#t0=(X0==X1).all()
#t1=(Y0==Y1).all()
#print(t0, t1)
print(np.array_equal(X0,X1)) # test if same shape, same elements values
print(np.array_equal(Y0,Y1)) # test if same shape, same elements values
#print(np.allclose   (X0,X1)) # test if same shape, elements have close enough values