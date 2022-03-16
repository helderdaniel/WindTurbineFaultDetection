#Form dataset from CTBU files stored on Dropbox
#Config dataset
#
#v0.1 jul 2019
#hdaniel@ualg.pt 
#

'''
base URL:
https://www.dropbox.com/sh/yyiyfphdfokmoue/AADqfkTgo6Pa9BUgntidAXs0a/temp

folder ".../temp" has:

folders:     C00-35
sub-folders: C00L1S00R01
C00-35
L1-3
S00-30-40-50
R00-10

files:       C00L1S00R01_Ch000.mat
Ch000-005 and 007
and for channel 6:  C00L1S00R07-Record0010-ICS645B-1_Ch006.mat

For original 7 classes CTBU dataset use:

#Set data to get
faults = [0,1,3,8,10,15,30] #0-35
loads  = [1]     #1 - 3
freqs  = [50]    #0 30 40 50
reps   = [1]     #1 - 10
chans  = [3,4,5] #0 - 7

#Set generation options
split       = 1 #0.0 - 1.0
skipSamples = 0
samples     = 1000  #If possible with specified windowWidth, if NOT is is reduced
windowWidth = 1024
shuffle     = False #best use function shuffle
'''


import sys    
from os import sep as folderSep
from CTBUDataset import CTBUDataset


#Run in Colab or local
useColab = False
if len(sys.argv)>1:
    if sys.argv[1].lower() == 'colab':
        useColab = True
 

if useColab:
    folder2store = 'gdrive/My Drive/Colab Notebooks/00data/generatedDatasets'
else:
    folder2store = '/storage/OneDrive - Universidade do Algarve/Works/I&D/04-Projectos/00-Turbinas/Dados-turbinas/datactbu-original/generatedDatasets'


baseUrl   = 'https://www.dropbox.com/sh/yyiyfphdfokmoue/AADqfkTgo6Pa9BUgntidAXs0a/temp'
datasetFN = folder2store + folderSep + 'raw_3channels-w2000-r01.mat'

chanDataName='Data_Ch_'
featuresName='X'
labelsName='Y'
getRefs  = False
getFiles = False
sortReps = False

#Set data to get
faults = [0,1,3,8,10,15,30] #0-35
loads  = [1]     #1 - 3
freqs  = [50]    #0 30 40 50
reps   = [1]     #1 - 10
chans  = [3,4,5] #0 - 7

#Set generation options
split       = 1 #0.0 - 1.0
skipSamples = 0
samples     = 1000  #If possible with specified windowWidth, if NOT it is reduced
windowWidth = 2000
shuffle     = False #best use function shuffle (NOT implemented yet)


#Generate dataset example
if __name__ == "__main__":

    data = CTBUDataset(baseUrl, faults, loads, freqs, reps, chans, workFolder = folder2store, filelistFN='filelist-r01.pickle')

    if getRefs:
        print('Getting file refs list from Dropbox ...')
        data.getFileRefs()
        fileRefs = data.fileRefs()
        data.saveFileRefs()
    else:
        #Load file list (sorted when created)
        print('Loading file refs list from workfolder ...')
        fileRefs = data.loadFileRefs()
        #So, NOT NEEDED: Make sure it is sorted by filename, first element in tupple
        #fileList = sorted(fileList, key=lambda x: x[0])
    
    #Sort references by Repetition
    if sortReps:
        fileRefs.sort(key = lambda x:  x[0].split('R')[1].split('_')[0])
    for i in fileRefs: print(i)
    
    if getFiles:
        print('Getting files from Dropbox in to workfolder: ', folder2store)
        data.getFiles()

    print('Generating dataset ...')
    X, Y = data.generateRawDataset(chanDataName, samples, windowWidth, skipSamples, True)

    print('Saving dataset as ', datasetFN)
    data.saveRawDataset(datasetFN, X, Y, featuresName, labelsName)
