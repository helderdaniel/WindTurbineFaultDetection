#Form dataset from CTBU files stored on Dropbox
#auxiliary I/O classes
#
#v0.1 jul 2019
#hdaniel@ualg.pt 
#

import requests, pickle 
import numpy as np
import hdf5storage as hdf
from hdlib.net.ChromeBrowser import ChromeBrowser
from os import sep as folderSep


class DatasetIO:
    @staticmethod
    def load(filename):
        return hdf.loadmat(filename)
    
    @staticmethod
    def saveRaw(filename, X, Y, xName, yName, xType, yType):
        hdf.savemat(filename, 
                {   xName: np.array(X, dtype=xType), 
                    yName: np.array(Y, dtype=yType) },
                do_compression=True, format='5')

    @staticmethod
    def saveTT( filename, 
                trainX, trainY, testX, testY, 
                trainXName, trainYName, testXName, testYName, 
                xType, yType):
        hdf.savemat(filename, 
                {   trainXName : np.array(trainX, dtype=xType), 
                    trainYName : np.array(trainY, dtype=yType), 
                    testXName  : np.array(testX,  dtype=xType), 
                    testYName  : np.array(testY,  dtype=yType) },
                do_compression=True, format='5')


class CTBUDataset:
    __channel6FixFN = '-Record0010-ICS645B-1' #Todo: Not the best way

    #
    #Init instance
    #
    def __init__(self, url, faults, loads, freqs, reps, chans, workFolder = './', fileExt = '.mat', filelistFN='filelist.pickle'):
        self.__url        = url
        self.__faults     = faults
        self.__loads      = loads
        self.__freqs      = freqs
        self.__reps       = reps
        self.__chans      = chans
        self.__workFolder = workFolder
        self.__fileExt    = fileExt
        self.__filelistFN = workFolder + folderSep + filelistFN
        self.__fileList   = []

    #
    # 'Private' util methods to travers Dropbox file list pages
    #
    def __getHRef(self, browser, linkText):
        element = browser.getElementByLinkText(linkText)
        if element is not None:
            href = element.get_attribute('href')
        else:
            raise Exception('href for link text "{}", NOT found in page'.format(linkText))
        return href


    def __getFaultsFolderRefs(self, browser, url, faultFolderNames):
        faultFolders = []
        
        browser.getPage(url)
        for f in faultFolderNames:
            faultFolders.append(self.__getHRef(browser, f))
        return faultFolders
        
    #
    # Get list of files that contains dataset specified on args
    #
    def getFileRefs(self):
        faultFolderNames = []
        self.__fileList = []

        #needed to parse javascript and get all js code
        #if NOT some may be missing
        FaultFolderBrowser = ChromeBrowser()   #(hideBrowser=False)
        
        for f in self.__faults:
            faultFolderNames.append('C{:02d}'.format(f))

        #get ref for each fault folder: 'C00' ... 'C35'
        faultFolderRefs = self.__getFaultsFolderRefs(FaultFolderBrowser, self.__url, faultFolderNames)
        
        i = 0
        for f in faultFolderRefs:

            #Get page for each fault folder
            FaultFolderBrowser.getPage(f)
            faultFolderName = faultFolderNames[i]
            i += 1

            for l in self.__loads:
                for f in self.__freqs:
                    FilesFolderBrowser = ChromeBrowser()   #(hideBrowser=False)
                    for r in self.__reps:
                        folderName = faultFolderName + 'L{:01d}S{:02d}R{:02d}'.format(l, f, r)
                        folderRef = self.__getHRef(FaultFolderBrowser, folderName)
                        FilesFolderBrowser.getPage(folderRef)

                        for c in self.__chans:
                            fileName = folderName
                            if c == 6:
                                fileName += self.__channel6FixFN
                            fileName += '_Ch{:03d}'.format(c)
                            fileName += self.__fileExt
                            print(fileName)
                            fileRef = self.__getHRef(FilesFolderBrowser, fileName)
                            self.__fileList.append((fileName, fileRef))
        return self.__fileList

    def fileRefs(self):
        return self.__fileList

    def saveFileRefs(self):
        with open(self.__filelistFN, 'wb') as fp: 
            pickle.dump(self.__fileList, fp)

    def loadFileRefs(self):
        try:
            with open(self.__filelistFN, 'rb') as fp: 
                self.__fileList = pickle.load(fp)
        except:
            raise Exception('File with refs NOT found at: ', self.__filelistFN)
        return self.__fileList

    #
    # Get files in list from Dropbox
    #
    def getFiles(self):
        numFiles = len(self.__fileList)
        for i in range(numFiles):
            url = self.__fileList[i][1]
            #?dl=0 -> dl=1, force download without ask to save
            url = url[:len(url)-1] + '1'  
            fn  = self.__workFolder + folderSep + self.__fileList[i][0]

            print('Downloading file {}/{}, from: {}'.format(i+1, numFiles, url))       
            response = requests.get(url)

            print('Storing as file: ', fn)
            with open(fn, 'wb') as fp:
                fp.write(response.content)

    #
    # Generate dataset
    #
    # Currently only supports one file for Load, Freq and repetition
    #
    def generateRawDataset(self, chanDataName, samples, windowWidth, skipSamples=0, onehot=False):
        if len(self.__fileList) <= 0:
            raise Exception('No data to generate data set in folder??')
        
        X = []
        Y = []
        numFiles    = len(self.__fileList)
        numFaults   = len(self.__faults)
        numChannels = len(self.__chans)


        for i in range(numFiles):
            fn  = self.__workFolder + folderSep + self.__fileList[i][0]
            print("loading file {}/{}: ".format(i+1, numFiles), fn)
            #read hdf5 files with partial data and extract data
            data = DatasetIO.load(fn)
            chidx = i % numChannels
            varChName = chanDataName + '{:03d}'.format(self.__chans[chidx])
            x0 = data[varChName]
                        
            #Reshape data according to requested
            st = skipSamples*windowWidth
            ed = st + samples*windowWidth
            if st > x0.shape[0] or ed > x0.shape[0]:
                raise Exception('Slice [{}:{}] outside variable length [0:{}]'.format(st, ed, x0.shape[0]))
            x0 = x0[st : ed]
            x0 = x0.reshape(samples, int(x0.shape[0]/samples))

            #assemble channel data contiguosly
            if chidx == 0:
                x = x0
            else:
                x = np.append(x, x0, axis=1)

            #Add all channels class data to dataset
            if chidx == numChannels-1:      
                X.append(x)

                faultClass = int(self.__fileList[i][0][1:3])
                if onehot:
                    #raises exception if does not exis
                    faultClassIdx = self.__faults.index(faultClass)
                    y = np.zeros((samples, numFaults))
                    y[:,faultClassIdx] = 1
                else:
                    y = np.ones(samples, 1) * faultClass             
                Y.append(y)
            
        X = np.concatenate(X)
        Y = np.concatenate(Y)

        return X, Y


    def saveRawDataset( self, fn, X, Y, 
                        featuresName='X', labelsName='Y', 
                        xType='float32', yType='uint8'):
        DatasetIO.saveRaw(fn, X, Y, featuresName, labelsName, xType, yType)
