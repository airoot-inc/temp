import sys
sys.path.append("F:/Riku_Ka/AudioTestCode")
import shutil
import os
from pydub  import AudioSegment
import FHC_Audio
import librosa
import python_speech_features
import numpy as np

resDIR = "F:/Riku_Ka/VOICE data set for Depression-医者 only"
dstDirMFCCdelta012order = "F:/Riku_Ka/VOICE data set for Depression-医者 onlyーMFCC-012order"
dstDirMFCC012NPY = "F:/Riku_Ka/VOICE data set for Depression-医者 onlyーMFCC-012order-npy"

dirLIST = []

'''
if not os.path.isdir(dstDirMFCC):
    os.mkdir(dstDirMFCC)
if not os.path.isdir(dstDirMFCCdelta1st):
    os.mkdir(dstDirMFCCdelta1st)
if not os.path.isdir(dstDirMFCCdelta2nd):
    os.mkdir(dstDirMFCCdelta2nd)
if not os.path.isdir(dstDirMFCCdelta012order):
    os.mkdir(dstDirMFCCdelta012order)
'''



files = os.listdir(resDIR)
try:
    for f in files:
        tempFile = resDIR + '/' + f
        filename = f.split('.wav')[0]
        print(filename)
        resultMFCC =  FHC_Audio.getMFCC4SingleFile(tempFile, 13)
        resultMFCCdelta1st = librosa.feature.delta(resultMFCC, axis=0, order=1)
        resultMFCCdelta2nd = librosa.feature.delta(resultMFCC, axis=0, order=2)
        resultMFCCdelta012order = np.hstack((resultMFCC,resultMFCCdelta1st,resultMFCCdelta2nd))
        np.savetxt(dstDirMFCCdelta012order + '/' + filename + '_MFCC-012order.txt', resultMFCCdelta012order)
        np.save(dstDirMFCC012NPY + '/' + filename + '_MFCC-012order.npy', resultMFCCdelta012order)
        print('All of the features have been extracted!')
except librosa.util.exceptions.ParameterError:
    print('There is a error librosa.util.exceptions.ParameterError for file:')
    print(tempFile)

'''
listDementia=[]
count = 1


resFILE = "H:/Num of MFCC lines for Depression/Depression-result.txt"
with open(resFILE, 'r') as of:
    allFileList = of.read()
    for dirPATH, dirNAME, files in os.walk(resDIR):
        #print(dirPATH, '*****', dirNAME, '******', files)
        for f in files:
            tempFILE = dirPATH +'/'+f
            if os.path.isfile(tempFILE):
                if f[0:25] in allFileList:
                    dstFILE = dstDIR + '/' + f
                    shutil.copy(tempFILE,dstFILE)
                    print('We found a file!')
                    print(tempFILE)
                    count += 1
    print(count)
    print('-------------')
'''





