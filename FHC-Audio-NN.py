import os
import numpy as np
import librosa
import wave
from random import choice
import random
import shutil
from pydub  import AudioSegment
from keras import Sequential
from keras.layers import Dense, Activation
from keras.optimizers import adam
from keras.models import load_model
import time
from keras.models import model_from_json


# config all of the parameters:
rootDIR ="F:/Riku_Ka/finalTest"
depressionFileDIR = rootDIR + '/' + "depressionVoiceFiles"
healthyFileDIR = rootDIR + '/' +  "healthyVoiceFiles"
#The following DIRs can be made automatically.
labeledMFCC4depressionFileDIR = rootDIR + '/' +  "1MFCC4depression"
labeledMFCC4healthyFileDIR = rootDIR + '/' +  "0MFCC4healthy"
testFileDIR = rootDIR + '/' +  "testDIR"
trainFileDIR = rootDIR + '/' +  "trainDIR"
modelFileDIR = rootDIR + '/' +  "modelDIR" 
dirLIST = [labeledMFCC4healthyFileDIR, labeledMFCC4depressionFileDIR, testFileDIR, trainFileDIR, modelFileDIR]
#The following are const.
trainTestRate = 0.3 # The percentage of total number of files for test.
numOfEpoch = 2
dementionOf0OrderMFCC = 13
#The following are name of files that must be given before.
resultFile = rootDIR + '/' + "result.txt"
jsonModelFILE = modelFileDIR + '/' + "NN Modelfor 0Health-1Dementia with 39MFCC with npy form by 333 samples.json"
weightModelFILE = modelFileDIR + '/' + "NN Modelfor 0Health-1Dementia with 39MFCC with npy form by 333 samples.h5"
train_npy_FILE = rootDIR + '/' + "train_npy_FILE.npy"


for d in dirLIST:
    for root, dirs, files in os.walk(d, topdown = False):
        for name in files:
            os.remove(os.path.join(root, name))
        for name in dirs:
            os.rmdir(os.path.join(root, name))
os.remove(resultFile)
os.remove(train_npy_FILE)

for d in dirLIST:
    if not os.path.isdir(d):
        os.mkdir(d)

# Extracting MFCC and attatching labels for depression files.
files = os.listdir(depressionFileDIR)
try:
    for f in files:
        tempFile = depressionFileDIR + '/' + f
        filename = f.split('.wav')[0]
        print(filename)
        resultMFCC = np.zeros((1,dementionOf0OrderMFCC))
        myWave = wave.open(tempFile, 'rb')
        sr = myWave.getframerate()
        myWave.close()
        y, sr= librosa.load(tempFile, sr=sr)
        resultMFCC = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=dementionOf0OrderMFCC).T
        n, m = np.shape(resultMFCC)
        print("n and m are {0} and {1}.".format(n, m))
        labelLIST = np.ones((n,1))
        resultMFCCdelta1st = librosa.feature.delta(resultMFCC, axis=0, order=1)
        resultMFCCdelta2nd = librosa.feature.delta(resultMFCC, axis=0, order=2)
        resultMFCCdelta012order = np.hstack((labelLIST, resultMFCC,resultMFCCdelta1st,resultMFCCdelta2nd))       
        print(resultMFCCdelta012order.shape)
        # save MFCC in txt form, slowly, not recommended 
        #np.savetxt(dstDirMFCCdelta012order + '/' + filename + '_MFCC-012order.txt', resultMFCCdelta012order)
        # save MFCC in txt npy, fast, recommended 
        np.save(labeledMFCC4depressionFileDIR + '/1_' + filename + '_MFCC-012order.npy', resultMFCCdelta012order)
        print('All of the features have been extracted!')
except librosa.util.exceptions.ParameterError:
    print('There is a error librosa.util.exceptions.ParameterError for file:')
    print(tempFile)



# Extracting MFCC and attatching labels for healthy files.
files = os.listdir(healthyFileDIR)
try:
    for f in files:
        tempFile = healthyFileDIR + '/' + f
        filename = f.split('.wav')[0]
        print(filename)
        resultMFCC = np.zeros((1,dementionOf0OrderMFCC))
        myWave = wave.open(tempFile, 'rb')
        sr = myWave.getframerate()
        myWave.close()
        y, sr= librosa.load(tempFile, sr=sr)
        resultMFCC = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=dementionOf0OrderMFCC).T
        n, m = np.shape(resultMFCC)
        labelLIST = np.zeros((n,1))
        resultMFCCdelta1st = librosa.feature.delta(resultMFCC, axis=0, order=1)
        resultMFCCdelta2nd = librosa.feature.delta(resultMFCC, axis=0, order=2)
        resultMFCCdelta012order = np.hstack((labelLIST, resultMFCC,resultMFCCdelta1st,resultMFCCdelta2nd))       
        print("n and m are {0} and {1}.".format(n, m))       
        # save MFCC in txt form, slowly, not recommended 
        #np.savetxt(dstDirMFCCdelta012order + '/' + filename + '_MFCC-012order.txt', resultMFCCdelta012order)
        # save MFCC in txt npy, fast, recommended 
        np.save(labeledMFCC4healthyFileDIR + '/0_' + filename + '_MFCC-012order.npy', resultMFCCdelta012order)
        print('All of the features have been extracted!')
except librosa.util.exceptions.ParameterError:
    print('There is a error librosa.util.exceptions.ParameterError for file:')
    print(tempFile)

# choose depression files for test and train respectively
originalDepressionFiles = os.listdir(labeledMFCC4depressionFileDIR)
testDepressionLIST = random.sample(originalDepressionFiles, int(len(originalDepressionFiles)*trainTestRate))

trainDepressionLIST = originalDepressionFiles
for x in testDepressionLIST:
    if x in trainDepressionLIST:
        trainDepressionLIST.remove(x)

# Copy test files of depression
filesNotFoundLIST = []
for f in testDepressionLIST:
    if any(k in f for k in testDepressionLIST):
        print("The file {} is matched.".format(f))
        resFILE = labeledMFCC4depressionFileDIR + '/' + f
        dstFILE = testFileDIR + '/' + f
        try:
            shutil.copy(resFILE, dstFILE)
            print('The file {} has been copied.'.format(f))
        except PermissionError:
            print("PermissionError")
        except ValueError:
            print("There is a ValueError.")
        except FileNotFoundError:
            print("The file {} was not found.".format(f))
            filesNotFoundLIST.append(f)
print("Files not found for test: {}.".format(filesNotFoundLIST))


# Copy train files of depression
filesNotFoundLIST = []
for f in trainDepressionLIST:
    if any(k in f for k in trainDepressionLIST):
        print("The file {} is matched.".format(f))
        resFILE = labeledMFCC4depressionFileDIR + '/' + f
        dstFILE = trainFileDIR + '/' + f
        try:
            shutil.copy(resFILE, dstFILE)
            print('The file {} has been copied.'.format(f))
        except PermissionError:
            print("PermissionError")
        except ValueError:
            print("There is a ValueError.")
        except FileNotFoundError:
            print("The file {} was not found.".format(f))
            filesNotFoundLIST.append(f)
print("Files not found for train: {}.".format(filesNotFoundLIST))

# choose healthy files for test and train respectively
originalHealthyFiles = os.listdir(labeledMFCC4healthyFileDIR)
testHealthyLIST = random.sample(originalHealthyFiles, int(len(originalHealthyFiles)*trainTestRate))

trainHealthyLIST = originalHealthyFiles
for x in testHealthyLIST:
    if x in trainHealthyLIST:
        trainHealthyLIST.remove(x)

# Copy test files of healthy
filesNotFoundLIST = []
for f in testHealthyLIST:
    if any(k in f for k in testHealthyLIST):
        print("The file {} is matched.".format(f))
        resFILE = labeledMFCC4healthyFileDIR + '/' + f
        dstFILE = testFileDIR + '/' + f
        try:
            shutil.copy(resFILE, dstFILE)
            print('The file {} has been copied.'.format(f))
        except PermissionError:
            print("PermissionError")
        except ValueError:
            print("There is a ValueError.")
        except FileNotFoundError:
            print("The file {} was not found.".format(f))
            filesNotFoundLIST.append(f)
print("Files not found for test: {}.".format(filesNotFoundLIST))


# Copy train files of healthy
filesNotFoundLIST = []
for f in trainHealthyLIST:
    if any(k in f for k in trainHealthyLIST):
        print("The file {} is matched.".format(f))
        resFILE = labeledMFCC4healthyFileDIR + '/' + f
        dstFILE = trainFileDIR + '/' + f
        try:
            shutil.copy(resFILE, dstFILE)
            print('The file {} has been copied.'.format(f))
        except PermissionError:
            print("PermissionError")
        except ValueError:
            print("There is a ValueError.")
        except FileNotFoundError:
            print("The file {} was not found.".format(f))
            filesNotFoundLIST.append(f)
print("Files not found for train: {}.".format(filesNotFoundLIST))



# Combining files for training.
trainFiles = os.listdir(trainFileDIR)
trainMatrix = np.ones((1,40))
for f in trainFiles:
    tempNPY = np.load(trainFileDIR + '/' + f)
    print(np.shape(tempNPY))
    trainMatrix = np.vstack((trainMatrix, tempNPY))
np.save(train_npy_FILE, trainMatrix)




# Training and testing Neural Networks
print("Loading data....")
startTime = time.time()

print("The shape of train dataset is {}".format(trainMatrix.shape))
trainDATA = trainMatrix[:,1:]
trainLABLE = trainMatrix[:,0]

print("Start training!")
print("{} secondes have passed. ".format(str(time.time() - startTime)))


np.random.seed(10)

myModel = Sequential()

myModel.add(Dense(units=64,input_dim=39, activation='relu'))
myModel.add(Dense(units=8, activation='relu'))
myModel.add(Dense(1, activation='sigmoid'))

myModel.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

myModel.fit(x=trainDATA, y=trainLABLE, epochs=numOfEpoch, batch_size=64)

# Save the trained model
myModel_json = myModel.to_json()
with open(jsonModelFILE, 'w') as json_file:
    json_file.write(myModel_json)
myModel.save_weights(weightModelFILE)
print("The trained model has been saved.")


print("Start testing!")
print("Time point 3 is " + str(time.time() - startTime))

# Load the trained model
# Load the model
with open(jsonModelFILE) as modelFile:
    model_json = modelFile.read()

myModel = model_from_json(model_json)
myModel.load_weights(weightModelFILE)
myModel.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])



#produce test data set.
NPYfiles = os.listdir(testFileDIR)
for f in NPYfiles:
    tempFile = testFileDIR + '/' + f
    tempMatrix = np.load(tempFile)
    print("The shape of {0} is {1}.".format(f, tempMatrix.shape))
    testData = tempMatrix[:,1:]
    testLabels = tempMatrix[:,0]
    
    scores = myModel.evaluate(x=testData, y=testLabels)
    print(scores)
    print("The accuracy for file {0} is {1}".format(f,scores[1]*100))

with open(resultFile, 'w+') as of:
    NPYfiles = os.listdir(testFileDIR)
    resultLIST = []
    for f in NPYfiles:
        try:
            tempFile = testFileDIR + '/' + f
            tempMatrix = np.load(tempFile)
            #print("The shape of {0} is {1}.".format(f, tempMatrix.shape))
            testData = tempMatrix[:,1:]
            yLabels = tempMatrix[:,0]
    
            scores = myModel.evaluate(x=testData, y=yLabels)
            #print(scores)
            #print("The accuracy for file {0} is {1}".format(f,scores[1]*100))
            resultLIST.append(f + '\t' + str(scores[1]*100) + '\n')
        except OSError:
            print("Failed to interpret file {} as a pickle.".format(f))
        except IOError:
            print("Can not open file {} correctly.".format(f))
        
    of.writelines(resultLIST)

