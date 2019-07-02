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
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.utils.data as Data
from torch.autograd import Variable
import matplotlib.pyplot as plt
import numpy as np
import os
import torch.nn.functional as F
from prettytable import PrettyTable
import time


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
BATCH_SIZE = 200
LEARNING_RATE = 0.001
#The following are name of files that must be given before.
resultFile = rootDIR + '/' + "result.txt"
modelFILE = modelFileDIR + '/' + "1D-CNN Modelfor 0Health-1Dementia with 39MFCC of npy form by 333 samples.pth"
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


#print the seperating line
def print_line(char,string):
    print(char*33,string,char*32)

timePoint1 = time.time()

# define the Conv. neural network
class myCNN1D(nn.Module):

    def num_flat_features(self,x):
        size=x.size()[1:] # all dimensions except the batch dimension
        num_features=1
        for s in size:
            num_features*=s
        return num_features

    def __init__(self):
        super(myCNN1D, self).__init__()
        self.layer1_conv = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=2, kernel_size=5, stride=1, padding=0, dilation=1, groups=1,
                      bias=True),
            nn.ReLU(),
            # nn.MaxPool1d(2)
        )
        self.layer2_conv = nn.Sequential(
            nn.Conv1d(in_channels=2, out_channels=3, kernel_size=6, stride=1, padding=0, dilation=1, groups=1,
                      bias=True),
            nn.ReLU()
        )
        self.layer3_fc1 = nn.Linear(10, 100)
        self.layer4_output = nn.Linear(100, 1)

    def forward(self, x):
        out = self.layer1_conv(x)
        out = self.layer2_conv(out)
        out = out.view(-1, self.num_flat_features(out))
        out = self.layer3_fc1(out)
        out = self.layer4_output(out)
        return out




# build a CNN
myConv = myCNN1D()

# Define Loss and Optimizer
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(myConv.parameters(), lr=LEARNING_RATE)


# prepare the training dataset
trainDS = np.load(train_npy_FILE)
#print("The shape of trainDS is {}".format(trainDS.shape))
trainDS = trainDS[:,np.newaxis]
print("The shape of trainDS is {}".format(trainDS.shape))

x_train = trainDS[:,:,1:]
y_train = trainDS[:,:,0]
y_train = torch.from_numpy(y_train).float()
x_train = torch.from_numpy(x_train).float()

trainDataSet = Data.TensorDataset(x_train, y_train)

trainLoader = Data.DataLoader(
    dataset=trainDataSet,  # torch TensorDataset format
    batch_size=BATCH_SIZE,  # mini batch size
    shuffle=True,  # shuffle the data
)

timePoint2 = time.time()
print("{} seconds have passed!".format(timePoint2 - timePoint1))

# Train the Model
print_line('*','Training Begin')
for epoch in range(numOfEpoch):
    for i, (x_train, y_train) in enumerate(trainLoader):
        x_train = Variable(x_train)
        y_train = Variable(y_train)

        # Forward + Backward + Optimize
        optimizer.zero_grad()
        outputs = myConv(x_train)

        loss = criterion(outputs, y_train)
        loss.backward()
        optimizer.step()

    print('Epoch [{}/{}] Loss: {:.8f}'.format(epoch + 1, numOfEpoch, loss.data))

torch.save(myConv, modelFILE)
reload_model = torch.load(modelFILE)

timePoint3 = time.time()
print("{} seconds have passed!".format(timePoint3 - timePoint2))

# Test the Model
print_line('*', 'Testing Begin')

testFILES = os.listdir(testFileDIR)

for f in testFILES:
    if f[0] == '.':
        continue
    correct = 0
    try:
        tempFILE = testFileDIR + '/' + f
        tempMatrix = np.load(tempFILE)
        testMatrix = tempMatrix[:, np.newaxis]

        x_test = testMatrix[:, :, 1:]
        y_test = testMatrix[:, :, 0]
        y_test = torch.from_numpy(y_test).float()
        x_test = torch.from_numpy(x_test).float()

        testDataSet = Data.TensorDataset(x_test, y_test)

        testLoader = Data.DataLoader(
            dataset=testDataSet,  # torch TensorDataset format
            batch_size=BATCH_SIZE,  # mini batch size
            shuffle=True  # shuffle the data
        )

        
        for x_test, y_test in testLoader:
            x_test = Variable(x_test)
            outputs = reload_model(x_test)
            outputs = outputs.data.squeeze()

        is_sick = 0
        in_one_file_total = 0
        for output in outputs:
            in_one_file_total += 1
            if (output - 0) ** 2 - (output - 1) ** 2 < 0:
                predicted = 0
            else:
                predicted = 1
                is_sick += 1
            if (predicted - y_test[0])**2<0.00001:
                correct+=1    
        print('This guy may be {} percentage in sick.'.format(100*is_sick / in_one_file_total))
    
    except ZeroDivisionError: 
        print("division by zero: in_one_file_total of file {} is equal to 0.".format(f))
    except TypeError:
        print("There is a TypeError with file {}.".format(f))

    try:
        print('Test Accuracy of file {0} is {1}%.'.format(f, 100 * correct / in_one_file_total))
    except ZeroDivisionError: 
        print("division by zero: in_one_file_total is equal to 0.")
    

    

timePoint4 = time.time()
print("{} seconds have passed!".format(timePoint4 - timePoint3))
print("Test Ends")


