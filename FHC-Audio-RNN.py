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
BATCH_SIZE = 64
TIME_STEP = 2000          # rnn time step / image height
INPUT_SIZE = 39        # rnn input size / image width
LR = 0.01               # learning rate
#The following are name of files that must be given before.
resultFile = rootDIR + '/' + "result.txt"
modelFILE = modelFileDIR + '/' + "RNN Modelfor 0Health-1Dementia with 39MFCC of npy form by 333 samples.pth"
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

# prepare the training dataset
trainDS = np.load(train_npy_FILE)
print("The shape of trainDS is {}".format(trainDS.shape))
num_of_line = np.shape(trainDS)[0]
print("The number of trainDS's line is {}".format(num_of_line))
win_trainDS = np.zeros([1, TIME_STEP, INPUT_SIZE+1])
print("The shape of win_trainDS is {}".format(win_trainDS.shape))
# The following for step
for i in range(int(num_of_line/TIME_STEP)):
    window = trainDS[np.newaxis, i * TIME_STEP:(i + 1) * TIME_STEP, :]
    win_trainDS = np.vstack((win_trainDS, window))
    print("The shape of win_testDS is {}".format(win_trainDS.shape))

print("The shape of win_trainDS is {}".format(win_trainDS.shape))


x_train = win_trainDS[:,:,1:]
y_train = win_trainDS[:,0,0]
y_train = torch.from_numpy(y_train).float()
x_train = torch.from_numpy(x_train).float()

trainDataSet = Data.TensorDataset(x_train, y_train)

trainLoader = Data.DataLoader(
    dataset=trainDataSet,  # torch TensorDataset format
    batch_size=BATCH_SIZE,  # mini batch size
    shuffle=True,  # shuffle the data
)


class RNN(nn.Module):
    def __init__(self):
        super(RNN, self).__init__()

        self.rnn = nn.LSTM(         # if use nn.RNN(), it hardly learns
            input_size=INPUT_SIZE,
            hidden_size=64,         # rnn hidden unit
            num_layers=1,           # number of rnn layer
            batch_first=True,       # input & output will has batch size as 1s dimension. e.g. (batch, time_step, input_size)
        )

        self.out = nn.Linear(64, 2)

    def forward(self, x):
        r_out, (h_n, h_c) = self.rnn(x, None)   # None represents zero initial hidden state

        # choose r_out at the last time step
        out = self.out(r_out[:, -1, :])
        return out

rnn = RNN()

optimizer = torch.optim.Adam(rnn.parameters(), lr=LR)   # optimize all cnn parameters
loss_func = nn.CrossEntropyLoss()                       # the target label is not one-hotted

# training
for epoch in range(EPOCH):
    for step, (b_x, b_y) in enumerate(trainLoader):        # gives batch data
        b_x = b_x.view(-1, TIME_STEP, 39)              # reshape x to (batch, time_step, input_size)
        output = rnn(b_x)                               # rnn output
        b_y = b_y.type(torch.LongTensor)
        loss = loss_func(output, b_y)                   # cross entropy loss
        optimizer.zero_grad()                           # clear gradients for this training step
        loss.backward()                                 # backpropagation, compute gradients
        optimizer.step()                                # apply gradients
        print('Epoch [{}/{}] Loss: {:.8f}'.format(epoch + 1, EPOCH, loss.data))

# Test the Model
print_line('*', 'Testing Begin')

testFILES = os.listdir(testDIR)


in_all_file_total = 0
for f in testFILES:
    if f[0] == '.':
        continue
    correct = 0
    try:
        in_all_file_total += 1
        is_sick = 0
        in_one_file_total = 0
        testDS = np.load(testDIR + '/' + f)
        print("The shape of testDS is {}".format(testDS.shape))
        num_of_line = np.shape(testDS)[0]
        print("The number of testDS's line is {}".format(num_of_line))
        win_testDS = np.zeros([1, TIME_STEP, INPUT_SIZE + 1])
        for i in range(int(num_of_line / TIME_STEP)):
            window = testDS[np.newaxis, i * TIME_STEP:(i + 1) * TIME_STEP, :]
            win_testDS = np.vstack((win_testDS, window))
            print("The shape of win_testDS is {}".format(win_testDS.shape))
        print("The shape of win_testDS is {}".format(win_testDS.shape))

        x_test = win_testDS[:, :, 1:]
        x_test = torch.from_numpy(x_test).float()
        label = win_testDS[1, 0, 0] #


        for b_x in x_test:
            in_one_file_total += 1
            b_x = b_x.view(-1, TIME_STEP, 39)  # reshape x to (batch, time_step, input_size)
            output = rnn(b_x)  # rnn output
            if output[0,0]<output[0,1]:
                is_sick += 1
        sick_prob = 100 * is_sick / in_one_file_total
        print('This guy may be {} percentage in sick.'.format(sick_prob))

        if sick_prob > 0.5:
            predicted = 1
        else:
            predicted = 0
        if (predicted - label)**2<0.0001:
            correct += 1

    except ZeroDivisionError:
        print("division by zero: in_one_file_total of file {} is equal to 0.".format(f))
    except TypeError:
        print("There is a TypeError with file {}.".format(f))

    try:
        print('Test Accuracy of for file {0} is {1}%.'.format(f, 100 * correct / in_one_file_total))
    except ZeroDivisionError:
        print("division by zero: in_one_file_total is equal to 0.")   

