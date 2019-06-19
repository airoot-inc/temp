
import numpy as np
from random import choice
import random
import shutil
import os

healthyOrDepression11 = 'depression11'

listFILE = "F:/Riku_Ka/for the final report/MPCC-npy Data Set/ready data considering hamd17 with threshold 8 without depression 10/ID of "+ healthyOrDepression11 + ".txt"

originalLIST = []
with open(listFILE, 'r') as f:
    lines = f.readlines()
    for l in lines:
        l = l.replace('\n', '')
        originalLIST.append(l)
#print("The length of originalLIST is {}.".format(len(originalLIST)))

fileListForTrainTest = "F:/Riku_Ka/for the final report/MPCC-npy Data Set/ready data considering hamd17 with threshold 8 without depression 10/NewDS3/"+ healthyOrDepression11 + "FileListForTrainTest.txt"
with open(fileListForTrainTest, 'w+') as of:
    testLIST = random.sample(originalLIST, int(len(originalLIST)/2))
    #print("The length of randomLIST1 is {}.".format(len(randomLIST1)))
    #print(randomLIST1)
    of.write("Test ID list of " + healthyOrDepression11 + ":\n")
    for x in testLIST:
        of.write(str(x) + '\n')

    trainLIST = originalLIST
    for x in testLIST:
        if x in trainLIST:
            trainLIST.remove(x)
            #print("The element {} is deleted.".format(x))
    #print("The length of randomLIST2 is {}.".format(len(randomLIST2)))
    #print(randomLIST2)
    of.write("Train ID list of "+  healthyOrDepression11 + ":\n")
    for x in trainLIST:
        of.write(str(x) + '\n')

resDIR = "F:/Riku_Ka/for the final report/MPCC-npy Data Set/ready data considering hamd17 with threshold 8 without depression 10/depression11"
resFILES = os.listdir(resDIR)
dstDIRtest = "F:/Riku_Ka/for the final report/MPCC-npy Data Set/ready data considering hamd17 with threshold 8 without depression 10/NewDS3/test"
dstDIRtrain = "F:/Riku_Ka/for the final report/MPCC-npy Data Set/ready data considering hamd17 with threshold 8 without depression 10/NewDS3/train"

if not os.path.isdir(dstDIRtest):
    os.mkdir(dstDIRtest)

if not os.path.isdir(dstDIRtrain):
    os.mkdir(dstDIRtrain)


# Copy test files
filesNotFoundLIST = []
for f in resFILES:
    if any(k in f for k in testLIST):
        print("The file {} is matched.".format(f))
        resFILE = resDIR + '/' + f
        dstFILE = dstDIRtest + '/' + f
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
print(filesNotFoundLIST)


# Copy train files
for f in resFILES:
    filesNotFoundLIST = []
    if any(k in f for k in trainLIST):
        print("The file {} is matched.".format(f))
        resFILE = resDIR + '/' + f
        dstFILE = dstDIRtrain + '/' + f
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
print(filesNotFoundLIST)
