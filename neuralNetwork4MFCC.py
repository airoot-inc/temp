#training the NN
import sys
#sys.path.append("F:/Riku_Ka/AudioTestCode")
import shutil
import os
from pydub  import AudioSegment
import numpy as np
from keras import Sequential
from keras.layers import Dense, Activation
from keras.optimizers import adam
from keras.models import load_model
import time
from keras.models import model_from_json

print("Loading data....")
startTime = time.time()

# predefine the const
NUM_EPOCH = 20

#produce training data set.
train_npy_FILE = "F:/Riku_Ka/for the final report/MPCC-npy Data Set/ready data considering hamd17 with threshold 8 non-overlap/DS2/DS2 train.npy"
testDIR = r"F:\Riku_Ka\for the final report\MPCC-npy Data Set\ready data considering hamd17 with threshold 8 non-overlap\DS2\test"
modelDIR = r"F:\Riku_Ka\for the final report\MPCC-npy Data Set\ready data considering hamd17 with threshold 8 non-overlap\DS2\model"
modelFILE = "new NN Model for DS2 0Health-1Depression with 39MFCC.json"
weightFILE = "new NN Model for DS2 0Health-1Depression with 39MFCC.h5"

trainDS = np.load(train_npy_FILE)
print("The shape of train dataset is {}".format(trainDS.shape))
trainDATA = trainDS[:,1:]
trainLABLE = trainDS[:,0]

print("Start training!")
print("{} secondes have passed. ".format(str(time.time() - startTime)))


np.random.seed(10)

myModel = Sequential()

myModel.add(Dense(units=64,input_dim=39, activation='relu'))
myModel.add(Dense(units=8, activation='relu'))
myModel.add(Dense(1, activation='sigmoid'))

myModel.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

myModel.fit(x=trainDATA, y=trainLABLE, epochs=NUM_EPOCH, batch_size=64)

# Save the trained model
myModel_json = myModel.to_json()
with open(modelDIR + '/' + modelFILE, 'w') as json_file:
    json_file.write(myModel_json)
myModel.save_weights(modelDIR + '/' + weightFILE)
print("The trained model has been saved.")


print("Start testing!")
print("Time point 3 is " + str(time.time() - startTime))

# Load the model
with open(modelDIR + '/' + jsonFILE) as modelFile:
    model_json = modelFile.read()

myModel = model_from_json(model_json)
myModel.load_weights(modelDIR + '/' + weightFILE)
myModel.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

print("Start testing!")
print("Time point 2 is " + str(time.time() - startTime))

#produce test data set
with open(modelDIR + '/' + testResultFILE, 'w+') as of:
    NPYfiles = os.listdir(testDIR)
    resultLIST = []
    for f in NPYfiles:
        try:
            tempFile = testDIR + '/' + f
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

    

print("It is finished.")



