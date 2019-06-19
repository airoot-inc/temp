import os
import librosa
import numpy

root = r"F:\Riku_Ka\PROMPT-UnzipAudioFile\original used voice files"
listFILE = r"F:\Riku_Ka\for the final report\MPCC-npy Data Set\ready data considering hamd17 with threshold 8 non-overlap\files of healthy without depression 10.txt"
dstFILE = r"F:\Riku_Ka\for the final report\MPCC-npy Data Set\ready data considering hamd17 with threshold 8 non-overlap\Duration of files of healthy without depression 10.txt"

#files = os.listdir(root)

keywords = []
with open(listFILE) as of:    
    lines = of.readlines()
    for l in lines:
        keywords.append(l.strip('\n'))
print(keywords)

lenLIST = []

for root, dirs, files in os.walk(root):
    print("root is {0}, dirs is {1}, files is {2}.".format(root, dirs, files))
    for fileName in files:
        print(fileName)
        if os.path.isfile(os.path.join(root,fileName)):
            for k in keywords:
                #print("For keyword {}".format(k))
                if fileName.find(k) != -1:
                    myLen = librosa.get_duration(filename= str(os.path.join(root,fileName)))
                    print("The length of file {0} is {1}.".format(fileName, myLen))
                    lenLIST.append(fileName.strip('.wav') + '\t')
                    lenLIST.append(str(myLen) + '\n')

print(lenLIST)

with open(dstFILE, 'w') as of:
    of.writelines(lenLIST)