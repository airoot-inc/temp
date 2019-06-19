'''
The following script copy the files contained in a list 
from a DIR to another DIR.
'''

import os, random, shutil

resDIR = "F:/Riku_Ka/PROMPT-UnzipAudioFile/all can be used files/voice data"
dstDIR = "F:/Riku_Ka/VOICE data set for BiDEPRESSION"
listFILE = "G:/fileLIST.txt"

#create a file list
depression = []
with open(listFILE, 'r') as f:
    lines = f.readlines()
    for l in lines:
        l = l.strip('\n')
        depression.append(l)
    print(depression)

files = os.listdir(resDIR)

for f in files:
    if any(x in f for x in depression):
        resFILE = resDIR + '/' + f
        dstFILE = dstDIR + '/' + f
        print("For the file {}:          ".format(f))
        try:
            shutil.copy(resFILE, dstFILE)
            print('The file has been copied.')
        except PermissionError:
            print("PermissionError")
        except ValueError:
            print("There is a ValueError.")
