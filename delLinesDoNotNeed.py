import sys
import datetime
import codecs
import re

def deleteEmptyLine(filePath):
    ISOTIMEFORMAT = '%Y-%m%d-%H%M'
    myTime = datetime.datetime.now().strtime(ISOTIMEFORMAT)+".txt"
    myFile = codecs.open(filePath,"r", encoding="utf-8")
    myFinalFile = codecs.open(filePath.rstrip(".txt") + "-final-"+ myTime,"w",encoding="utf-8")

    line = myFile.readline().strip()

    while line != "":
        if line.find("</doc>") != -1:
            myFinalFile.write('\n')
            line = myFile.readline().strip() 
        
        if line.find("<doc id=") != -1:
            line = myFile.readline().strip('\n')
            myFinalFile.write(line)
            myFinalFile.write('\t')
            print(line)
            line = myFile.readline().strip('\n')
        else:
            myFinalFile.write(line)
            print(line)
            line = myFile.readline().strip('\n')
            

    myFinalFile.close()        

myFilePath_ = "C:/test/testWord2Vec.txt"
deleteEmptyLine(myFilePath_)


if __name__=='__main__':
	if len(sys.argv)==1:
		print("Please input the file path!!!")
else:
    deleteEmptyLine(sys.argv[1])