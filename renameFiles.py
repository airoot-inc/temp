import os
import shutil

resDIR = r"F:\Riku_Ka\for the final report\MPCC-npy Data Set\ready data considering hamd17 with threshold 8 non-overlap\depression11"

files = os.listdir(resDIR)

for f in files:
    if f.find('PaT') != -1:
        print("The file {} is renamed.".format(f))
        fNew = f.replace('PaT', 'PTa')
        shutil.copy(resDIR + '/' + f, resDIR + '/' + fNew)