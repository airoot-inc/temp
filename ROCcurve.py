from sklearn.metrics import roc_curve
from sklearn.metrics import auc
import matplotlib.pyplot as plt
import numpy as np


resDIR = "F:/Riku_Ka/for the final report\MPCC-npy Data Set/ready data considering hamd17 with threshold 8 non-overlap"
scoreFile = resDIR + '/' + "ROC data.txt"
myTitle = "DS3"
pngFILE = resDIR + '/' + myTitle + '.png'


originalScore = np.loadtxt(scoreFile)
print(originalScore.shape)
y_true = originalScore[:,0]
print(y_true)
y_score = originalScore[:,1]
print(y_score)
# Compute fpr, tpr, thresholds and roc auc
fpr, tpr, thresholds = roc_curve(y_true, y_score)
print(fpr)
print(tpr)
print(thresholds)
roc_auc = auc(fpr, tpr)


minDistance = 2
j = 0
for i in range(len(fpr)):
    v1= np.array([fpr[i], tpr[i]])
    v2= np.array([0, 1])
    tempDistance = np.sqrt(np.sum(np.square(v1 - v2)))
    if tempDistance < minDistance:
        minDistance = tempDistance
        j = i
print(minDistance)
print("The nearest point to the point [0,1] is the {0}th point at the position[{1},{2}].".format(j, fpr[j], tpr[j]))

# Plot ROC curve
plt.plot(fpr, tpr, label='ROC curve (area = %0.3f)' % roc_auc)
plt.plot([0, 1], [0, 1], 'k--')  # random predictions curve
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.0])
plt.xlabel('False Positive Rate or (1 - Specificity)')
plt.ylabel('True Positive Rate or (Sensitivity)')
plt.title(myTitle)
plt.legend(loc="lower right")
plt.savefig(pngFILE)
plt.show(1)