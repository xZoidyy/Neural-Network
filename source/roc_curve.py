from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

data1 = pd.read_csv("../datas/boundery_decition.txt", sep='\s+', header=None)
data1 = pd.DataFrame(data1)

x2 = data1[0]
x1 = data1[1]
t = data1[2]

data2 = pd.read_csv("../datas/new_data.txt", sep='\s+', header=None)
data2 = pd.DataFrame(data2)

x2d = data2[1]
x1d = data2[2]
td = data2[10]
print(len(td))
y_test = td

# Step 1: Generate Predictions
y_pred = t  # Replace model with your neural network model

# Step 2: Compute True Positive Rate (TPR) and False Positive Rate (FPR)
fpr, tpr, thresholds = roc_curve(y_test, y_pred)

# Step 3: Plot the ROC Curve
plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % auc(fpr, tpr))
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc="lower right")
plt.show()

# Step 4: Calculate Area Under the Curve (AUC)
auc_score = auc(fpr, tpr)
print("Area Under ROC Curve (AUC):", auc_score)