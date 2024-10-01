import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

data1 = pd.read_csv("../datas/predicted_data.txt", sep='\s+', header=None)
data1 = pd.DataFrame(data1)

x2 = data1[0]
x1 = data1[1]
t = data1[2]

def plotting(x1a, x2a, ta):
    counter1 = 0
    counter2 = 0
    for i in range(0, len(x1a)):
        if (0.9 < ta[i] < 1.1):
            if counter1 == 0:
                plt.plot(x1a[i], x2a[i], 'b.', label='Class = 1 (Alpha decay)')
            else:
                plt.plot(x1a[i], x2a[i], 'b.')
            counter1 += 1
        else:
            if counter2 == 0:
                plt.plot(x1a[i], x2a[i], 'r.', label='Class = 0')
            else:
                plt.plot(x1a[i], x2a[i], 'r.')
            counter2 += 1

def boundary(x1_b, x2_b, tb):
    counterr = 0
    for i in range(0, len(x1_b)):
        if (0.499 < tb[i] < 0.501):
            if counterr == 0:
                plt.plot(x1_b[i], x2_b[i], 'g.', label='Predicted boundary')
            else:
                plt.plot(x1_b[i], x2_b[i], 'g.')
            counterr += 1

plt.subplot(1,1,1)
plt.xlabel(r'N')
plt.ylabel(r'Z')

plotting(x1, x2, t) # True = plot training data

#plt.legend()
"""
plt.subplot(1,2,2)
plt.xlabel(r'x2')
plt.ylabel(r'x1')
plt.xlim(-3, 3)
plt.ylim(-3, 3)

plotting(x1_new, x2_new, t_new) # False = plot new data
boundary(x1_boundary, x2_boundary, t_boundary)
"""

plt.legend()
plt.show()