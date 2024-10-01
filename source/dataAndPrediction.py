import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

data1 = pd.read_csv("../datas/predicted_data.txt", sep='\s+', header=None)
data1 = pd.DataFrame(data1)

x2 = data1[0]
x1 = data1[1]
t = data1[2]

data2 = pd.read_csv("../datas/new_data.txt", sep='\s+', header=None)
data2 = pd.DataFrame(data2)

x2d = data2[1]
x1d = data2[2]
td = data2[10]

error = 0
for i in range(len(x2)):
    error += abs(t[i] - td[i])
error = error/len(x2)

error = int(error*100)

def plotting(x1a, x2a, ta):
    counter1 = 0
    counter2 = 0
    for i in range(0, len(x1a)):
        if (0.9 < ta[i] < 1.1):
            if counter1 == 0:
                plt.plot(x1a[i], x2a[i], 'b.', label='Class = 1 (Alpha decay)')
            elif counter1 == 1:
                plt.plot(x1a[i], x2a[i], 'b.', label='Error = {}%'.format(error))
            else:
                plt.plot(x1a[i], x2a[i], 'b.')
            counter1 += 1
        """else:
            if counter2 == 0:
                plt.plot(x1a[i], x2a[i], 'r.', label='Class = 0')
            else:
                plt.plot(x1a[i], x2a[i], 'r.')
            counter2 += 1
        """

plt.subplot(1,2,1)
plt.xlabel(r'N')
plt.ylabel(r'Z')
plt.xlim(0, 180)
plt.ylim(0, 130)
plt.title('Predicted data')

plotting(x1, x2, t) # True = plot training data
plt.legend()

plt.subplot(1,2,2)
plt.xlabel(r'N')
plt.ylabel(r'Z')
plt.xlim(0, 180)
plt.ylim(0, 130)
plt.title('True data')

plotting(x1d, x2d, td) # True = plot training data
plt.legend()

plt.show()