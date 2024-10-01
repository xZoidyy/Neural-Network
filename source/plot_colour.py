import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

data1 = pd.read_csv('../datas/boundery_decition.txt', sep='\s+', header=None)
data1 = pd.DataFrame(data1)

x2 = data1[0]
x1 = data1[1]
t = data1[2]

# Create a scatter plot with color mapping
plt.scatter(x1, x2, c=t, cmap='viridis')  # You can choose a different colormap ('viridis' is just an example)

# Add colorbar to show the mapping of values to colors
plt.colorbar(label='Output value (0.5 = decition value)')

# Set labels for x and y axes
plt.xlabel('x2')
plt.ylabel('x1')

plt.title('Decision Boundary Plot (NN)')

###############################################################################

# Show the plot
plt.show()