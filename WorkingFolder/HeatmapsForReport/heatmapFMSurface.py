import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

#width är x-led från 3-5
#mu är y-led från dry-snow
mu = ["dry", "wet", "snow"]
width = [3, 3.5, 4, 4.5, 5, 5.5, 6, 6.5, 7, 7.5, 8]
phi = [[8.403, 6.209, 4.397, 2.397, 1.164, 1.126, 1.194, 1.268, 1.322, 1.364, 1.407],
       [8.065, 6.612, 4.845, 2.2928, 0.942, 0.628, 1.012, 1.106, 1.158, 1.215, 1.350], 
       [7.036, 4.052, 2.310, 1.645, 0.857, 0.868, 0.854, 0.864, 0.850, 0.856, 0.795]]
ax = sns.heatmap(phi, xticklabels=width, yticklabels=mu, annot=True, cbar_kws={'label': 'SFA [deg]'})
ax.set_xlabel('Width [m]')
ax.set_ylabel('Surface')
plt.show()