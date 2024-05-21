import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

#width är x-led från 3-8
#massa är y-led från 2100-6100
mass = [6100, 4100, 2100]
width = [3, 3.5, 4, 4.5, 5, 5.5, 6, 6.5, 7, 7.5, 8]
phi_FM = [[8.411, 6.209, 4.397, 2.394, 1.164, 1.117, 1.173, 1.851, 1.322, 1.364, 1.409],
       [8.411, 6.209, 4.397, 2.426, 1.164, 1.126, 1.167, 2.545, 1.322, 1.029, 1.407], 
       [8.403, 6.209, 4.397, 2.397, 1.164, 1.126, 1.194, 1.268, 1.322, 1.364, 1.407]]
ax_FM = sns.heatmap(phi_FM, xticklabels=width, yticklabels=mass, annot=True, cbar_kws={'label': 'SFA [deg]'})
ax_FM.set_xlabel('Width [m]')
ax_FM.set_ylabel('Mass [kg]')

plt.show()