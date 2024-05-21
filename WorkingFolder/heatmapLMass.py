import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

#width är x-led från 3-8
#massa är y-led från 2100-6100
mass = [6100, 4100, 2100]
width = [3, 3.5, 4, 4.5, 5, 5.5, 6, 6.5, 7, 7.5, 8]
phi_L = [[12.572, 16.272, 12.530, 10.353, 9.374, 9.280, 9.499, 6.702, 6.685, 4.356, 4.115],
       [13.833, 16.082, 11.302, 10.836, 10.647, 10.040, 5.870, 5.665, 2.702, 2.745, 1.010], 
       [12.673, 14.073, 9.269, 6.859, 4.501, 2.202, 0.809, 0.579, 0.409, 0.411, 0.395]]

ax_L = sns.heatmap(phi_L, xticklabels=width, yticklabels=mass, annot=True, cbar_kws={'label': 'SFA [deg]'})
ax_L.set_xlabel('Width [m]')
ax_L.set_ylabel('Mass [kg]')

plt.show()