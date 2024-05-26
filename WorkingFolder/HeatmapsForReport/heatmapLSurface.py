import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

#width är x-led från 3-5
#mu är y-led från dry-snow
mu = ["dry", "wet", "snow"]
width = [3, 3.5, 4, 4.5, 5, 5.5, 6, 6.5, 7, 7.5, 8]
phi = [[12.673, 14.073, 9.269, 6.859, 4.501, 2.202, 0.809, 0.579, 0.409, 0.411, 0.395],
       [12.588, 14.570, 8.495, 5.489, 3.261, 1.339, 0.384, 0.381, 0.382, 0.382, 0.415], 
       [5.182, 8.280, 0.173, 0.191, 0.191, 0.198, 0.203, 0.731, 0.239, 0.221, 0.591]]
ax = sns.heatmap(phi, xticklabels=width, yticklabels=mu, annot=True, cbar_kws={'label': 'SFA [deg]'})
ax.set_xlabel('Width [m]')
ax.set_ylabel('Surface')
plt.show()