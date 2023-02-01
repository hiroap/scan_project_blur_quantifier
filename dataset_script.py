# Example of a script to graph the std over a set of images contained in the same folder
from Utils import std_over_set
from os import walk
from matplotlib import pyplot as plt
import numpy as np

# Parameters for the execution:
path = './focus/Bis_Mag_2000_P_0_DW_1e-6_1675162602_res_1024x884'
zone = (600, 600, 200)

file_names = ['{0}/focus_{1}.tif'.format(path, i) for i in range(20)]
file_names.insert(10, path + '/ref.tif') # Inserts the reference image in the middle of the series



# Retrives the working distance for each image from the metadata
working_distances = np.empty(21)

for i in range(21):
    with open(file_names[i], "r", errors="ignore") as fp:
        lines = fp.readlines()

        # Takes the entire line after "WD=" as float
        working_distances[i] = float(lines[51][3:])

working_distances = 1000 * working_distances # Convertion to milimeters

# Computes the sharpness
stds = std_over_set(file_names, zone)

stds = - stds / stds[10] + 1

plt.figure()

plt.plot(working_distances, stds)
plt.xlabel("Working distance (mm)")
plt.ylabel("Sharpness")

plt.show()

