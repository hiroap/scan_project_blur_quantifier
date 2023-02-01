#%%
# Imports
from Utils import *
from Kernels import *

#%%
# Example script: loads an image and prints its sharpness value on a specific zone
path = "./img/ref.tif"
zone: ZoneDescription = (500, 500, 200)

img_mat = load_image_grayscale(path)

std = std_sharpness(img_mat, zone, laplacian)

print(std)

#%%
# Applies a sobel filter and shows the result
filtered_img_mat = apply_filter(img_mat, sobel_horizontal, absolute_rect=False)

display_img_matrix(filtered_img_mat)

#%%
# Directly prints the std sharpness value from the image path
std = std_sharpness_by_path(path)

print(std)
# %%
