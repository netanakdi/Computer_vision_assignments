# %% [markdown]
# # EX2_1
# build dilate and erode functions
# %%
import numpy as np
import matplotlib.pyplot as plt
import cv2

figsize = (10, 10)

# %%

img = np.zeros((50, 50))
img[20:30, 20:30] = 1

plt.figure(figsize=figsize)
plt.imshow(img, cmap="gray")
plt.show()

# %%
kernel = np.zeros((5, 5), dtype=np.uint8)
kernel[2, :] = 1
kernel[:, 2] = 1

plt.figure(figsize=figsize)
plt.imshow(kernel, cmap="gray")
plt.show()


# %%

# %%
def my_dilate(img, kernel):
    # TODO: build dilate function without cv2.dilate
    img_h, img_w = img.shape
    kernel_h, kernel_w = kernel.shape
    # Calculate padding size for both height and width
    pad_h = kernel_h // 2
    pad_w = kernel_w // 2
    # Add padding around the image to handle edges during dilation
    padded_img = np.pad(img, ((pad_h, pad_h), (pad_w, pad_w)), mode='constant', constant_values=0)
    # Initialize the output image (same size as the input image)
    dilated_img = np.zeros_like(img)
    # Loop through every pixel in the input image
    for i in range(img_h):
        for j in range(img_w):
            # Extract the region in the padded image corresponding to the kernel size
            pixels_to_dilate = padded_img[i:i + kernel_h, j:j + kernel_w]
            # Perform the dilation check (if any overlap exists with the kernel)
            if np.sum(pixels_to_dilate * kernel) >= 1:
                dilated_img[i, j] = 1
            else:
                dilated_img[i, j] = 0
    return dilated_img


plt.figure(figsize=figsize)
plt.imshow(my_dilate(img, kernel), cmap="gray")
plt.show()

# %%
# TODO: show that cv2.dilate and my_dilate are the same using absolute difference
if np.all([cv2.dilate(img, kernel, iterations=1) == my_dilate(img,kernel)]):
    print("cv2.dilate & my_dilate are the same!")
else:
    print("try again...")


# %%
def my_erode(img, kernel):
    # TODO: build erode function without cv2.erode
    img_h, img_w = img.shape
    kernel_h, kernel_w = kernel.shape
    th = kernel.sum() # Threshold: total number of 1s in the kernel
    # Calculate padding size for both height and width
    pad_h = kernel_h // 2
    pad_w = kernel_w // 2
    # Add padding around the image to handle edges during erosion
    padded_img = np.pad(img, ((pad_h, pad_h), (pad_w, pad_w)), mode='constant', constant_values=1)
    eroded_img = np.zeros_like(img)
    # Loop through every pixel in the input image
    for i in range(img_h):
        for j in range(img_w):
            # Extract the region in the padded image corresponding to the kernel size
            pixels_to_dilate = padded_img[i:i + kernel_h, j:j + kernel_w]
            # Perform the erosion check (all kernel elements must match the region)
            if np.sum(pixels_to_dilate * kernel) >= th:
                eroded_img[i, j] = 1
        else:
            eroded_img[i, j] = 0
    return eroded_img
plt.figure(figsize=figsize)
plt.imshow(my_erode(img, kernel), cmap="gray")
plt.show()

# %%
# TODO: show that cv2.erode and my_erode are the same using absolute difference
if np.all([cv2.erode(img, kernel, iterations=1) == my_erode(img,kernel)]):
    print("cv2.erode & my_erode are the same!")
else:
    print("try again...")

# %%
