# %%
# to run in google colab
import sys

if "google.colab" in sys.modules:

    def download_from_web(url):
        import requests

        response = requests.get(url)
        if response.status_code == 200:
            with open(url.split("/")[-1], "wb") as file:
                file.write(response.content)
        else:
            raise Exception(
                f"Failed to download the image. Status code: {response.status_code}"
            )


    download_from_web(
        "https://github.com/YoniChechik/AI_is_Math/raw/master/c_03_edge_detection/ex3/butterfly_noisy.jpg"
    )

# %%
import cv2
import matplotlib.pyplot as plt
import numpy as np

figsize = (10, 10)


# %%
def bilateral_one_pixel(source, x, y, d, sigma_r, sigma_s):
    # === init vars
    filtered_pix = 0
    Wp = 0

    # TODO:
    # 1. run on all neighboors (~3 lines)
    # 2. if neighboor out of matrix indices - don't count him in your computation (~2 lines)
    # 3. find filtered_pix (~6 lines)
    height, width = source.shape
    # Iterate over the neighborhood of the pixel(square window centered in [x, y])
    for i in range(-d, d + 1):
        for j in range(-d, d + 1):
            neighbor_x = x + i
            neighbor_y = y + j
            # Check if the neighbor is within the matrix indices
            if 0 <= neighbor_x < width and 0 <= neighbor_y < height:
                # Calc the weights
                spatial_weight = np.exp(-(i ** 2 + j ** 2) / (2 * sigma_s ** 2))
                intensity_diff = source[neighbor_y, neighbor_x] - source[y, x]
                range_weight = np.exp(-(intensity_diff ** 2) / (2 * sigma_r ** 2))
                weight = spatial_weight * range_weight
                # Sum up intensity values weighted by the combined weight
                filtered_pix += weight * source[neighbor_y, neighbor_x]
                Wp += weight
    # Normalizing
    if Wp != 0:
        filtered_pix /= Wp
    # make result uint8
    filtered_pix = np.clip(filtered_pix, 0, 255).astype(np.uint8)
    return filtered_pix


# %%
def bilateral_filter(source, d, sigma_r, sigma_s):
    # build empty filtered_image
    filtered_image = np.zeros(source.shape, np.uint8)
    # make input float
    source = source.astype(float)
    # d must be odd!
    assert d % 2 == 1, "d input must be odd"

    # TODO: run on all pixels with bilateral_one_pixel(...) (~4 lines)
    height, width = source.shape
    for i in range(width):
        for j in range(height):
            filtered_image[j, i] = bilateral_one_pixel(source, i, j, d, sigma_r, sigma_s)

    return filtered_image


# %%
# upload noisy image
src = cv2.imread("butterfly_noisy.jpg")
src = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)

plt.figure(figsize=(10, 10))
plt.imshow(src, cmap="gray", vmin=0, vmax=255)
plt.colorbar()
plt.show()

# %%
# ======== run
d = 5  # edge size of neighborhood perimeter
sigma_r = 12  # sigma range
sigma_s = 16  # sigma spatial

my_bilateral_filtered_image = bilateral_filter(src, d, sigma_r, sigma_s)

plt.figure(figsize=(10, 10))
plt.imshow(my_bilateral_filtered_image)
plt.colorbar()
plt.show()

# %%
# compare to opencv
cv2_bilateral_filtered_image = cv2.bilateralFilter(src, d, sigma_r, sigma_s)

plt.figure(figsize=(10, 10))
plt.imshow(cv2_bilateral_filtered_image)
plt.colorbar()
plt.show()

# %%
# compare to regular gaussian blur
gaussian_filtered_image = cv2.GaussianBlur(src, (d, d), sigma_s)
plt.figure(figsize=(10, 10))
plt.imshow(gaussian_filtered_image)
plt.colorbar()
plt.show()

# %%
# copare canny results between regular  two images
th_low = 100
th_high = 200
res = cv2.Canny(my_bilateral_filtered_image, th_low, th_high)
plt.figure(figsize=(10, 10))
plt.imshow(res)
plt.colorbar()
plt.show()

res = cv2.Canny(gaussian_filtered_image, th_low, th_high)
plt.figure(figsize=(10, 10))
plt.imshow(res)
plt.colorbar()
plt.show()
