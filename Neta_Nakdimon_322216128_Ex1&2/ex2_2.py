# %% [markdown]
# # EX2_2
# Find different words in newspaper article
# We'll do this using morphology operators and connected components.
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
        "https://github.com/YoniChechik/AI_is_Math/raw/master/c_02a_basic_image_processing/ex2/news.jpg"
    )
# %%
import cv2
import matplotlib.pyplot as plt
import numpy as np

figsize = (10, 10)

# %%
im = cv2.imread("news.jpg")
im_gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)

plt.figure(figsize=figsize)
plt.imshow(im_gray, cmap="gray", vmin=0, vmax=255)
plt.show()

# %%
# TODO: let's start with turning the image to a binary one
# Apply a threshold to convert the grayscale image to a binary image
_, im_th = cv2.threshold(im_gray, 180, 255, cv2.THRESH_BINARY_INV)
plt.figure(figsize=(20, 20))
plt.imshow(im_th, cmap="gray", vmin=0, vmax=255)
plt.show()

# %%
# TODO: next, merge all pixels of the same word together to make one connected component using a morphologic operator
# Create a rectangular structuring element (kernel) for dilation
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (4,2))
# Apply the dilation operation to merge nearby pixels into connected components
dilated_im = cv2.dilate(im_th, kernel, iterations=1)
plt.figure(figsize=(20, 20))
plt.imshow(dilated_im, cmap="gray", vmin=0, vmax=255)
plt.show()

# %%


def find_words(dilated_im, im):
    res = im.copy()

    # TODO: draw rectengles around each word:
    # 1. find all connected components
    # 2. build a mask of only one connected component each time, and find it extremeties
    # Find connected components in the dilated image:
    # - num_labels: number of connected components (including background)
    # - labels: array where each pixel is labeled with its component
    # - stats: statistics for each component
    # - _: centroids of connected components (not used here because isnt needed)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(dilated_im)
    # Loop through each connected component (skip the background, label 0)
    for label in range(1, num_labels):
        x, y, w, h, _ = stats[label]
        # Create a mask for the current connected component
        mask_rect = np.zeros_like(dilated_im, dtype=np.uint8)
        mask_rect[y:y+h, x:x+w] = 1 # Set the mask to include only the current component
        # Draw the rectangle on the result image
        res = plot_rec(mask_rect, res)
    # TODO: did it came out perfect? Why? Why not?
    # It did not come out perfect. After adjusting the kernel boundaries multiple times,
    # I achieved the desired result. (Full explanation provided in the PDF)
    return res


def plot_rec(mask, res_im):
    # plot a rectengle around each word in res image using mask image of the word
    xy = np.nonzero(mask)
    y = xy[0]
    x = xy[1]
    left = x.min()
    right = x.max()
    up = y.min()
    down = y.max()

    res_im = cv2.rectangle(res_im, (left, up), (right, down), (0, 20, 200), 2)
    return res_im


# %%
plt.figure(figsize=(20, 20))
plt.imshow(find_words(dilated_im, im))
plt.show()


# %%
# TODO: now we want to mark only the big title words, and do this ONLY using morphological operators
# Create a square kernel for erosion to remove small noise and separate connected components
erode_kernel = np.ones((7, 7), np.uint8)
# Apply erosion to the binary image to reduce noise and shrink components
eroded_im = cv2.erode(im_th.copy(), erode_kernel, iterations=1)
# Create a rectangular kernel for dilation to merge nearby components into larger ones
dilate_kernel = np.ones((7, 26), np.uint8)
# Apply dilation to reconnect parts of larger components - the big title words
dilated_im = cv2.dilate(eroded_im, dilate_kernel, iterations=1)
binary_only_title_cc_img = dilated_im.copy()

plt.figure(figsize=(20,20))
plt.imshow(find_words(binary_only_title_cc_img, im))
plt.show()
