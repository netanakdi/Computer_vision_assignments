# %% [markdown]
# # EX 1.3- Opencv: forest pyramid
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
        "https://github.com/YoniChechik/AI_is_Math/raw/master/c_01_intro_to_CV_and_Python/ex1/forest.jpg"
    )
    download_from_web(
        "https://github.com/YoniChechik/AI_is_Math/raw/master/c_01_intro_to_CV_and_Python/ex1/pyramids.png"
    )


# %%
import cv2
import matplotlib.pyplot as plt
import numpy as np  # noqa: F401

# %%
forest_img = cv2.imread("forest.jpg")
pyramid_img = cv2.imread("pyramids.png")
pyramid_img = cv2.cvtColor(pyramid_img, cv2.COLOR_BGR2GRAY)
forest_img = cv2.cvtColor(forest_img, cv2.COLOR_BGR2GRAY)

# %%
# TODO: your goal is to build the pyramids in the forest!
#  Hints to think about:
#  - are the images the same size?
#  - How to mask/overlay the pixels from pyramids image to forest image (use numpy masks that we've seen in the numpy notebook!!!)
#  - A result example is added to this zip file
#
#  TODO: this section can be done in 3-5 lines of nice looking code, try not to write more then 10 lines here.
res_im = []
# Create a binary mask from the pyramid image where non-zero pixels indicate the pyramid
_, mask = cv2.threshold(pyramid_img, 1, 255, cv2.THRESH_BINARY)
# Resize the pyramid image and its mask to match the forest image dimensions
pyramid_img = cv2.resize(pyramid_img, (forest_img.shape[1], forest_img.shape[0]))
mask = cv2.resize(mask, (forest_img.shape[1], forest_img.shape[0]))
# Use the mask to overlay the pyramid image on the forest image
res_im = np.where(mask > 0, pyramid_img, forest_img)

# %%
plt.figure()
plt.imshow(res_im, cmap="gray", vmin=0, vmax=255)
plt.show()
# %%
