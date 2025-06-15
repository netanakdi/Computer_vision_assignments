# %% [markdown]
# # EX4a: Vignetting
# In photography and optics, vignetting is a reduction of an
# image's brightness or saturation toward the periphery compared
# to the image center.
#
# Mechanical vignetting (for example) occurs when light beams emanating from
# object points located off-axis are partially blocked by
# external objects such as thick or stacked filters, secondary
# lenses, and improper lens hoods. [Wikipedia]
#
# Read more about it here:
# https://en.wikipedia.org/wiki/Vignetting
# https://photographylife.com/what-is-vignetting
#
# You are an algorithm engineer in a new cutting-edge camera
# startup with a new problem of automatically correct vignetting problems.

# The simple approach:
# Getting a calibration image of a white wall and then simply divide this calib_im with each raw image to get a fixed image.
# The problem:
# Not enough memory in the camera to save multiple calib images per each lens configuration.
#
# The New process the we propose is as follows:
# 1. For each new lens the end-user wants to use, he will shoot one calibration image of a white wall.
# 2. Calibration params are saved for this lens configuration.
# 3. The user can now work with this lens configuration and the vignetting will be corrected for all images.

# Behind the scenes we need to build 3 functions:
# 1. `get_index_matrix()` that returns the
# index matrix (called `X` in our lesson). We will build this in python but our engineers will convert it to ASIC,
# so this function CAN'T change.
# 2. `get_calib_coeffs(calib_map)` that gets a raw image of the white wall and returns the betta vector (the calibration params).
# 3. `fix_raw_im(b, vig_im)` that gets the user image and the betta vector and fixes the image vignetting.
# It reconstruct the calib image input from (2) using (1) + betta vector and then divide the input image with this reconstructed calib_map to get a fixed image.

# %%
# to run in google colab
import sys

import numpy as np

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
        "https://github.com/YoniChechik/AI_is_Math/raw/master/c_04a_curve_fitting/ex4a/calib_im1.jpg"
    )
    download_from_web(
        "https://github.com/YoniChechik/AI_is_Math/raw/master/c_04a_curve_fitting/ex4a/calib_im2.jpg"
    )
    download_from_web(
        "https://github.com/YoniChechik/AI_is_Math/raw/master/c_04a_curve_fitting/ex4a/calib_im3.jpg"
    )
    download_from_web(
        "https://github.com/YoniChechik/AI_is_Math/raw/master/c_04a_curve_fitting/ex4a/vignette_im1.jpg"
    )
    download_from_web(
        "https://github.com/YoniChechik/AI_is_Math/raw/master/c_04a_curve_fitting/ex4a/vignette_im2.jpg"
    )
    download_from_web(
        "https://github.com/YoniChechik/AI_is_Math/raw/master/c_04a_curve_fitting/ex4a/vignette_im3.jpg"
    )

# %%
import cv2
import matplotlib.pyplot as plt

IMAGE_SHAPE = [768, 1366]


# %%
def get_index_matrix():
    # TODO: get x,y index for each pixel as column vectors
    # HINT: use np.meshgrid() + reshape
    x, y = np.meshgrid(np.arange(IMAGE_SHAPE[1]), np.arange(IMAGE_SHAPE[0]))
    x = x.reshape(-1, 1)
    y = y.reshape(-1, 1)

    # X is the raw dataset from which we will reconstruct the calib map
    # X@b = calib_map
    # TODO: build X using x,y and function of them
    # hint: use np.concatenate()
    # hint2: the calibration map looks radial- so to build a good representation of it we need to use x,y but also x^2, y^2 and even xy.
    # this is only one line, but a hard one
    X = np.concatenate([x, y, x**2, y**2, x*y],  axis=1)
    return X


# %%
def get_calib_coeffs(calib_map):
    # This is the calibration function when he user switches lenses.
    # Since it's memory consuming to save the intire calib map,
    # we will save only a parametric representation of it using X,b

    # TODO: transform calib_map to column vector for least-squares
    # one line
    y = calib_map.reshape(-1, 1)

    # build X
    X = get_index_matrix()

    # TODO: use least-squares to find the beta params for later use.
    # one line
    b = np.linalg.lstsq(X, y, rcond=None)[0].flatten()

    return b


# %%


def fix_raw_im(b, vig_im):
    # Each image taken is passed through this block to correct for vignetting
    # Normalize the input vignette image
    vig_im = vig_im.astype(float) / 255
    # build data matrix X
    X = get_index_matrix()

    # TODO: build reconstructed calib map using b params from calibration step
    # use X,b (this is the LS part!!!)
    # one line
    reconstructed_calib_im_column_vector = X @ b
    # reshape into 2d image
    rec_calib_map = reconstructed_calib_im_column_vector.reshape(IMAGE_SHAPE)
    # build 3 copies of the 2d result and concat along the third dim, in order to divide with RGB images
    rec_calib_map_rgb = np.stack([rec_calib_map] * 3, axis=-1)

    # TODO: divide the reconstructed calib image with the input image to get the fixed result
    # one line
    res = np.clip(vig_im / (rec_calib_map_rgb + 1e-10), 0, 1)

    # return the final result + rec_calib_map for debug and testing purposes.
    return res, rec_calib_map


# %%
def calib_testing(calib_map, rec_calib_map):
    # test your calib map reconstruction relative to the original
    # calib map
    # this is just for testing in the lab, not for the end user...

    # TODO:what is the RMSE of the reconstruction?
    # one line
    rmse = np.sqrt(np.mean((calib_map - rec_calib_map) ** 2))

    # TODO: print L1 map of reconstruction
    # one line
    abs_error_map = np.abs(calib_map - rec_calib_map)

    plt.figure()
    plt.imshow(abs_error_map)
    plt.colorbar()
    plt.title("rmse error is " + str(rmse) + ". L1 map:")
    plt.show()


# %%
if __name__ == "__main__":
    for i in range(3):
        calib_im = cv2.imread("calib_im" + str(i + 1) + ".jpg")
        calib_im = cv2.cvtColor(calib_im, cv2.COLOR_BGR2GRAY)
        calib_map = calib_im.astype(float) / 255

        vig_im = cv2.imread("vignette_im" + str(i + 1) + ".jpg")
        vig_im = cv2.cvtColor(vig_im, cv2.COLOR_BGR2RGB)

        # ===== happens in the factory per lens setup
        b = get_calib_coeffs(calib_map)

        # ===== b is then saved to the camera hardware coupled to the lens configuration.
        # so to fix the problem one must use b on the raw image each time he takes a photo:
        res, rec_calib_map = fix_raw_im(b, vig_im)

        # ===== plot results
        plt.figure()
        plt.imshow(vig_im)
        plt.title("original image")
        plt.show()

        plt.figure()
        plt.imshow(res)
        plt.title("fixed image")
        plt.show()

        calib_testing(calib_map, rec_calib_map)

# %%
