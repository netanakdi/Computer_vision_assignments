# %% [markdown]
# Let's identify coins!
# in the image given below we want to detect each coin currency,
# and we'll do it with cv2.HoughCircles!

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
        "https://github.com/YoniChechik/AI_is_Math/raw/master/c_04b_hough_transform/ex4b/coins.png"
    )


# %%
import cv2
from matplotlib import pyplot as plt

figsize = (10, 10)

# %%
im3 = cv2.imread("coins.png")
im3 = cv2.cvtColor(im3, cv2.COLOR_BGR2RGB)
im = cv2.cvtColor(im3, cv2.COLOR_RGB2GRAY)
res = im3.copy()

# TODO: fill in the best values possible
# to detect the right circle dimeter and place
acc_ratio = 1
min_dist = 70
canny_upper_th = 500
acc_th = 30
circles = cv2.HoughCircles(
    im,
    cv2.HOUGH_GRADIENT,
    acc_ratio,
    min_dist,
    param1=canny_upper_th,
    param2=acc_th,
    minRadius=15,
    maxRadius=70,
)
# Round and convert circle coordinates to integers
circles = np.round(circles[0, :]).astype("int")
# === font vars
font = cv2.FONT_HERSHEY_SIMPLEX
bottomLeftCornerOfText = (10, 500)
fontScale = 0.8
fontColor = (0, 0, 0)
lineType = 2

# ==== for each detected circle
for xyr in circles:
    radius = xyr[2]
    # draw the outer circle
    res = cv2.circle(res, (xyr[0], xyr[1]), radius, (0, 255, 0), 3)

    # TODO: write currency type on each coin.
    # use cv2.putText() and the font vars above.
    # If you need, different coin sizes can be found here:
    # https://avocadoughtoast.com/weights-sizes-us-coins/
    # Classify the coin based on radius
    if radius <= 53:
        text = "Dime"
    elif radius <= 60:
        text = "Nickel"
    elif radius <= 67:
        text = "Quarter"
    elif radius <= 77:
        text = "Dollar"
    else:
        text = "Other"

    cv2.putText(
        res,
        text,
        (xyr[0] - radius, xyr[1]),
        font,
        fontScale,
        fontColor,
        lineType,
    )



plt.figure(figsize=figsize)
plt.imshow(res)
plt.title("final result- coins detection")
plt.show()

# %%
