import cv2
import numpy as np

from .config import CROP_BLUR_KERNEL, CROP_DILATE_ITERS, CROP_ERODE_ITERS, CROP_THRESH


def _grab_contours(cnts):
    if len(cnts) == 2:
        return cnts[0]
    if len(cnts) == 3:
        return cnts[1]
    return cnts


def crop_brain_contour(image: np.ndarray) -> np.ndarray:
    """Crop to brain extrema. Matches Crop_Brain_Contours.ipynb / paper Fig. 2."""
    if image.ndim == 2:
        bgr = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    else:
        bgr = image

    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, CROP_BLUR_KERNEL, 0)
    thresh = cv2.threshold(gray, CROP_THRESH, 255, cv2.THRESH_BINARY)[1]
    thresh = cv2.erode(thresh, None, iterations=CROP_ERODE_ITERS)
    thresh = cv2.dilate(thresh, None, iterations=CROP_DILATE_ITERS)

    cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = _grab_contours(cnts)
    if not len(cnts):
        return bgr

    contour = max(cnts, key=cv2.contourArea)
    ext_left = tuple(contour[contour[:, :, 0].argmin()][0])
    ext_right = tuple(contour[contour[:, :, 0].argmax()][0])
    ext_top = tuple(contour[contour[:, :, 1].argmin()][0])
    ext_bot = tuple(contour[contour[:, :, 1].argmax()][0])

    cropped = bgr[ext_top[1] : ext_bot[1], ext_left[0] : ext_right[0]]
    if cropped.size == 0:
        return bgr
    return cropped
