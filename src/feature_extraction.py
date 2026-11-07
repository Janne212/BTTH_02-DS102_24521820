import numpy as np
import cv2

def ExtractHSVHistogramFeature(image, bins = (8, 8, 8)):
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    hist = cv2.calcHist([hsv_image], [0, 1, 2], None, bins, [0, 180, 0, 256, 0, 256])

    cv2.normalize(hist, hist)

    return hist.flatten()

def ExtractHOGFeature(
    image,
    cell_size = (8, 8),
    block_size = (2, 2),
    nbins = 9,
    image_size = (256, 256),
):
    resized_image = cv2.resize(image, image_size)
    gray_image = cv2.cvtColor(resized_image, cv2.COLOR_BGR2GRAY)

    hog_descriptor = cv2.HOGDescriptor(
        _winSize = image_size,
        _blockSize = (  block_size[1] * cell_size[1],
                        block_size[0] * cell_size[0]),
        _blockStride = (cell_size[1], cell_size[0]),
        _cellSize = (cell_size[1], cell_size[0]),
        _nbins = nbins,
    )

    hog_features = hog_descriptor.compute(gray_image)

    return hog_features.flatten()

def ExtractFeature(
    image,
    bins = (8, 8, 8),
    cell_size = (8, 8),
    block_size = (2, 2),
    nbins = 9,
    image_size = (256, 256),
):
    hsv_features = ExtractHSVHistogramFeature(image, bins = bins)
    hog_features = ExtractHOGFeature(
        image,
        cell_size = cell_size,
        block_size = block_size,
        nbins = nbins,
        image_size = image_size,
    )

    hsv_features /= np.linalg.norm(hsv_features) + 1e-12
    hog_features /= np.linalg.norm(hog_features) + 1e-12

    return np.concatenate((hsv_features, hog_features))