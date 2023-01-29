def raster_scan(self, img):
    """
    Take 2d binary image (bw) as an input. Raster scan will create an additional 0 padded
    border around the original image. This will handle all true pixel (255) that sit at the edge
    of the image

    Parameters
    ----------
    img : number[][]
        Bianry image.

    Returns
    -------
    List of contours
    """
    (ret, thresh2) = cv2.threshold(img, 127, 1, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    padded_image = np.pad(thresh2, pad_width=[(1, 1), (1, 1)], mode="constant")
    rows, cols = padded_image.shape
    contours = []

    LNBD = 0
    for i in range(1, rows - 1):
        for j in range(1, cols - 1):
            # Check if pixel is a starting point or not
            if padded_image[i][j] == 1 and padded_image[i][j - 1] == 0:
                if LNBD == 4 or LNBD == 0:
                    contour = self.border_following(padded_image, [i, j], [i, j - 1])
                    contours.append(contour)

            # Check LNBD
            if padded_image[i][j] != 1:
                LNBD = padded_image[i][j]
        LNBD = 0

    return np.array(contours)

def findCountourCustom(self, img):
    return self.raster_scan(copy.copy(img))