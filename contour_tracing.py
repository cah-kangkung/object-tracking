import numpy as np
import copy as copy
import cv2 as cv2

class ContourTracing(object):
    def __init__(self):
        pass

    def move_pointer(self, direction, i, j):
        """
            u
            |
            |
      l-----------r
            |
            |
            d

      Parameters
      ----------
      direction : string
         DESCRIPTION.
      i : number
         DESCRIPTION.
      j : number
         DESCRIPTION.

      Returns
      -------
      None.
      
      """
        if direction == "u":
            i -= 1
            j = j
        elif direction == "r":
            i = i
            j += 1
        elif direction == "d":
            i += 1
            j = j
        elif direction == "l":
            i = i
            j -= 1

        return i, j

    def next_pointer_position(self, current, pivot, direction):
        """
      Parameters
      ----------
      current : number[][]
         Current pixel coordinate.
      pivot : number[]
         Pivot.
      direction: number
         1 for clockwise, 2 for counter-clockwise

      Returns
      -------
      position: string
         Position char.
      next_pointer: number[][]
         Next Pointer.
      """
        position = ""
        if current[0] == pivot[0] and current[1] == pivot[1] - 1:
            position = "l"
            if direction == 1:
                current[0], current[1] = self.move_pointer("u", current[0], current[1])
            else:
                current[0], current[1] = self.move_pointer("d", current[0], current[1])
        elif current[0] == pivot[0] - 1 and current[1] == pivot[1] - 1:
            position = "ul"
            if direction == 1:
                current[0], current[1] = self.move_pointer("r", current[0], current[1])
            else:
                current[0], current[1] = self.move_pointer("d", current[0], current[1])
        elif current[0] == pivot[0] - 1 and current[1] == pivot[1]:
            position = "u"
            if direction == 1:
                current[0], current[1] = self.move_pointer("r", current[0], current[1])
            else:
                current[0], current[1] = self.move_pointer("l", current[0], current[1])
        elif current[0] == pivot[0] - 1 and current[1] == pivot[1] + 1:
            position = "ur"
            if direction == 1:
                current[0], current[1] = self.move_pointer("d", current[0], current[1])
            else:
                current[0], current[1] = self.move_pointer("l", current[0], current[1])
        elif current[0] == pivot[0] and current[1] == pivot[1] + 1:
            position = "r"
            if direction == 1:
                current[0], current[1] = self.move_pointer("d", current[0], current[1])
            else:
                current[0], current[1] = self.move_pointer("u", current[0], current[1])
        elif current[0] == pivot[0] + 1 and current[1] == pivot[1] + 1:
            position = "dr"
            if direction == 1:
                current[0], current[1] = self.move_pointer("l", current[0], current[1])
            else:
                current[0], current[1] = self.move_pointer("u", current[0], current[1])
        elif current[0] == pivot[0] + 1 and current[1] == pivot[1]:
            position = "d"
            if direction == 1:
                current[0], current[1] = self.move_pointer("l", current[0], current[1])
            else:
                current[0], current[1] = self.move_pointer("r", current[0], current[1])
        elif current[0] == pivot[0] + 1 and current[1] == pivot[1] - 1:
            position = "dl"
            if direction == 1:
                current[0], current[1] = self.move_pointer("u", current[0], current[1])
            else:
                current[0], current[1] = self.move_pointer("r", current[0], current[1])

        next_pointer = current
        return position, next_pointer

    def border_following(self, img, start, previous):
        """
        Tracing border of an object
        
        Parameters
        ----------
        img : number[][]
            Image Object.
        start : number[]
            A pixel coordinate which border following starts from.
        previous : number[]
            Previous pixel coordinate of starting point.

        Returns
        -------
        List of coordinate for border points
        """
        pointer_one = previous
        pointer_three = start
        contour = []

        # Step 3.1 Move clockwise
        count = 0
        while img[pointer_one[0]][pointer_one[1]] == 0:
            # If the starting pixel is a single pixel dot
            if count > 7:
                img[pointer_three[0]][pointer_three[1]] = 4
                contour.append(np.array([[pointer_three[1] - 1, pointer_three[0] - 1]]))
                return np.array(contour)
                
            position, next_pointer = self.next_pointer_position(pointer_one, pointer_three, 1)
            pointer_one = next_pointer
            count += 1

        # Step 3.2
        pointer_two = copy.copy(pointer_one)

        counter = 0
        while True:
            # Step 3.3 Move counter clockwise
            # First, move pointer one time in counter-clockwise direction
            position, next_pointer = self.next_pointer_position(pointer_two, pointer_three, 2)
            pointer_two = next_pointer
            while img[pointer_two[0]][pointer_two[1]] == 0:
                position, next_pointer = self.next_pointer_position(pointer_two, pointer_three, 2)
                pointer_two = next_pointer
            pointer_four = pointer_two

            # Step 3.4 Assign NBD
            # rows or i represent y-axis
            # cols or j represent x-axis
            # the coordinate are inverted because we wanted to return a set of (x, y) points, not (y, x)
            # we use 2 and 4 (-2) since we only extract outer border
            nbd_coordinate = copy.copy(pointer_three)
            if img[nbd_coordinate[0]][nbd_coordinate[1] + 1] == 0:
                img[nbd_coordinate[0]][nbd_coordinate[1]] = 4
            elif img[nbd_coordinate[0]][nbd_coordinate[1] + 1] != 0 and img[nbd_coordinate[0]][nbd_coordinate[1]] == 1:
                img[nbd_coordinate[0]][nbd_coordinate[1]] = 2

            contour.append(np.array([[nbd_coordinate[1] - 1, nbd_coordinate[0] - 1]]))

            # Step 3.5 Determine new pointer or break
            if pointer_four[0] == start[0] and pointer_four[1] == start[1]:
                if pointer_three[0] == pointer_one[0] and pointer_three[1] == pointer_one[1]:
                    break
            pointer_two = copy.copy(pointer_three)
            pointer_three = copy.copy(pointer_four)

            counter += 1

        return np.array(contour)

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
        (ret, thresh2) = cv2.threshold(img, 127, 1, cv2.THRESH_BINARY)
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
