from imutils.perspective import four_point_transform
from skimage.segmentation import clear_border
import numpy as np
import imutils
import cv2

def find_puzzle(image, debug=False):

    grayImage = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    imageBlurred = cv2.GaussianBlur(grayImage, (7, 7), 3)

    threshImage = cv2.adaptiveThreshold(imageBlurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    invertThreshImage = cv2.bitwise_not(threshImage)

    if debug:
        cv2.imshow("Thresh Image Of Puzzle", invertThreshImage)
        cv2.waitKey(0)

    contours = cv2.findContours(invertThreshImage.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = imutils.grab_contours(contours)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)

    puzzleContour = None

    for conts in contours:
        perimeter = cv2.arcLength(conts, True)
        approx = cv2.approxPolyDP(conts, 0.02*perimeter, True)

        if len(approx) == 4:
            puzzleContour = approx
            break

    if puzzleContour is None:
        raise Exception(("The Sudoku puzzle outline could not be found." 
                        "Try debugging your threshold values and contours steps"))

    if debug:
        output = image.copy()
        cv2.drawContours(output, [puzzleContour], -1, (0, 255, 0), 2)
        cv2.imshow("Puzzle Outline window", output)
        cv2.waitKey(0)

    actualPuzzle = four_point_transform(image, puzzleContour.reshape(4, 2))
    warped = four_point_transform(grayImage, puzzleContour.reshape(4, 2))

    if debug:
        cv2.imshow("Birds Eye View of the Puzzle", actualPuzzle)
        cv2.waitKey(0)

    return(actualPuzzle, warped)

def digit_extraction(cell, debug=False):

    threshCell = cv2.threshold(cell, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
    threshCell = clear_border(threshCell)

    if debug:
        cv2.imshow("Thresh Image of Cell", threshCell)
        cv2.waitKey(0)

    cellContours = cv2.findContours(threshCell.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cellContours = imutils.grab_contours(cellContours)

    if len(cellContours) == 0:
        return None

    largestContour = max(cellContours, key=cv2.contourArea)
    contourMask = np.zeros(threshCell.shape, dtype="uint8")
    cv2.drawContours(contourMask, [largestContour], -1, 255, -1)

    (h, w) = threshCell.shape
    percentAreaFilled = cv2.countNonZero(contourMask)/float(w*h)

    if percentAreaFilled < 0.03:
        return None

    digit = cv2.bitwise_and(threshCell, threshCell, mask=contourMask)

    if debug:
        cv2.imshow("Digit in Cell", digit)
        cv2.waitKey(0)

    return digit
