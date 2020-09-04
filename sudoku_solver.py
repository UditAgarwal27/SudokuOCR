from PuzzleRecogniser.PuzzleRecogniser import digit_extraction, find_puzzle
from keras.preprocessing.image import img_to_array
from keras.models import load_model
from sudoku import Sudoku
import numpy as np
import argparse
import imutils
import cv2

ap = argparse.ArgumentParser()
ap.add_argument("-m", "--model", required=True, help="path to trained digit classifier")
ap.add_argument("-i", "--image", required=True, help="path to input Sudoku puzzle image")
ap.add_argument("-d", "--debug", type=int, default=-1,
                help="whether or not we are visualizing each step of the pipeline")
args = vars(ap.parse_args())

model = load_model(args["model"])

image = cv2.imread(args["image"])
image = imutils.resize(image, width=600)

(puzzleImage, warped) = find_puzzle(image, debug=args["debug"] > 0)

puzzleBoard = np.zeros((9, 9), dtype=int)

stepX = warped.shape[1]//9
stepY = warped.shape[0]//9

cellLocation = []

for y in range(0, 9):
    row = []

    for x in range(0, 9):
        startX = x*stepX
        startY = y*stepY
        endX = (x+1)*stepX
        endY = (y+1)*stepY

        row.append((startX, startY, endX, endY))

        cell = warped[startY:endY, startX:endX]
        digit = digit_extraction(cell, debug=args["debug"] > 0)

        if digit is not None:
            regionOfInterest = cv2.resize(digit, (28, 28))
            regionOfInterest = regionOfInterest.astype("float")/255.0
            regionOfInterest = img_to_array(regionOfInterest)
            regionOfInterest = np.expand_dims(regionOfInterest, axis=0)

            predictedDigit = model.predict(regionOfInterest).argmax(axis=1)[0]
            puzzleBoard[y, x] = predictedDigit

    cellLocation.append(row)

puzzle = Sudoku(3, 3, board=puzzleBoard.tolist())
puzzle.show()

puzzleSolution = puzzle.solve()
puzzleSolution.show_full()

for(cellRow, boardRow) in zip(cellLocation, puzzleSolution.board):
    for(box, digit) in zip(cellRow, boardRow):
        Xstart, Ystart, Xend, Yend = box

        textX = int((Xend - Xstart)* 0.33)
        textY = int((Yend - Ystart)* -0.2)
        textX += Xstart
        textY += Yend

        cv2.putText(puzzleImage, str(digit), (textX, textY), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)

cv2.imshow("Sudoku Result Image", puzzleImage)
cv2.waitKey(0)

