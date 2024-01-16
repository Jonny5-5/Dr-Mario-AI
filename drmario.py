import random
import time as time
import numpy as np
import cv2
from mss import mss
from PIL import Image
import win32con
import win32gui
import win32api
import win32com.client
import queue
import threading
import time


# Configuration variables and global variables
TIME_BETWEEN_KEYSTROKES = 0.075
monitor_bounds = {"top": 64, "left": 0, "width": 512, "height": 448}
game_width = 8
game_height = 16
PILL_START_COL = 3
# Extended keys and their corresponding scancodes
KEY_ENTER = win32con.VK_RETURN  # Maybe??
KEY_UP = win32con.VK_UP
KEY_DOWN = win32con.VK_DOWN
KEY_LEFT = win32con.VK_LEFT
KEY_RIGHT = win32con.VK_RIGHT
KEY_ROTATE_CLOCKWISE = ord("S")
KEY_ROTATE_COUNTER_CLOCKWISE = ord("F")
# print(KEY_ROTATE_COUNTER_CLOCKWISE)
# exit()
# Some color constants
NO_COLOR = 0
RED_COLOR = 1
BLUE_COLOR = 2
YELLOW_COLOR = 3
RED_CHAR = "R"
BLUE_CHAR = "B"
YELLOW_CHAR = "Y"
HORIZONTAL_NO_FLIP = 0
HORIZONTAL_FLIPPED = 1
VERTICAL_FLIPPED = 2  # Counter-clockwise 90 degrees
VERTICAL_NO_FLIP = 3  # Clockwise 90 degrees
VIRUS_POP_SCORE = 5


class Pill:
    def __init__(self, lColor, rColor):
        self.leftColor = lColor
        self.rightColor = rColor

    def __eq__(self, other):
        if isinstance(other, Pill):
            return (
                self.leftColor == other.leftColor
                and self.rightColor == other.rightColor
            )
        return False


class Location:
    def __init__(self, lLoc, rLoc):
        self.leftLocation = lLoc
        self.rightLocaiton = rLoc


class MyWindow:
    def __init__(self):
        pass  # empty init function

    def print_window_titles(hwnd, window_titles):
        window_titles.append(win32gui.GetWindowText(hwnd))

    def getWindow():
        window_titles = []
        win32gui.EnumWindows(MyWindow.print_window_titles, window_titles)
        # print(window_titles)
        for title in window_titles:
            if "Snes9X" in title:
                print("\nFound the Snes9X window!", title)
                WINDOW_NAME = title
        hwnd = win32gui.FindWindow(None, WINDOW_NAME)
        assert hwnd != 0, "hwnd is 0"
        return hwnd


class MyKeyboard:
    key_queue = queue.Queue()
    isRunning = False

    def __init__(self) -> None:
        self.isPressingDown = False

    # def add_key(self, key):
    #     self.key_queue.put(key)
    def send_piece_down(self):
        if self.isPressingDown:
            return
        else:
            self.isPressingDown = True
            win32api.keybd_event(
                KEY_DOWN,
                0,
            )
            time.sleep(TIME_BETWEEN_KEYSTROKES)

    def stop_send_piece_down(self):
        win32api.keybd_event(KEY_DOWN, 0, win32con.KEYEVENTF_KEYUP)
        self.isPressingDown = False

    def send_key(self, key):
        if key:
            print("pressing Key: ", self.printKey(key))
            if key == KEY_ROTATE_CLOCKWISE or key == KEY_ROTATE_COUNTER_CLOCKWISE:
                win32api.keybd_event(
                    key,
                    0,
                )
                time.sleep(TIME_BETWEEN_KEYSTROKES)
                win32api.keybd_event(key, 0, win32con.KEYEVENTF_KEYUP)
                time.sleep(TIME_BETWEEN_KEYSTROKES)
            else:
                win32api.keybd_event(key, 0, win32con.KEYEVENTF_EXTENDEDKEY, 0)
                time.sleep(TIME_BETWEEN_KEYSTROKES)
                win32api.keybd_event(key, 0, win32con.KEYEVENTF_KEYUP)
                time.sleep(TIME_BETWEEN_KEYSTROKES)

    def stop(self):
        self.isRunning = False
        self.key_queue.queue.clear()

    def runSystemsCheck(self):
        for i in range(10):
            self.send_key(KEY_RIGHT)
        for i in range(10):
            self.send_key(KEY_LEFT)
        self.send_key(KEY_DOWN)
        self.send_key(KEY_DOWN)
        self.send_key(KEY_UP)
        self.send_key(KEY_UP)

    def printKey(self, key):
        if key == KEY_ENTER:
            return "sending key: KEY_ENTER"
        elif key == KEY_UP:
            return "sending key: KEY_UP"
        elif key == KEY_DOWN:
            return "sending key: KEY_DOWN"
        elif key == KEY_LEFT:
            return "sending key: KEY_LEFT"
        elif key == KEY_RIGHT:
            return "sending key: KEY_RIGHT"
        elif key == KEY_ROTATE_CLOCKWISE:
            return "sending key: KEY_ROTATE_CLOCKWISE"
        elif key == KEY_ROTATE_COUNTER_CLOCKWISE:
            return "sending key: KEY_ROTATE_COUNTER_CLOCKWISE"

    def run(self):
        if self.isRunning:
            return
        self.isRunning = True
        print("Keyboard is running...")

        def send_key(key):
            if key:
                print("pressing Key: ", self.printKey(key))
                if key == KEY_ROTATE_CLOCKWISE or key == KEY_ROTATE_COUNTER_CLOCKWISE:
                    win32api.keybd_event(
                        key,
                        0,
                    )
                    time.sleep(TIME_BETWEEN_KEYSTROKES)
                    win32api.keybd_event(key, 0, win32con.KEYEVENTF_KEYUP)
                    time.sleep(TIME_BETWEEN_KEYSTROKES)
                else:
                    win32api.keybd_event(key, 0, win32con.KEYEVENTF_EXTENDEDKEY, 0)
                    time.sleep(TIME_BETWEEN_KEYSTROKES)
                    win32api.keybd_event(key, 0, win32con.KEYEVENTF_KEYUP)
                    time.sleep(TIME_BETWEEN_KEYSTROKES)

        while self.isRunning:
            try:
                key = self.key_queue.get(block=False)
                send_key(key)
            except queue.Empty:
                pass


class MyScreenRecorder:
    sct = None

    def __init__(self) -> None:
        self.sct = mss()
        pass

    def print_screenshot(self, screenshot_array):
        # print(screenshot.shape)
        with open("screenshot.txt", "w") as txt_file:
            for line in screenshot_array:
                for x in line:
                    txt_file.write("{:5d}".format(x))
                txt_file.write("\n")
        # Save the screenshot as a PNG file
        cv2.imwrite("screenshot.png", screenshot_array)

    def show_screen(self, screenshot_array):
        cv2.imshow("Computer Vision", screenshot_array)
        print("avg FPS {}".format(1 / (time.time() - loop_time)))
        loop_time = time.time()
        if cv2.waitKey(1) == ord("q"):
            cv2.destroyAllWindows()

    # Returns a pixel array bounded by the monitor
    def get_screen(self):
        # There is 0 optimization for this code here...
        # Literally every line that I type will be processed literally.
        # It's slow as molasses.
        screen = self.sct.grab(monitor_bounds)
        # pixels = screen.pixels
        screenshot_array = np.array(screen)
        # screenshot_array = cv2.cvtColor(screenshot_array, cv2.COLOR_RGB2GRAY)
        # self.show_screen(screenshot_array)

        # self.print_screenshot(screenshot_array)

        return screenshot_array

    def get_debug_screen(self):
        screen = cv2.imread("screenshot.png")
        screenshot_array = np.array(screen)
        return screenshot_array


class MyAI:
    def __init__(self):
        self.recorder = MyScreenRecorder()
        pass

    def getColorFromKernel(self, kernel):
        MIN_BLUE_AVG = 75
        MIN_RED_AVG = 100
        MIN_YELLOW_AVG = 150
        MIN_GREEN_AVG = 20
        blueAvg = int(np.average(kernel[:, :, 0]))
        redAvg = int(np.average(kernel[:, :, 2]))
        greenAvg = int(np.average(kernel[:, :, 1]))
        yellowAvg = int((redAvg + greenAvg) / 1.5)
        avgs = [redAvg, blueAvg, yellowAvg]
        maxAvg = max(avgs)

        if maxAvg == yellowAvg and yellowAvg > MIN_YELLOW_AVG:
            return YELLOW_COLOR
        elif maxAvg == redAvg and redAvg > MIN_RED_AVG:
            return RED_COLOR
        # elif maxAvg == blueAvg and blueAvg > MIN_BLUE_AVG and greenAvg > MIN_GREEN_AVG:
        elif greenAvg > MIN_GREEN_AVG:
            return BLUE_COLOR
        else:
            return NO_COLOR

    def getNextPill(self, screenshot):
        NEXT_ROW = 104
        NEXT_LEFT_COL = 385
        NEXT_RIGHT_COL = 397
        leftKernel = screenshot[
            NEXT_ROW - 3 : NEXT_ROW + 3, NEXT_LEFT_COL - 3 : NEXT_LEFT_COL + 3
        ]
        rightKernel = screenshot[
            NEXT_ROW - 3 : NEXT_ROW + 3, NEXT_RIGHT_COL - 3 : NEXT_RIGHT_COL + 3
        ]
        leftColor = self.getColorFromKernel(leftKernel)
        rightColor = self.getColorFromKernel(rightKernel)
        return Pill(leftColor, rightColor)

    # Runs a simple simulation on the given board and returns the number of pieces popped
    def simulateBoard(self, board):
        boardCopy = np.copy(board)

        # Simulate the rows and then transpose and repeat
        def doThing():
            rowIndex = -1
            for row in board:
                rowIndex += 1
                i = 0
                while i < len(row):
                    count = 1
                    j = i + 1
                    while j < len(row) and row[j] == row[i] and row[j] != 0:
                        count += 1
                        j += 1
                    if count >= 4:
                        for k in range(i, j):
                            boardCopy[rowIndex][k] += 10
                    i = j

        doThing()
        board = board.T
        boardCopy = boardCopy.T
        doThing()
        board = board.T
        boardCopy = boardCopy.T
        score = 0
        rowIndex = 0
        colIndex = 0
        for row in boardCopy:
            colIndex = 0
            for val in row:
                if val > 10:
                    score += 1
                if self.virusMatrix[rowIndex][colIndex] != 0:
                    score += VIRUS_POP_SCORE
                colIndex += 1
            rowIndex += 1
        return score

    def getViruses(self, screenshot):
        self.virusMatrix = self.getMatrix(screenshot)

    def proxSum(self, matrix):
        splashSum = 0
        for row in range(len(matrix) - 1):
            for col in range(len(matrix[0]) - 1):
                if row == 0:
                    row += 1
                if col == 0:
                    col += 1
                # Splash out and make a summation
                val = matrix[row][col]
                if val == 0:
                    continue
                if matrix[row + 1][col] != 0:
                    if val == matrix[row + 1][col]:
                        splashSum += 1
                    else:
                        splashSum -= 1
                if matrix[row - 1][col] != 0:
                    if val == matrix[row - 1][col]:
                        splashSum += 1
                    else:
                        splashSum -= 1
                if matrix[row][col + 1] != 0:
                    if val == matrix[row][col + 1]:
                        splashSum += 1
                    else:
                        splashSum -= 1
                if matrix[row][col - 1] != 0:
                    if val == matrix[row][col - 1]:
                        splashSum += 1
                    else:
                        splashSum -= 1

        rowSum = 0
        for row in matrix:
            # Sum along the rows
            prev = row[0]
            for val in row[1:]:
                if val == prev and val != 0:
                    rowSum += 1
                prev = val

        colSum = 0
        for i in range(len(matrix[0])):
            # Sum along the columns
            col = matrix[:, i]
            prev = col[0]
            for val in col[1:]:
                if prev == val and val != 0:
                    colSum += 1
                prev = val

        return splashSum + rowSum + colSum

    def getProximityPosition(self, matrix, nextPill):
        if not isinstance(nextPill, Pill):
            print("You can't do that")
        # Rotate all the pills and "insert" into the matrix
        # Run a sum on each row/column and see which location has the most adjacent pieces
        maxScore = -9999999999
        maxPosition = ()
        maxOrientation = HORIZONTAL_NO_FLIP
        orientations = [
            HORIZONTAL_NO_FLIP,
            HORIZONTAL_FLIPPED,
            VERTICAL_FLIPPED,
            VERTICAL_NO_FLIP,
        ]
        for orientation in orientations:
            if orientation == HORIZONTAL_NO_FLIP:
                tempPiece = nextPill
            elif orientation == HORIZONTAL_FLIPPED:
                tempPiece = Pill(nextPill.rightColor, nextPill.leftColor)
            elif orientation == VERTICAL_FLIPPED:
                tempPiece = Pill(nextPill.rightColor, nextPill.leftColor)
            elif orientation == VERTICAL_NO_FLIP:
                tempPiece = nextPill

            # This is only for HORIZONTAL pieces
            if orientation == HORIZONTAL_NO_FLIP or orientation == HORIZONTAL_FLIPPED:
                for col in range(len(matrix[0]) - 1):
                    row = 0
                    # Find the first row that has two 0 spots in it
                    while (
                        row < len(matrix)
                        and matrix[row][col] == 0
                        and matrix[row][col + 1] == 0
                    ):
                        row += 1
                    row -= 1
                    # "Insert" the piece
                    temp = np.copy(matrix)
                    temp[row][col] = tempPiece.leftColor
                    temp[row][col + 1] = tempPiece.rightColor
                    score = self.proxSum(temp)
                    if score > maxScore:
                        # print(temp)
                        # print("score ^ =", score)
                        maxScore = score
                        maxPosition = (row, col)
                        maxOrientation = orientation

            # This is only for VERTICAL pieces
            if orientation == VERTICAL_NO_FLIP or orientation == VERTICAL_FLIPPED:
                for col in range(len(matrix[0])):
                    row = 0
                    while row < len(matrix) and matrix[row][col] == 0:
                        row += 1
                    row -= 1
                    # "Insert" the piece
                    temp = np.copy(matrix)
                    temp[row][col] = tempPiece.leftColor
                    temp[row - 1][col] = tempPiece.rightColor
                    score = self.proxSum(temp)
                    # if (col == 5): print("\n", temp, "\n", row, col, orientation, score, maxScore)
                    if score > maxScore:
                        # print(temp)
                        # print("score ^ =", score)
                        maxScore = score
                        maxPosition = (col, row)
                        maxOrientation = orientation

        return maxPosition, maxOrientation

    def findOptimalPosition(self, matrix, nextPiece):
        matrix[0][3] = 0
        matrix[0][4] = 0  # Get rid of that pesky code that
        if not isinstance(nextPiece, Pill):
            raise TypeError("Pass in a piece")
        popOffset = self.simulateBoard(matrix)
        if popOffset > 0:
            print("WARNING: The board already has pops")
        optimalPosiiton = ()
        maxPops = 0
        optimalOrientation = HORIZONTAL_NO_FLIP
        orientations = [
            HORIZONTAL_NO_FLIP,
            HORIZONTAL_FLIPPED,
            VERTICAL_FLIPPED,
            VERTICAL_NO_FLIP,
        ]
        for orientation in orientations:
            if orientation == HORIZONTAL_NO_FLIP:
                tempPiece = nextPiece
            elif orientation == HORIZONTAL_FLIPPED:
                tempPiece = Pill(nextPiece.rightColor, nextPiece.leftColor)
            elif orientation == VERTICAL_FLIPPED:
                tempPiece = Pill(nextPiece.rightColor, nextPiece.leftColor)
            elif orientation == VERTICAL_NO_FLIP:
                tempPiece = nextPiece

            # Horizontal pieces
            if orientation == HORIZONTAL_NO_FLIP or orientation == HORIZONTAL_FLIPPED:
                for col in range(len(matrix[0]) - 1):
                    row = 0
                    while (
                        row >= 0
                        and row < len(matrix)
                        and matrix[row][col] == 0
                        and matrix[row][col + 1] == 0
                    ):
                        row += 1
                    if row != 0:
                        row -= 1
                    matrixTemp = np.copy(matrix)
                    matrixTemp[row][col] = tempPiece.leftColor
                    matrixTemp[row][col + 1] = tempPiece.rightColor
                    pops = self.simulateBoard(matrixTemp)
                    # print("row=", row, "col=", col, "pops=", pops, "max=", maxPops, "orientation", orientation)
                    if pops - popOffset > maxPops:
                        print(
                            "pops=",
                            pops,
                            "max=",
                            maxPops,
                            "row=",
                            row,
                            "col=",
                            col,
                            "orientation",
                            orientation,
                        )
                        print(matrixTemp)
                        maxPops = pops
                        optimalPosiiton = (row, col)
                        optimalOrientation = orientation
            else:
                for col in range(len(matrix[0])):
                    row = 0
                    while row >= 0 and row < len(matrix) and matrix[row][col] == 0:
                        row += 1
                    if row != 0:
                        row -= 1
                    matrixTemp = np.copy(matrix)
                    matrixTemp[row][col] = tempPiece.leftColor
                    matrixTemp[row - 1][col] = tempPiece.rightColor
                    pops = self.simulateBoard(matrixTemp)
                    # print("row=", row, "col=", col, "pops=", pops, "max=", maxPops, "orientation", orientation)
                    if pops - popOffset > maxPops:
                        print(
                            "pops=",
                            pops,
                            "max=",
                            maxPops,
                            "row=",
                            row,
                            "col=",
                            col,
                            "orientation",
                            orientation,
                        )
                        print(matrixTemp)
                        maxPops = pops
                        optimalPosiiton = (col, row)
                        optimalOrientation = orientation

        # This line of code has the AI choose to pop something if it possibly can.
        if maxPops == 0:
            optimalPosiiton, optimalOrientation = self.getProximityPosition(
                matrix, nextPiece
            )
        if (
            optimalOrientation == VERTICAL_FLIPPED
            or optimalOrientation == VERTICAL_NO_FLIP
        ):
            optimalPosiiton = optimalPosiiton[::-1]
        return optimalPosiiton, optimalOrientation

    def getInputsToPosition(self, position, orientation):
        if position == ():
            return []
        inputs = []
        if orientation == HORIZONTAL_FLIPPED:
            inputs.append(KEY_ROTATE_CLOCKWISE)
            inputs.append(KEY_ROTATE_CLOCKWISE)
        elif orientation == VERTICAL_FLIPPED:
            inputs.append(KEY_ROTATE_CLOCKWISE)
        elif orientation == VERTICAL_NO_FLIP:
            inputs.append(KEY_ROTATE_COUNTER_CLOCKWISE)

        col = PILL_START_COL
        while col != position[1]:
            if col < position[1]:
                inputs.append(KEY_RIGHT)
                col += 1
            else:
                inputs.append(KEY_LEFT)
                col -= 1

        return inputs

    def getInputs(self, screenshot, nextPill):
        matrix = self.getMatrix(screenshot)
        # print("We are getting inputs for the following matrix\n", matrix)
        inputs = None

        position, orientation = self.findOptimalPosition(
            matrix, nextPill
        )  # Returns a Location
        # print("pos, orien", position, orientation)
        # Decode that location into a list of inputs
        inputs = self.getInputsToPosition(position, orientation)

        if inputs is not None:
            return inputs
        else:
            return []

    # Make sure that I only get the screenshot once and then use the matrix
    # for the rest of the calculations so that it doesn't slow everything
    # down like crazy.
    def getMatrix(self, screenshot):
        matrix = np.zeros((game_height, game_width)).astype(int)
        px_per_row = 16
        px_per_col = 16
        row_offset = 10
        col_offset = 1
        start_row = 153
        start_col = 192
        for i in range(game_height):
            for j in range(game_width):
                row = i * px_per_row + row_offset + start_row
                col = j * px_per_col + col_offset + start_col
                kernel = screenshot[row : row + 5, col : col + 10]
                color = self.getColorFromKernel(kernel)
                matrix[i][j] = color

        return matrix


myWindow = MyWindow
hwnd = myWindow.getWindow()
print("HWND =", hwnd)
win32gui.SetForegroundWindow(hwnd)  # Set the window in focus

# Make an instance of a Queue
keyboard = MyKeyboard()
# Start the while loop in a separate thread
# thread = threading.Thread(target=keyboard.run)
# thread.start()

ai = MyAI()


def printMatrix(matrix):
    with open("martix.txt", "w") as txt_file:
        for line in matrix:
            for x in line:
                txt_file.write("{:3d}".format(x))
            txt_file.write("\n")


def printInputs(inputs):
    for key in inputs:
        if key == KEY_ENTER:
            print("KEY_ENTER")
        elif key == KEY_UP:
            print("KEY_UP")
        elif key == KEY_DOWN:
            print("KEY_DOWN")
        elif key == KEY_LEFT:
            print("KEY_LEFT")
        elif key == KEY_RIGHT:
            print("KEY_RIGHT")
        elif key == KEY_ROTATE_CLOCKWISE:
            print("KEY_ROTATE_CLOCKWISE")
        elif key == KEY_ROTATE_COUNTER_CLOCKWISE:
            print("KEY_ROTATE_COUNTER_CLOCKWISE")


# This is the real game loop here:
wsh = win32com.client.Dispatch("WScript.Shell")
isFirstLoop = True
while True:
    if win32gui.GetForegroundWindow() == hwnd:
        # Run a systems check
        if isFirstLoop:
            print("running system check")
            keyboard.runSystemsCheck()
            keyboard.send_key(KEY_ENTER)
            # print("sleeping")
            time.sleep(2)
            print("Starting for real!")

            screenshot = ai.recorder.get_screen()
            print("getting virus locations")
            ai.getViruses(screenshot)
            print("Virus matrix is\n", ai.virusMatrix)

            isFirstLoop = False

        # Wait for him to load a pill
        screenshot = ai.recorder.get_screen()
        while win32gui.GetForegroundWindow() == hwnd and ai.getNextPill(
            screenshot
        ) == Pill(NO_COLOR, NO_COLOR):
            screenshot = ai.recorder.get_screen()

        pill = ai.getNextPill(screenshot)

        # While he is still holding the same pill
        while pill == ai.getNextPill(screenshot):
            screenshot = ai.recorder.get_screen()
            keyboard.send_piece_down()
        keyboard.stop_send_piece_down()

        # He just threw it, Wait for him to load the next pill
        while ai.getNextPill(screenshot) == Pill(NO_COLOR, NO_COLOR):
            screenshot = ai.recorder.get_screen()

        print("Pill is ", pill.leftColor, pill.rightColor)
        inputs = ai.getInputs(screenshot, pill)

        # if inputs == []:
        #     position = (0, random.randint(0, game_width - 2))
        #     inputs = ai.getInputsToPosition(position, HORIZONTAL_NO_FLIP)
        #     print("-----RANDOM INPUTS to---", position)

        # while not keyboard.key_queue.empty:
        #     print("WARNING! waiting for inputs to finish")
        #     pass

        # printInputs(inputs)
        time.sleep(0.4)  # Wait for just a split second so we don't miss inputs
        for input in inputs:
            keyboard.send_key(input)

        print("sent inputs")
        print()
        # time.sleep(sleep_time_send_inputs)
    else:
        keyboard.stop()
        # thread.join()
        break

print("Done.")
exit()
