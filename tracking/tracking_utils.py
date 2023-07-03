import cv2
import numpy as np
import os
import Jetson.GPIO as GPIO
import time


def box(img, x, y, h, w, id, color=(0, 0, 255), thickness=1, printXY=False):
    """
    Recibe una imagen, las coordenadas para dibujar la caja y el numero de id.
    Devuelve la imagen con la caja dibujada y el numero de id.
    Funciona tanto cuando los valores de x, y, h, w estan expresados en pixels
    como cuando estan normalizados entre 0 y 1.
    (x1, y1) son los puntos de la esquina superior izquierda del rectangulo
    (x2, y2) son los puntos de la esquina inferior derecha del rectangulo
    x1,y1 ------
    |          |
    |          |
    |          |
    --------x2,y2
    """

    # para de-normalizar y obtener los pixels si hace falta
    if x <= 1 and y <= 1 and h <= 1 and w <= 1:
        imgHeight, imgWidth, _ = img.shape
    else:
        imgHeight, imgWidth = 1, 1

    x1 = int((x - w / 2) * imgWidth)
    y1 = int((y - h / 2) * imgHeight)
    x2 = int((x + w / 2) * imgWidth)
    y2 = int((y + h / 2) * imgHeight)

    if printXY:
        print(x1, y1, x2, y2)

    # box
    cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)
    # id
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(
        img,
        str(int(id)),
        (x1 + 5, y1 + 15),
        fontFace=font,
        fontScale=0.5,
        color=color,
        thickness=thickness + 1,
    )

    return img


def boxXyxy(img, x1, y1, x2, y2, id="", color=(0, 0, 255), thickness=1, printXY=False):
    """
    Recibe una imagen, las coordenadas para dibujar la caja y el numero de id.
    Devuelve la imagen con la caja dibujada y el numero de id.
    Funciona tanto cuando los valores de x1, y1, x2, y2 estan expresados en pixels
    como cuando estan normalizados entre 0 y 1.
    (x1, y1) son los puntos de la esquina superior izquierda del rectangulo
    (x2, y2) son los puntos de la esquina inferior derecha del rectangulo
    x1,y1 ------
    |          |
    |          |
    |          |
    --------x2,y2
    """

    # para de-normalizar y obtener los pixels si hace falta
    if x1 <= 1 and y1 <= 1 and x2 <= 1 and y2 <= 1:
        imgHeight, imgWidth, _ = img.shape
    else:
        imgHeight, imgWidth = 1, 1

    if printXY:
        print(x1, y1, x2, y2)

    # box
    cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), color, thickness)
    # id
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(
        img,
        id if isinstance(id, str) else str(int(id)),
        (int(x1) + 5, int(y1) + 15),
        fontFace=font,
        fontScale=0.5,
        color=color,
        thickness=thickness + 1,
    )

    return img


def detections2Sort(predictionsPath, outPath, outName, imgShape):
    """
    The function takes the output files of the prediction algorithm and converts them into a
    single file compatible with the input of the SORT tracking algorithm.
    Params:
    * predictionsPath: path to the folder with the prediction files (output of the prediction algorithm).
    * outPath: path to the folder where the SORT formatted output will be.
    * outName: name + extension of the output file in SORT format.
    * imgShape: tuple with the shape of the images. (width, height)
    """

    width, height = imgShape

    # get only files
    file_list = [
        f
        for f in os.listdir(predictionsPath)
        if os.path.isfile(os.path.join(predictionsPath, f))
    ]
    file_list = sorted(
        file_list, key=lambda x: int(os.path.splitext(x)[0].split("_")[-1])
    )
    i = 0
    out_csv = ""

    for filename in file_list:
        imgNumber = os.path.splitext(filename)[0].split("_")[-1]
        np_line = np.loadtxt(predictionsPath + "/" + filename, delimiter=" ")
        np_line = np.reshape(
            np_line, (-1, 7)
        )  # reshape for files with only one detection. 7 is the number of columns of the detection files
        for j in range(len(np_line)):
            line = (
                str(imgNumber)
                + ",-1,"
                + str(np_line[j, 1] * width)
                + ","
                + str(np_line[j, 2] * height)
                + ","
                + str(np_line[j, 3] * width)
                + ","
                + str(np_line[j, 4] * height)
                + ","
                + str(np_line[j, 5])
                + ",-1,-1,-1"
            )
            out_csv = out_csv + line + "\r\n"

    # Creacion de carpeta
    if not os.path.isdir(outPath):
        os.makedirs(outPath)

    # Borrar archivo si existe
    if os.path.exists(outPath + "/" + outName):
        os.remove(outPath + "/" + outName)
        print("Created folder: ", outPath + "/" + outName)

    # Escribir el archivo de salida
    out_file = open(outPath + "/" + outName, "w")
    out_file.write(out_csv)
    out_file.close()


# split list "l" into chunks of size "n"
def divideChunks(l, n):
    for i in range(0, len(l), n):
        yield l[i : i + n]


def getImgShape(imgBasePath):
    """
    Opens one image from a folder to look for it's shape.
    The way this is done is AWFUL, but I don't to think of a better way today :(
    Returns:
    A tuple with the images width and height: (width, height)
    """

    # open an image to find out the size of the frames (HORRIBLE way of doing it)
    for _, imgName in enumerate(sorted(os.listdir(imgBasePath)), start=1):
        imgShape = cv2.imread(os.path.join(imgBasePath, imgName)).shape
        break
    return (imgShape[1], imgShape[0])


def tracking2Video(outputVideo, fps, imgsPath, detectionsPath):
    """
    Returns a video with the detections and the tracked objects with their IDs.
    Params:
    * outputVideo: path + name + extension of the output video.
    * fps: frames per second of the output video.
    * imgsPath: path to the image that will make up the video.
    * detectionsPath: path to the output of the SORT algorithm.
    """
    detections = np.loadtxt(detectionsPath, delimiter=",")
    frameShape = getImgShape(imgsPath)
    video = cv2.VideoWriter(
        outputVideo, cv2.VideoWriter_fourcc(*"DIVX"), fps, frameShape
    )

    # loop through images
    for imgNumber, imgName in enumerate(
        sorted(
            os.listdir(imgsPath),
            key=lambda x: int(os.path.splitext(x)[0].split("_")[-1]),
        ),
        start=1,
    ):
        imgNumber = int(os.path.splitext(imgName)[0].split("_")[-1])
        img = cv2.imread(os.path.join(imgsPath, imgName))
        # loop through SORT detections
        for detection in detections:
            # unpack detection values
            detImgNumber, detId, detX, detY, detW, detH, *_ = detection

            if imgNumber == detImgNumber:
                img = box(img, x=detX, y=detY, h=detH, w=detW, id=detId)

        # write frames into video
        video.write(img)


class DrawLaneCoordinates(object):
    def __init__(self, img, color, thickness):
        self.coordinates = []
        self.image = img
        self.color = color
        self.thickness = thickness
        self.windowName = "DrawLanes"

        cv2.namedWindow(self.windowName)
        cv2.imshow(self.windowName, self.image)
        cv2.setMouseCallback(self.windowName, self.extractLaneCoordinates)

        key = cv2.waitKey(0)
        # wait for ESC key to exit
        if key == 27:
            cv2.destroyAllWindows()

    def extractLaneCoordinates(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN and len(self.coordinates) < 4:
            self.coordinates.append((x, y))
            cv2.circle(self.image, (x, y), 4, (self.color), -1)
            cv2.imshow(self.windowName, self.image)


class Buffer(object):
    def __init__(self, capacity=50):
        self.capacity = capacity  # indicates the maximum size of the buffer
        self.actualSize = 0  # indicates the actual size of the buffer
        self.data = (
            []
        )  # List of dictionaries {"isOccupied": isOccupied, "vehicleId": id,"timeStamp": timeStamp}

    def Enqueue(self, isOccupied, vehicleID, timeStamp):
        if len(self.data) == self.capacity:
            raise Exception("Buffer full. First Dequeue")

        else:
            self.data.append(
                {
                    "isOccupied": isOccupied,
                    "vehicleId": vehicleID,
                    "timeStamp": timeStamp,
                }
            )
            self.actualSize += 1

    def Dequeue(self):
        if len(self.data) == 0:
            print("Buffer empty. First Enqueue some item")
            return None

        else:
            self.actualSize -= 1
            return self.data.pop(0)

    def isFull(self):
        if len(self.data) == self.capacity:
            return True

        else:
            return False

    def SwitchValuesToLastDifferent(self, valueToSet, idToSet):
        if len(self.data) == 0:
            return

        valueToSwitch = self.data[-1]["isOccupied"]

        for index in range(len(self.data) - 1):
            if self.data[-1 - index]["isOccupied"] == valueToSwitch:
                self.data[-1 - index]["isOccupied"] = valueToSet
                self.data[-1 - index]["vehicleId"] = idToSet
                index += 1
            else:
                break

        return


class Lane(object):
    def __init__(
        self, coordinates, laneId, color=(255, 0, 0), thickness=2, buff_size=20
    ):
        # plot and coordinates atributes
        self.color = color  # lane's color in the plot
        self.thickness = thickness  # lane's thickness in the plot
        self.coordinates = coordinates  # lane's coordinates in the image/video
        # signal atributes
        self.laneId = laneId  # lane's unique ID
        self.buffer = Buffer(buff_size)
        self.vehicleList = set()  # set with uniques IDs
        self.lastDetectedId = -1  # save the last value real vehicle (!= '-1')
        self.lastIsOccupied = False  # save the last value flag of Occupied

    def getCoordinates(self):
        return self.coordinates

    def getLaneId(self):
        return self.laneId

    def popFirstValue(self):
        return self.buffer.Dequeue()

    def getVehicleListCount(self):
        return len(self.vehicleList)

    def updateIsOccupied(self, isOccupied, id, timeStamp):
        self.vehicleList.add(id)  # add vehicleID to the list
        self.buffer.Enqueue(isOccupied, id, timeStamp)  # add sample to the Signal
        self.lastIsOccupied = isOccupied  # save the flag
        if id != -1:
            self.lastDetectedId = id  # save the last value

    def getLastDetectedId(self):
        return self.lastDetectedId

    def getLastIsOccupied(self):
        return self.lastIsOccupied

    def correctBackwards(self, id):
        self.buffer.SwitchValuesToLastDifferent(True, id)


gPin1 = 7
gPin2 = 11
gPin3 = 12
gPin4 = 13
pinArray = [gPin1, gPin2, gPin3, gPin4]


class VirtualLoop(Lane):
    def __init__(
        self, coordinates, virtualLoopId, color=(255, 0, 0), thickness=2, buff_size=20
    ):
        Lane.__init__(
            self,
            coordinates=coordinates,
            laneId=virtualLoopId,
            color=color,
            thickness=thickness,
        )
        self.pin = pinArray[virtualLoopId - 1]
        GPIO.setmode(GPIO.BOARD)  # Set GPIO Mode (for pins enumerations)
        GPIO.setup(self.pin, GPIO.OUT)  # Set PIN as Output

    def popFirstValue(self):
        retValue = self.buffer.Dequeue()

        GPIO.output(self.pin, GPIO.HIGH) if retValue["isOccupied"] else GPIO.output(
            self.pin, GPIO.LOW
        )

        return retValue
