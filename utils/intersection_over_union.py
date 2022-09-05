import csv
from json.encoder import INFINITY
from tkinter.messagebox import NO
import cv2 as cv
import logging
import numpy as np


def yolov5ToXY(row, img):
    # yolov5: <object-class> <x> <y> <width> <height>

    widthToAdd = row[3] / 2
    heightToAdd = row[4] / 2

    xMin = int((row[1] - widthToAdd) * img.shape[1])
    xMax = int((row[1] + widthToAdd) * img.shape[1])
    yMin = int((row[2] - heightToAdd) * img.shape[0])
    yMax = int((row[2] + heightToAdd) * img.shape[0])

    # XY {<object-class> <xMin> <xMax> <yMin> <yMax>}
    # CUIDADO: el (0, 0) de coordenadas es la esquina superior izquierda
    # Se cuenta hacia a la izquierda en X y hacia abajo en Y
    return {
        "class": int(row[0]),
        "xMin": xMin,
        "xCenter": row[1],
        "xMax": xMax,
        "yMin": yMin,
        "yCenter": row[2],
        "yMax": yMax,
    }


# * Calcula un unico IoU
def singleIou(predictedXY, trueXY):
    # areas
    predictedArea = abs(predictedXY["xMax"] - predictedXY["xMin"]) * abs(
        predictedXY["yMax"] - predictedXY["yMin"]
    )
    trueArea = abs(trueXY["xMax"] - trueXY["xMin"]) * abs(
        trueXY["yMax"] - trueXY["yMin"]
    )

    isIntersection = True
    if max(predictedXY["xMin"], trueXY["xMin"]) > min(
        predictedXY["xMax"], trueXY["xMax"]
    ) or max(predictedXY["yMin"], trueXY["yMin"]) > min(
        predictedXY["yMax"], trueXY["yMax"]
    ):
        isIntersection = False

    # Intersection
    if isIntersection:
        intersection = abs(
            min(predictedXY["xMax"], trueXY["xMax"])
            - max(predictedXY["xMin"], trueXY["xMin"]),
        ) * abs(
            min(predictedXY["yMax"], trueXY["yMax"])
            - max(predictedXY["yMin"], trueXY["yMin"])
        )
    else:
        intersection = 0

    # Union
    union = predictedArea + trueArea - intersection

    # IoU
    iou = intersection / union

    return iou


# * Devuelve dos diccionarios con coordenadas de nuestro espacio (x, y, z)
# * Un diccionario es para la predicciones y el otro para los verdaderos valores
def iou(filePredicted, fileTrue, imgPath):
    # filePredicted:    file path to the predicted labels
    # fileTrue:         file path to the true labels
    # imgPath:          path to the image

    try:
        with open(filePredicted) as csvFilePredicted, open(fileTrue) as csvFileTrue:
            csvReaderPredicted = csv.reader(csvFilePredicted, delimiter=" ")
            csvReaderTrue = csv.reader(csvFileTrue, delimiter=" ")

            lenPredicted = len(csvFilePredicted.readlines())
            lenTrue = len(csvFileTrue.readlines())

            if lenPredicted == 0 or lenTrue == 0:
                csvFilePredicted.close()
                csvFileTrue.close()
                return 0, lenPredicted, lenTrue

            csvFilePredicted.seek(0)
            csvFileTrue.seek(0)

            # TODO: Sacar las imagenes. Sólo se necesitan para plottear los puntos, cosa que no va a ser
            # TODO: necesaria cuando se verifique que todo funciona OK. Se puede dejar las posiciones relativas (entre 0 y 1).
            img = cv.imread(imgPath)

            iouListOfDicPredicted = []
            iouListOfDicTrue = []
            xCenterPredictedList = []
            yCenterPredictedList = []
            predictedXY = {}
            trueXY = {}
            lenTrue = 0

            for rowTrue in csvReaderTrue:
                rowTrue = [float(i) for i in rowTrue]

                iouList = []
                lenTrue += 1
                csvFilePredicted.seek(0)

                for rowPredicted in csvReaderPredicted:
                    rowPredicted = [float(i) for i in rowPredicted]

                    # para los casos donde hay menos predicciones que verdades
                    if len(rowPredicted) == 0:
                        break

                    predictedXY = yolov5ToXY(rowPredicted, img)
                    trueXY = yolov5ToXY(rowTrue, img)

                    # calculo un unico IoU
                    iou = singleIou(predictedXY, trueXY)

                    iouList.append(iou)

                    if lenTrue == 1:
                        xCenterPredictedList.append(predictedXY["xCenter"])
                        yCenterPredictedList.append(predictedXY["yCenter"])

                iouListOfDicPredicted.append(
                    {
                        "x": 0.0,
                        "y": 0.0,
                        "z": iouList,
                    }
                )

                # Si entra aca, es porque no hay ninguna prediccion
                if trueXY == {}:
                    return 0, None, None

                iouListOfDicTrue.append(
                    {"x": trueXY["xCenter"], "y": trueXY["yCenter"], "z": 1}
                )

            # guardo el punto medio de cada prediccion
            for idxTrue in range(lenTrue):
                for idxPredicted in range(lenPredicted):
                    iouListOfDicPredicted[idxTrue]["x"] = xCenterPredictedList[
                        idxPredicted
                    ]
                    iouListOfDicPredicted[idxTrue]["y"] = yCenterPredictedList[
                        idxPredicted
                    ]

        csvFilePredicted.close()
        csvFileTrue.close()
        return (
            {"iouPredicted": iouListOfDicPredicted, "iouTrue": iouListOfDicTrue},
            lenPredicted,
            lenTrue,
        )

    except:
        logging.exception("Failed to calculate IoU.")


# * Recibe los IoU calculados y devuelve:
# * Score total para una imagen
# * Cantidad de Predicciones
# * Cantidad de Verdades
def iouScore(filePredicted, fileTrue, imgPath):
    # ! IMPORTANTE: Chequear que no queden lineas vacias al final de los archivos de labels.
    # ! Por ahora eso no esta validado.

    # calculo los IoU de todas las predicciones contra todos los valores verdaderos
    iouResults, lenPredicted, lenTrue = iou(filePredicted, fileTrue, imgPath)

    if lenPredicted == 0 or lenTrue == 0:
        return 0, lenPredicted, lenTrue

    iouPredictedList = iouResults["iouPredicted"]
    iouTrueList = iouResults["iouTrue"]

    iouScoreArray = np.array([])

    for idxTrue, iouTrue in enumerate(iouTrueList):
        truePoint = np.array([iouTrue["x"], iouTrue["y"], iouTrue["z"]])
        minDistance = 1e10
        iouMax = 0

        for idxPrediction in range(lenPredicted):
            predictedPoint = np.array(
                [
                    iouPredictedList[idxTrue]["x"],
                    iouPredictedList[idxTrue]["y"],
                    iouPredictedList[idxTrue]["z"][idxPrediction],
                ]
            )

            euclideanDistance = np.linalg.norm(truePoint - predictedPoint)

            if euclideanDistance < minDistance:
                minDistance = euclideanDistance
                iouMax = iouPredictedList[idxTrue]["z"][idxPrediction]

        iouScoreArray = np.append(iouScoreArray, iouMax)

    # * Para calcular el IoU final, normalizo por la cantidad de elementos del vector mas largo.

    # igual cantidad de predicciones y verdades
    if lenPredicted == lenTrue:
        iouScore = np.sum(iouScoreArray) / lenTrue

    # menos predicciones que verdades
    elif lenPredicted < lenTrue:
        iouScoreArray = np.sort(iouScoreArray)
        iouScore = np.sum(iouScoreArray[1:]) / lenTrue

    # mas predicciones que verdades
    else:
        iouScoreArray = np.sort(iouScoreArray)
        iouScore = np.sum(iouScoreArray[1:]) / lenPredicted

    return iouScore, lenPredicted, lenTrue
