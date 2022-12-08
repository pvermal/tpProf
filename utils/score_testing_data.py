import logging
import os
import utils.intersection_over_union as iou


def scoreTestingData(predictionsPath, labelsPath, imagesPath):
    """
    Calcula el IoU para cada imagen de una carpeta.
    Recibe: el path a la carpeta que contiene las carpetas de images, labels y predictions.
    Devuelve: tres listas. Una con los IoUs calculados para cada imagen, otra con la cantidad
    de predicciones realizadas por el modelo bajo prueba y una ultima con la cantidad de
    objetos que realmente hay en las imagenes.
    """

    iouScoreList = []
    qtyPredictedList = []
    qtyTrueList = []

    for filename in os.listdir(predictionsPath):
        filePredicted = os.path.join(predictionsPath, filename)
        fileTrue = os.path.join(labelsPath, filename)
        imgPath = os.path.join(imagesPath, filename).replace(".txt", ".jpg")

        if os.path.isfile(filePredicted) and os.path.isfile(fileTrue):
            logging.basicConfig(level=logging.INFO)
            logging.info("Calculando IoU para: %s", imgPath)

            iouScore, qtyPredicted, qtyTrue = iou.iouScore(
                filePredicted, fileTrue, imgPath
            )

            iouScoreList.append(iouScore)
            qtyPredictedList.append(qtyPredicted)
            qtyTrueList.append(qtyTrue)

    return iouScoreList, qtyPredictedList, qtyTrueList
