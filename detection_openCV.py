import argparse
import cv2
import logging
import numpy as np
import os
import time

from torch import rand


def generatePrediction(dataSetPath, model, dirName, showImages):
    """
    Crea una carpeta 'predictions' en el mismo directorio que dataSetPath con las
    predicciones que devuelve el modelo 'model' cargado con OpenCV.
    showImages sirve para mostrar una imagen con las predicciones, para ver
    resultados rápidos.
    """

    predictionsPath = os.path.join(dataSetPath, dirName)
    imagesPath = os.path.join(dataSetPath, "images")

    for filename in os.listdir(imagesPath):

        imgPath = os.path.join(imagesPath, filename)
        predPath = os.path.join(predictionsPath, filename).replace(".jpg", ".txt")

        # Creo la carpeta para las predicciones si no existe
        if not os.path.isdir(predictionsPath):
            os.mkdir(predictionsPath)

        img = cv2.imread(imgPath)
        height, width, _ = img.shape

        detectionInitialTime = time.time()
        classIds, scores, boxes = model.detect(img)
        detectionFinalTime = time.time()

        # * Aplico NMS para que no haya multiples detecciones para un mismo objeto
        boxIdsAfterNms = cv2.dnn.NMSBoxes(
            list(boxes), scores, score_threshold=0.25, nms_threshold=0.45
        )
        # reshape para que queden dimensiones [1, n] -> [[a, b, ...]]
        # tomo el elemento [0] para que devuelva una lista de la forma [a, b, ...]
        boxIdsAfterNms = np.reshape(boxIdsAfterNms, (1, -1))[0]
        scores = np.reshape(scores, (1, -1))[0]

        with open(predPath, "w") as f:

            logging.basicConfig(level=logging.INFO)
            logging.info("Creando predicciones para: %s", filename)

            # aca imprimir las predicciones: class x y w h
            for i in boxIdsAfterNms:
                f.write(
                    str(classIds[i][0])  # clase
                    + " "
                    + str(boxes[i][0] / width)  # x_center
                    + " "
                    + str(boxes[i][1] / height)  # y_center
                    + " "
                    + str(boxes[i][2] / height)  # height
                    + " "
                    + str(boxes[i][3] / width)  # width
                    + " "
                    + str(scores[i])  # confidence
                    + " "
                    + str(
                        (detectionFinalTime - detectionInitialTime) * 1e3
                    )  # inference time [ms]
                    + "\n"
                )

                # * Solo para mostrar plots rapidos
                if showImages:
                    (x, y, w, h) = boxes[i]

                    cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)

                    cv2.imshow("Frame", img)

            logging.info("Finalizado exitosamente: %s", filename)

            if showImages:
                key = cv2.waitKey(-1)
                if key == 27:
                    break

            f.close()


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", type=str, help="model weights file")
    parser.add_argument("--config", type=str, help="model configuration file")
    parser.add_argument("--dataSetPath", type=str, help="source")
    parser.add_argument("--dirName", type=str, help="name for the predctions directory")

    opt = parser.parse_args()
    print(opt)

    # definition of the model
    net = cv2.dnn.readNet(opt.weights, opt.config)
    model = cv2.dnn_DetectionModel(net)
    model.setInputParams(size=(320, 320), scale=1 / 255)
    generatePrediction(opt.dataSetPath, model, opt.dirName, showImages=False)
