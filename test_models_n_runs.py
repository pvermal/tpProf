from matplotlib import pyplot as plt
import numpy as np
import os
from pathlib import Path
import utils.score_testing_data as scoreTestingData

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]
#MODELS = os.path.join(ROOT, r"tpProf_cars_dataset")
MODELS = os.path.join(ROOT, r"runs/detect")
LABELS = os.path.join(ROOT, r"tpProf_cars_dataset/labels")
IMAGES = os.path.join(ROOT, r"tpProf_cars_dataset/images")

modelNamesRun = ["yolov5n", "yolov5s","yolov6n","yolov7-tiny","yolov8n"]

modelNamesRunMean = {}
modelNamesRunStd = {}

modelsResults = {}
for model in os.listdir(MODELS):
    # ignore the folders containing images and labels
    if model not in ["images", "labels"]:
        PREDICTIONS = os.path.join(MODELS, model, "labels")
        iouScoreList, qtyPredictedList, qtyTrueList = scoreTestingData.scoreTestingData(
            PREDICTIONS, LABELS, IMAGES
        )

        modelsResults.update(
            {
                model: {
                    "iouScoreList": iouScoreList,
                    "qtyPredictedList": qtyPredictedList,
                    "qtyTrueList": qtyTrueList,
                }
            }
        )

# average IoU for each model
averageIou = {}
print("Average IoU:")
for model, values in modelsResults.items():
    print("len: {}".format( len(np.asarray(values["iouScoreList"])) ) )
    mean = np.mean(np.asarray(values["iouScoreList"]))
    averageIou.update({model: mean})
    print("{}: {}".format(model, mean))

# * Gráficos con pruebas para sacar primeras conclusiones
# IoU for each image in the dataset
# plt.figure()
# plt.plot(iouScoreList)
# plt.xlabel("Image n°")
# plt.ylabel("IoU")

# Sorted IoU
plt.figure()
for model, values in modelsResults.items():
    values["iouScoreList"].sort()
    plt.plot(values["iouScoreList"], label=model)

plt.title("Sorted IoU for each model")
plt.xlabel("Image n°")
plt.ylabel("IoU")
plt.legend()

# Histogram
# plt.figure()
# plt.hist(iouScoreList)
# plt.xlabel("IoU")
# plt.ylabel("Quantity")

# Quantity of detections
plt.figure()
# puedo usar cualquier modelo para obtener la cantidad verdadera de detecciones
plt.plot(
    modelsResults["yolov5n_0"]["qtyTrueList"],
    label="True",
    color="black",
    linestyle="dashed",
)
for model, values in modelsResults.items():
    plt.plot(values["qtyPredictedList"], label=model)

plt.title("Quantity of detections for each model")
plt.xlabel("Image n°")
plt.ylabel("Quantity of detections")
plt.legend()
plt.show()
