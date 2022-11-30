from matplotlib import pyplot as plt
import os
from pathlib import Path
import utils.score_testing_data as scoreTestingData

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]
MODELS = os.path.join(ROOT, r"tpProf_cars_dataset")
LABELS = os.path.join(ROOT, r"tpProf_cars_dataset\labels")
IMAGES = os.path.join(ROOT, r"tpProf_cars_dataset\images")

modelsResults = {}
for model in os.listdir(MODELS):
    # ignore the folders containing images and labels
    if model in ["ssdMobileNetV3", "yolov4"]:
        PREDICTIONS = os.path.join(MODELS, model, "predictions")
        print("PREDICTIONS: ", PREDICTIONS)
        print("LABELS: ", LABELS)
        print("IMAGES: ", IMAGES)
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

print("modelsResults: ", modelsResults)
# print("iouScoreList", iouScoreList)
# print("qtyPredictedList", qtyPredictedList)
# print("qtyTrueList", qtyTrueList)
# print(len(qtyPredictedList))

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
    modelsResults["yolov4"]["qtyTrueList"],
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
