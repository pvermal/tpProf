from matplotlib import pyplot as plt
import os
from pathlib import Path
import utils.score_testing_data as scoreTestingData

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]
DATA = os.path.join(ROOT, r"data\testing-data-v3")

iouScoreList, qtyPredictedList, qtyTrueList = scoreTestingData.scoreTestingData(DATA)
print("iouScoreList", iouScoreList)
print("qtyPredictedList", qtyPredictedList)
print("qtyTrueList", qtyTrueList)
print(len(qtyPredictedList))

# * Gráficos con pruebas para sacar primeras conclusiones
# IoU for each image in the dataset
plt.figure()
plt.plot(iouScoreList)
plt.xlabel("Image n°")
plt.ylabel("IoU")

# Sorted IoU
plt.figure()
iouScoreList.sort()
plt.plot(iouScoreList)
plt.xlabel("Image n°")
plt.ylabel("IoU")

# Histogram
plt.figure()
plt.hist(iouScoreList)
plt.xlabel("IoU")
plt.ylabel("Quantity")

# Quantity of detections
plt.figure()
plt.plot(qtyPredictedList, label="Predicted")
plt.plot(qtyTrueList, label="True")
plt.xlabel("Image n°")
plt.ylabel("Quantity of detections")
plt.legend()
plt.show()
