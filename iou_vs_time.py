from matplotlib import pyplot as plt
import numpy as np

# models = {
#     "yolov5l.pt": {"timeMean": 458.74, "iouMean": 0.63},
#     "yolov5m.pt": {"timeMean": 257.86, "iouMean": 0.59},
#     "yolov5n.pt": {"timeMean": 61.4, "iouMean": 0.42},
#     "yolov5s.pt": {"timeMean": 117.94, "iouMean": 0.55},
#     "yolov6l.pt": {"timeMean": 577.87, "iouMean": 0.62},
#     "yolov6m.pt": {"timeMean": 364.22, "iouMean": 0.6},
#     "yolov6n.pt": {"timeMean": 73.75, "iouMean": 0.49},
#     "yolov6s.pt": {"timeMean": 179.44, "iouMean": 0.57},
#     "yolov7-tiny.pt": {"timeMean": 117.17, "iouMean": 0.5},
#     "yolov8n.pt": {"timeMean": 73.61, "iouMean": 0.47},
#     "yolov8s.pt": {"timeMean": 170.98, "iouMean": 0.57},
# }

# * Los modelos mas grandes fueron corridos una sóla vez:
# * - yolov5l.pt
# * - yolov5m.pt
# * - yolov6m.pt
# * - yolov6s.pt
# * - yolov8s.pt
# * Los siguientes modelos fueron corridos 100 veces, con
# * el objetivo de obtener una media y una varianza de los
# * tiempos de ejecución para nuestro set de datos:
# * - yolov5n.pt
# * - yolov5s.pt
# * - yolov6n.pt
# * - yolov7-tiny.pt
# * - yolov8n.pt
# * Hay algunos modelos que no pudimos correr por temas de
# * memoria de la Jetson.
models = {
    # corridos 1 vez
    "yolov5l.pt": {"timeMean": 458.74, "timeStd": 0, "iouMean": 0.63, "iouStd": 0},
    "yolov5m.pt": {"timeMean": 257.86, "timeStd": 0, "iouMean": 0.59, "iouStd": 0},
    "yolov6m.pt": {"timeMean": 364.22, "timeStd": 0, "iouMean": 0.6, "iouStd": 0},
    "yolov6s.pt": {"timeMean": 179.44, "timeStd": 0, "iouMean": 0.57, "iouStd": 0},
    "yolov8s.pt": {"timeMean": 170.98, "timeStd": 0, "iouMean": 0.57, "iouStd": 0},
    # corridos 100 veces
    "yolov5n.pt": {"timeMean": 49.868, "timeStd": 1.095, "iouMean": 0.42, "iouStd": 0},
    "yolov5s.pt": {"timeMean": 107.258, "timeStd": 1.208, "iouMean": 0.55, "iouStd": 0},
    "yolov6n.pt": {"timeMean": 73.866, "timeStd": 0.936, "iouMean": 0.51, "iouStd": 0},
    "yolov7-tiny.pt": {
        "timeMean": 109.504,
        "timeStd": 2.255,
        "iouMean": 0.51,
        "iouStd": 0,
    },
    "yolov8n.pt": {"timeMean": 63.404, "timeStd": 0.895, "iouMean": 0.47, "iouStd": 0},
}

plt.figure()
plt.grid()
for model in models.keys():
    if models[model]["timeStd"] != 0:
        plt.errorbar(
            models[model]["timeMean"],
            models[model]["iouMean"],
            xerr=models[model]["timeStd"]
            if models[model]["timeStd"] != 0
            else models[model]["timeStd"],
        )
        plt.scatter(
            models[model]["timeMean"],
            models[model]["iouMean"],
            label=model,
        )
plt.title("IoU vs Time")
plt.xlabel("Time [ms]")
plt.ylabel("IoU")
plt.legend(loc="lower right")
plt.show()
